"""
Anonymous, opt-out usage telemetry.

One small JSON beacon is sent at the end of a run so the project can see which models and platforms
people actually use. It is deliberately minimal and privacy-preserving:

- **No personal or content data.** File paths, filenames, photo pixels, generated tags, titles,
  descriptions, prompts, and API keys are never collected. See :func:`build_payload` for the exact,
  closed set of fields that leave the machine.
- **Opt-out, three ways.** Set ``PHOTO_TAGGER_NO_TELEMETRY=1`` (or the cross-tool
  ``DO_NOT_TRACK=1``) in the environment, pass ``--no-telemetry``, or put ``enabled = false`` under
  ``[telemetry]`` in the config file. The environment variables win over everything else.
- **Never in the way.** Sending happens on a daemon thread with a short timeout, and every failure
  is swallowed. Telemetry must never crash, slow, or block a tagging run.

The collector is our own Cloudflare Worker at ``telemetry.tagger.photo`` (its source lives in the
``telemetry/`` directory of this repo). No third-party analytics service is involved.

The per-machine ``install_id`` is a random UUID generated once and stored in the user state
directory. It is **not** derived from any hardware identifier, so it cannot be used to fingerprint a
machine; it only lets repeated runs from the same install be counted as one active user over time.
"""

import os
import platform
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx
from loguru import logger

from photo_tagger import __version__


# The collector endpoint (our own Cloudflare Worker). Kept as a module constant so tests can patch
# it and so there is a single place to repoint it.
ENDPOINT = "https://telemetry.tagger.photo"

# Bumped only when the payload shape changes, so the Worker can branch on it if it ever needs to.
SCHEMA_VERSION = 1

# Environment opt-out switches. Our own variable plus the de-facto cross-tool standard from
# consoledonottrack.com, which many CLIs already honor.
_ENV_DISABLE = "PHOTO_TAGGER_NO_TELEMETRY"
_ENV_DO_NOT_TRACK = "DO_NOT_TRACK"

# Truthy spellings accepted for the opt-out variables. Anything else (including "0" and "") is
# treated as "not set", so a stray empty value never silently disables telemetry.
_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Network call budget. A healthy beacon is a few tens of milliseconds; this only bounds the bad case
# (no network, DNS hang) so the call cannot wedge process exit.
_SEND_TIMEOUT_SECONDS = 2.0
# How long a blocking caller (the CLI, which exits right after) waits for the beacon to flush before
# giving up and letting the daemon thread be abandoned at interpreter exit.
_FLUSH_TIMEOUT_SECONDS = 2.5

# State files. The install id, the "first-run notice already shown" marker, and the GUI's persisted
# on/off choice live under the user state directory, mirroring config_file.py's XDG-style location.
_INSTALL_ID_FILE = "install-id"
_NOTICE_MARKER_FILE = "telemetry-notice-shown"
_GUI_PREF_FILE = "gui-telemetry"

# Shown once, to stderr, on the first run telemetry is active. Opt-out tools are expected to
# disclose up front; this is that disclosure.
FIRST_RUN_NOTICE = (
    "photo-tagger collects anonymous usage stats (model name, batch size, OS, CPU arch, timing) to "
    "guide development.\n"
    "No photos, file paths, filenames, tags, or personal data are ever sent. See the Telemetry "
    "section of the README.\n"
    "Disable it any time with --no-telemetry, PHOTO_TAGGER_NO_TELEMETRY=1, or [telemetry] "
    "enabled = false in your config."
)


@dataclass(slots=True, frozen=True)
class RunInfo:
    """
    The caller-supplied facts about one run.

    Everything else in the beacon (platform, version, install id) is filled in by
    :func:`build_payload`, so callers only describe what they alone know.
    """

    interface: str  # "cli" or "gui"
    provider: str
    model: str
    batch_size: int
    duration_seconds: float


def _is_truthy(value: str | None) -> bool:
    """Return True when *value* is one of the accepted opt-out spellings."""
    return value is not None and value.strip().lower() in _TRUTHY


def env_opt_out() -> bool:
    """Return True when either opt-out environment variable is set to a truthy value."""
    return _is_truthy(os.getenv(_ENV_DISABLE)) or _is_truthy(os.getenv(_ENV_DO_NOT_TRACK))


def should_send(*, config_enabled: bool) -> bool:
    """
    Resolve whether to send, given the already-merged flag/config value.

    The CLI folds ``--no-telemetry`` and the ``[telemetry]`` config table into a single
    *config_enabled* bool before calling this. An environment opt-out always wins, so it can disable
    telemetry even when the config or flag asked to keep it on.
    """
    return config_enabled and not env_opt_out()


def _state_dir() -> Path:
    """
    Return the directory for telemetry state, honoring ``XDG_STATE_HOME``.

    Falls back to ``~/.local/state/photo-tagger``. This is state, not config, so it lives apart from
    the TOML config file the user edits by hand.
    """
    base = os.getenv("XDG_STATE_HOME")
    root = Path(base) if base else Path.home() / ".local" / "state"
    return root / "photo-tagger"


def install_id() -> str:
    """
    Return this install's stable anonymous id, creating it on first use.

    The id is a random UUID4 persisted under the state directory. If it cannot be read or written
    (read-only home, permission denied), a fresh ephemeral id is returned instead so the run is
    still counted; only the across-runs "same install" linkage is lost in that rare case.
    """
    path = _state_dir() / _INSTALL_ID_FILE
    try:
        existing = path.read_text(encoding="utf-8").strip()
        return str(uuid.UUID(existing))
    except OSError, ValueError:
        # Missing, empty, or corrupt: mint a new one below.
        pass

    new_id = str(uuid.uuid4())
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_id + "\n", encoding="utf-8")
    except OSError as exc:
        logger.debug("telemetry_install_id_persist_failed", error=str(exc))
    return new_id


def first_run_notice() -> str | None:
    """
    Return :data:`FIRST_RUN_NOTICE` the first time telemetry is active, else ``None``.

    A marker file records that the notice has been shown so it appears only once. The marker is
    written before returning, so two calls in one process do not both show it. If the marker cannot
    be written the notice may reappear on a later run, which is harmless.
    """
    marker = _state_dir() / _NOTICE_MARKER_FILE
    if marker.exists():
        return None
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("", encoding="utf-8")
    except OSError as exc:
        logger.debug("telemetry_notice_marker_failed", error=str(exc))
    return FIRST_RUN_NOTICE


def read_gui_pref() -> bool | None:
    """
    Return the GUI's persisted telemetry choice, or ``None`` if the user never made one.

    The GUI's Settings toggle writes this so the choice survives restarts. ``None`` means "no
    explicit choice", so the GUI falls back to the config-file default. Anything unreadable or
    unrecognized is treated as no choice.
    """
    try:
        text = (_state_dir() / _GUI_PREF_FILE).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if text == "1":
        return True
    if text == "0":
        return False
    return None


def write_gui_pref(*, enabled: bool) -> None:
    """Persist the GUI's telemetry choice so the Settings toggle sticks across restarts."""
    path = _state_dir() / _GUI_PREF_FILE
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("1" if enabled else "0", encoding="utf-8")
    except OSError as exc:
        logger.debug("telemetry_gui_pref_persist_failed", error=str(exc))


def build_payload(run: RunInfo) -> dict[str, object]:
    """
    Build the exact beacon body from *run* plus locally-derived platform facts.

    This is the single source of truth for what leaves the machine. Every value here is either a
    bounded enum (interface, provider), a coarse platform string, a count, a duration, or the random
    install id. Nothing derived from the user's photos, paths, or identity appears, by construction.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "install_id": install_id(),
        "app_version": __version__,
        "interface": run.interface,
        "provider": run.provider,
        "model": run.model,
        "batch_size": run.batch_size,
        "duration_seconds": round(run.duration_seconds, 3),
        "arch": platform.machine(),  # e.g. "arm64", "x86_64", "aarch64"
        "os": platform.system(),  # "Darwin", "Linux", "Windows"
        "os_release": platform.release(),
        "python_version": platform.python_version(),
    }


def _safe_post(payload: dict[str, object]) -> None:
    """POST *payload* to the collector, swallowing every error (telemetry is never load-bearing)."""
    try:
        httpx.post(ENDPOINT, json=payload, timeout=_SEND_TIMEOUT_SECONDS)
    except Exception as exc:  # noqa: BLE001 - a beacon must never surface a failure to the user.
        logger.debug("telemetry_send_failed", error=str(exc))


def emit(run: RunInfo, *, enabled: bool, block: bool = False) -> threading.Thread | None:
    """
    Fire one telemetry beacon for *run* on a background daemon thread.

    Returns the thread (or ``None`` when telemetry is disabled or the payload cannot be built). Pass
    ``block=True`` to wait up to :data:`_FLUSH_TIMEOUT_SECONDS` for delivery; the CLI uses this
    since it exits immediately afterwards and would otherwise lose the in-flight beacon. The GUI
    leaves it ``False`` so closing the window never stalls.
    """
    if not should_send(config_enabled=enabled):
        return None
    try:
        payload = build_payload(run)
    except Exception as exc:  # noqa: BLE001 - building the payload must not crash the caller.
        logger.debug("telemetry_payload_failed", error=str(exc))
        return None

    thread = threading.Thread(
        target=_safe_post,
        args=(payload,),
        name="photo-tagger-telemetry",
        daemon=True,
    )
    thread.start()
    if block:
        thread.join(timeout=_FLUSH_TIMEOUT_SECONDS)
    return thread
