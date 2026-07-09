"""
Tests for the anonymous, opt-out usage telemetry.

These prove the three things that actually matter: telemetry is off when the user asks for it off
(by any of the three mechanisms), the beacon never carries anything personal, and a send failure can
never escape to the caller.
"""

import platform
import uuid
from pathlib import Path

import httpx
import pytest

from photo_tagger import __version__, telemetry
from photo_tagger.telemetry import RunInfo


@pytest.fixture(autouse=True)
def _isolate_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    Point the state directory at a tmp location and clear the opt-out env vars.

    Without this a developer's real ``~/.local/state/photo-tagger`` (or a ``DO_NOT_TRACK`` in their
    shell) would leak into the tests and make the opt-out assertions flaky.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.delenv("PHOTO_TAGGER_NO_TELEMETRY", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)


def _sample_run() -> RunInfo:
    """Build a representative run for payload/emit tests."""
    return RunInfo(
        interface="cli",
        provider="lmstudio",
        model="qwen/qwen3-vl-30b",
        batch_size=12,
        duration_seconds=4.5,
    )


# --- opt-out precedence ------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["1", "true", "YES", "On"])
def test_env_opt_out_accepts_truthy_spellings(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """Any accepted truthy spelling of our env var opts out."""
    monkeypatch.setenv("PHOTO_TAGGER_NO_TELEMETRY", value)
    assert telemetry.env_opt_out() is True


@pytest.mark.parametrize("value", ["0", "", "false", "no"])
def test_env_opt_out_ignores_non_truthy(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """A falsy or empty value is treated as 'not set' and does not opt out."""
    monkeypatch.setenv("PHOTO_TAGGER_NO_TELEMETRY", value)
    assert telemetry.env_opt_out() is False


def test_env_opt_out_honors_do_not_track(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cross-tool DO_NOT_TRACK standard also opts out."""
    monkeypatch.setenv("DO_NOT_TRACK", "1")
    assert telemetry.env_opt_out() is True


def test_should_send_true_only_when_enabled_and_no_env_optout() -> None:
    """With nothing opting out, the merged flag/config bool decides."""
    assert telemetry.should_send(config_enabled=True) is True
    assert telemetry.should_send(config_enabled=False) is False


def test_env_optout_overrides_enabled_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """The environment wins even when flag/config asked to keep telemetry on."""
    monkeypatch.setenv("PHOTO_TAGGER_NO_TELEMETRY", "1")
    assert telemetry.should_send(config_enabled=True) is False


# --- install id --------------------------------------------------------------------------------


def test_install_id_is_stable_and_persisted(tmp_path: Path) -> None:
    """The id is a valid UUID, written to disk, and identical across calls."""
    first = telemetry.install_id()
    uuid.UUID(first)  # raises if not a valid UUID
    id_file = tmp_path / "state" / "photo-tagger" / "install-id"
    assert id_file.is_file()
    assert telemetry.install_id() == first


def test_install_id_regenerates_when_corrupt(tmp_path: Path) -> None:
    """A garbage id file is replaced with a fresh valid UUID."""
    id_file = tmp_path / "state" / "photo-tagger" / "install-id"
    id_file.parent.mkdir(parents=True)
    id_file.write_text("not-a-uuid")
    fresh = telemetry.install_id()
    uuid.UUID(fresh)
    assert fresh != "not-a-uuid"


def test_install_id_degrades_when_state_dir_unwritable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """If the state dir cannot be created, a valid ephemeral id is still returned."""
    blocker = tmp_path / "blocker"
    blocker.write_text("i am a file, not a directory")
    monkeypatch.setenv("XDG_STATE_HOME", str(blocker))
    fresh = telemetry.install_id()
    uuid.UUID(fresh)  # still usable, just not persisted


# --- first-run notice --------------------------------------------------------------------------


def test_first_run_notice_shows_once() -> None:
    """The notice text is returned on the first call and suppressed thereafter."""
    assert telemetry.first_run_notice() == telemetry.FIRST_RUN_NOTICE
    assert telemetry.first_run_notice() is None


def test_first_run_notice_still_shows_when_marker_unwritable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """If the marker file cannot be written, the notice is still returned (just not suppressed)."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    monkeypatch.setenv("XDG_STATE_HOME", str(blocker))
    assert telemetry.first_run_notice() == telemetry.FIRST_RUN_NOTICE


# --- GUI preference ----------------------------------------------------------------------------


def test_gui_pref_round_trips() -> None:
    """An unset pref reads as None; a written choice reads back unchanged."""
    assert telemetry.read_gui_pref() is None
    telemetry.write_gui_pref(enabled=False)
    assert telemetry.read_gui_pref() is False
    telemetry.write_gui_pref(enabled=True)
    assert telemetry.read_gui_pref() is True


def test_gui_pref_unrecognized_reads_as_none(tmp_path: Path) -> None:
    """A garbled pref file is treated as 'no choice', not a crash."""
    pref = tmp_path / "state" / "photo-tagger" / "gui-telemetry"
    pref.parent.mkdir(parents=True)
    pref.write_text("maybe")
    assert telemetry.read_gui_pref() is None


def test_gui_pref_write_degrades_when_unwritable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """If the state dir cannot be created, persisting the pref is swallowed, never raised."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    monkeypatch.setenv("XDG_STATE_HOME", str(blocker))
    telemetry.write_gui_pref(enabled=False)  # must not raise


# --- payload contents (the privacy guarantee) --------------------------------------------------


_EXPECTED_PAYLOAD_KEYS = {
    "schema_version",
    "install_id",
    "app_version",
    "interface",
    "provider",
    "model",
    "batch_size",
    "duration_seconds",
    "arch",
    "os",
    "os_release",
    "python_version",
}


def test_build_payload_has_exactly_the_allowlisted_keys() -> None:
    """The beacon body is a closed set: no field can sneak in unnoticed."""
    payload = telemetry.build_payload(_sample_run())
    assert set(payload) == _EXPECTED_PAYLOAD_KEYS


def test_build_payload_carries_run_and_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caller-supplied run facts and the resolved platform values land in the payload."""
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    payload = telemetry.build_payload(_sample_run())
    assert payload["interface"] == "cli"
    assert payload["model"] == "qwen/qwen3-vl-30b"
    assert payload["batch_size"] == 12  # noqa: PLR2004 - the sample run's count
    assert payload["arch"] == "arm64"
    assert payload["os"] == "Darwin"
    assert payload["app_version"] == __version__
    assert payload["schema_version"] == telemetry.SCHEMA_VERSION


def test_build_payload_leaks_no_paths_or_content() -> None:
    """No payload value resembles a filesystem path; install id is the only opaque token."""
    payload = telemetry.build_payload(_sample_run())
    for key, value in payload.items():
        if key in {"install_id", "model"}:  # uuid and the (slashed) model id are expected
            continue
        assert "/" not in str(value)
        assert "\\" not in str(value)
    uuid.UUID(str(payload["install_id"]))


# --- emit (sending) ----------------------------------------------------------------------------


def test_emit_returns_none_and_sends_nothing_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disabled config short-circuits before any network attempt."""
    calls: list[object] = []
    monkeypatch.setattr(httpx, "post", lambda *a, **k: calls.append((a, k)))
    assert telemetry.emit(_sample_run(), enabled=False) is None
    assert calls == []


def test_emit_posts_the_payload_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """An enabled run POSTs the built payload to the collector endpoint."""
    captured: dict[str, object] = {}

    def fake_post(url: str, *, json: dict[str, object], timeout: float) -> object:
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return object()

    monkeypatch.setattr(httpx, "post", fake_post)
    thread = telemetry.emit(_sample_run(), enabled=True, block=True)
    assert thread is not None
    assert captured["url"] == telemetry.ENDPOINT
    assert captured["json"]["interface"] == "cli"  # type: ignore[index]


def test_emit_non_blocking_returns_started_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default non-blocking path returns the worker thread without waiting on it."""
    monkeypatch.setattr(httpx, "post", lambda *_a, **_k: object())
    thread = telemetry.emit(_sample_run(), enabled=True)
    assert thread is not None
    thread.join(timeout=2.0)  # tidy up so the post lands before the test ends


def test_emit_returns_none_when_payload_build_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure while assembling the payload is swallowed and nothing is sent."""

    def boom(_run: RunInfo) -> dict[str, object]:
        msg = "platform probe blew up"
        raise RuntimeError(msg)

    posted: list[object] = []
    monkeypatch.setattr(telemetry, "build_payload", boom)
    monkeypatch.setattr(httpx, "post", lambda *a, **k: posted.append((a, k)))
    assert telemetry.emit(_sample_run(), enabled=True, block=True) is None
    assert posted == []


def test_emit_swallows_send_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A network failure inside the beacon never propagates to the caller."""

    def boom(*_a: object, **_k: object) -> object:
        msg = "network down"
        raise httpx.ConnectError(msg)

    monkeypatch.setattr(httpx, "post", boom)
    # block=True joins the worker; the test passes simply by not raising.
    telemetry.emit(_sample_run(), enabled=True, block=True)
