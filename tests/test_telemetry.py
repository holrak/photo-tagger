"""
Tests for the anonymous, opt-out usage telemetry.

These prove the three things that actually matter: telemetry is off when the user asks for it off
(by any of the three mechanisms), the beacon never carries anything personal, and a send failure can
never escape to the caller.
"""

import platform
import threading
import time
import uuid
from pathlib import Path

import httpx2
import pytest

from photo_tagger import __version__, telemetry
from photo_tagger.hardware import HardwareInfo
from photo_tagger.telemetry import RunInfo


_FAKE_HARDWARE = HardwareInfo(cpu="Apple M3 Pro", gpu="Apple M3 Pro", cpu_count=12, memory_gb=36)


@pytest.fixture(autouse=True)
def _isolate_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """
    Point the state directory at a tmp location, stub hardware, and clear the opt-out env vars.

    Without this a developer's real ``~/.local/state/photo-tagger`` (or a ``DO_NOT_TRACK`` in their
    shell) would leak into the tests and make the opt-out assertions flaky; the hardware stub keeps
    payload tests from shelling out to real probes.
    """
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.delenv("PHOTO_TAGGER_NO_TELEMETRY", raising=False)
    monkeypatch.delenv("DO_NOT_TRACK", raising=False)
    monkeypatch.setattr(telemetry, "hardware_info", lambda _path: _FAKE_HARDWARE)
    monkeypatch.setattr(telemetry, "_crash_beacons_sent", 0)


def _sample_run() -> RunInfo:
    """Build a representative run for payload/emit tests."""
    return RunInfo(
        interface="cli",
        provider="lmstudio",
        model="qwen/qwen3-vl-30b",
        batch_size=12,
        duration_seconds=4.5,
        output_language="English",
        ui_language="en",
        file_types="cr3,jpg",
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


def test_first_run_notice_returns_when_disclosure_version_changes(tmp_path: Path) -> None:
    """
    A marker from an older disclosure version does not suppress the updated notice.

    v1 installs wrote an empty marker; v2 added hardware and crash reports to the disclosure, and
    those users must see the expanded text once.
    """
    marker = tmp_path / "state" / "photo-tagger" / "telemetry-notice-shown"
    marker.parent.mkdir(parents=True)
    marker.write_text("", encoding="utf-8")  # what a v1 install left behind
    assert telemetry.first_run_notice() == telemetry.FIRST_RUN_NOTICE
    assert telemetry.first_run_notice() is None
    assert marker.read_text(encoding="utf-8").strip() == str(telemetry.NOTICE_VERSION)


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


_PLATFORM_KEYS = {
    "schema_version",
    "install_id",
    "app_version",
    "arch",
    "os",
    "os_release",
    "python_version",
    "cpu",
    "gpu",
    "cpu_count",
    "memory_gb",
}

_EXPECTED_PAYLOAD_KEYS = _PLATFORM_KEYS | {
    "event",
    "interface",
    "provider",
    "model",
    "batch_size",
    "duration_seconds",
    "output_language",
    "ui_language",
    "file_types",
    "success_count",
    "failure_count",
    "cache_hits",
    "retry_successes",
    "workers",
    "total_tokens",
    "inference_seconds",
    "dry_run",
    "failure_kinds",
}

_EXPECTED_CRASH_KEYS = _PLATFORM_KEYS | {
    "event",
    "interface",
    "exception_type",
    "crash_location",
    "crash_frames",
}


def test_build_payload_has_exactly_the_allowlisted_keys() -> None:
    """The beacon body is a closed set: no field can sneak in unnoticed."""
    payload = telemetry.build_payload(_sample_run())
    assert set(payload) == _EXPECTED_PAYLOAD_KEYS
    assert payload["event"] == "run"


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


def test_build_payload_carries_language_and_file_types() -> None:
    """The language and file-type run facts reach the beacon body verbatim."""
    payload = telemetry.build_payload(_sample_run())
    assert payload["output_language"] == "English"
    assert payload["ui_language"] == "en"
    assert payload["file_types"] == "cr3,jpg"


def test_build_payload_carries_hardware_and_outcome_fields() -> None:
    """Hardware facts and per-run outcome counters land in the beacon body."""
    run = RunInfo(
        interface="cli",
        provider="lmstudio",
        model="m",
        batch_size=10,
        duration_seconds=60.0,
        output_language="English",
        ui_language="en",
        file_types="cr3",
        success_count=9,
        failure_count=1,
        cache_hits=3,
        retry_successes=2,
        workers=4,
        total_tokens=12345,
        inference_seconds=41.5,
        dry_run=True,
    )
    payload = telemetry.build_payload(run)
    assert payload["cpu"] == "Apple M3 Pro"
    assert payload["gpu"] == "Apple M3 Pro"
    assert payload["cpu_count"] == 12  # noqa: PLR2004 - the stubbed hardware fixture
    assert payload["memory_gb"] == 36  # noqa: PLR2004 - the stubbed hardware fixture
    assert payload["success_count"] == 9  # noqa: PLR2004 - the sample run's count
    assert payload["failure_count"] == 1
    assert payload["cache_hits"] == 3  # noqa: PLR2004 - the sample run's count
    assert payload["retry_successes"] == 2  # noqa: PLR2004 - the sample run's count
    assert payload["workers"] == 4  # noqa: PLR2004 - the sample run's count
    assert payload["total_tokens"] == 12345  # noqa: PLR2004 - the sample run's count
    assert payload["inference_seconds"] == 41.5  # noqa: PLR2004 - the sample run's value
    assert payload["dry_run"] == 1


def test_failure_kinds_summary_is_compact_sorted_and_skips_zeros() -> None:
    """The per-run failure buckets encode as a stable "kind:count" list."""
    assert telemetry.failure_kinds_summary({}) == ""
    assert (
        telemetry.failure_kinds_summary({"timeout": 3, "other": 1, "connection": 0})
        == "other:1,timeout:3"
    )


def test_file_types_summary_is_sorted_lowercased_and_deduplicated() -> None:
    """Extensions collapse to a sorted, lowercased, dot-stripped, comma-joined set."""
    paths = [Path("/photos/a.CR3"), Path("/photos/b.jpg"), Path("/photos/c.cr3")]
    assert telemetry.file_types_summary(paths) == "cr3,jpg"


def test_file_types_summary_reveals_no_filenames() -> None:
    """Only extensions leave the machine: the summary carries no path or filename."""
    summary = telemetry.file_types_summary([Path("/private/vacation-2026/IMG_0001.DNG")])
    assert summary == "dng"


def test_file_types_summary_handles_empty_and_no_suffix() -> None:
    """No paths, or paths without a suffix, yield an empty string rather than a stray comma."""
    assert telemetry.file_types_summary([]) == ""
    assert telemetry.file_types_summary([Path("/photos/README")]) == ""


# --- crash reporting ---------------------------------------------------------------------------


def _package_exception() -> BaseException:
    """Raise (and catch) an error whose traceback passes through a real photo_tagger module."""
    from photo_tagger.keywords import parse_hierarchical_keyword  # noqa: PLC0415 - test-local

    try:
        # Deliberate misuse so a real traceback through a photo_tagger module exists.
        parse_hierarchical_keyword(None)  # type: ignore[arg-type]
    except AttributeError as exc:
        return exc
    msg = "expected parse_hierarchical_keyword(None) to raise"  # pragma: no cover
    raise AssertionError(msg)  # pragma: no cover


def test_crash_summary_reports_type_and_in_package_location() -> None:
    """The summary names the exception class and the deepest photo_tagger frame."""
    name, location, frames = telemetry.crash_summary(_package_exception())
    assert name == "AttributeError"
    assert location.startswith("keywords:parse_hierarchical_keyword:")
    assert location in frames


def test_crash_summary_skips_frames_outside_the_package() -> None:
    """Test-file (and stdlib) frames never appear; only photo_tagger code coordinates do."""
    _, _, frames = telemetry.crash_summary(_package_exception())
    assert "test_telemetry" not in frames


def test_crash_summary_never_carries_the_message() -> None:
    """
    The exception message is dropped by construction.

    Messages routinely embed file paths ("could not open /Users/x/IMG.CR3"), so no part of the
    summary may contain it.
    """
    try:
        msg = "/Users/someone/secret/IMG_0001.CR3 could not be read"
        raise RuntimeError(msg)  # noqa: TRY301 - the test needs a real traceback.
    except RuntimeError as exc:
        summary = telemetry.crash_summary(exc)
    assert all("/Users/" not in part and "IMG_0001" not in part for part in summary)
    # Raised in this test file, outside the package: no code coordinates at all.
    assert summary == ("RuntimeError", "", "")


def test_build_crash_payload_has_exactly_the_allowlisted_keys() -> None:
    """The crash beacon is a closed set too, with the crash event marker."""
    payload = telemetry.build_crash_payload(_package_exception(), interface="cli")
    assert set(payload) == _EXPECTED_CRASH_KEYS
    assert payload["event"] == "crash"
    assert payload["exception_type"] == "AttributeError"


def test_emit_crash_posts_when_enabled_and_respects_optout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An enabled crash beacon POSTs; the config opt-out short-circuits before any network."""
    posted: list[dict[str, object]] = []
    monkeypatch.setattr(httpx2, "post", lambda _url, *, json, timeout: posted.append(json))  # noqa: ARG005 - httpx2.post signature

    assert telemetry.emit_crash(_package_exception(), interface="cli", enabled=False) is None
    assert posted == []

    thread = telemetry.emit_crash(_package_exception(), interface="cli", enabled=True)
    assert thread is not None
    assert posted[0]["event"] == "crash"


def test_emit_crash_is_capped_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """A crash loop cannot spam the collector: beacons stop after the per-process cap."""
    posted: list[object] = []
    monkeypatch.setattr(httpx2, "post", lambda *_a, **k: posted.append(k))
    exc = _package_exception()
    sent = [telemetry.emit_crash(exc, interface="gui", enabled=True) for _ in range(5)]
    assert sum(thread is not None for thread in sent) == 3  # noqa: PLR2004 - the cap
    assert len(posted) == 3  # noqa: PLR2004 - the cap


# --- emit (sending) ----------------------------------------------------------------------------


def test_emit_returns_none_and_sends_nothing_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disabled config short-circuits before any network attempt."""
    calls: list[object] = []
    monkeypatch.setattr(httpx2, "post", lambda *a, **k: calls.append((a, k)))
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

    monkeypatch.setattr(httpx2, "post", fake_post)
    thread = telemetry.emit(_sample_run(), enabled=True, block=True)
    assert thread is not None
    assert captured["url"] == telemetry.ENDPOINT
    assert captured["json"]["interface"] == "cli"  # type: ignore[index]


def test_emit_non_blocking_returns_started_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default non-blocking path returns the worker thread without waiting on it."""
    monkeypatch.setattr(httpx2, "post", lambda *_a, **_k: object())
    thread = telemetry.emit(_sample_run(), enabled=True)
    assert thread is not None
    thread.join(timeout=2.0)  # tidy up so the post lands before the test ends


def test_emit_swallows_payload_build_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    A failure while assembling the payload sends nothing and never escapes to the caller.

    The builder now runs inside the background thread (see
    test_emit_does_not_block_the_caller_while_building), so emit() itself cannot know in advance
    whether the build will fail; it still returns the started thread, not None.
    """

    def boom(_run: RunInfo) -> dict[str, object]:
        msg = "platform probe blew up"
        raise RuntimeError(msg)

    posted: list[object] = []
    monkeypatch.setattr(telemetry, "build_payload", boom)
    monkeypatch.setattr(httpx2, "post", lambda *a, **k: posted.append((a, k)))
    thread = telemetry.emit(_sample_run(), enabled=True, block=True)
    assert thread is not None
    assert posted == []


def test_emit_does_not_block_the_caller_while_building(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Building the payload never happens on the caller's thread.

    build_payload can shell out to probe hardware on a cache miss; the non-blocking path must return
    immediately regardless of how long that takes, or a cold hardware-probe cache would stall the
    tagging run right as it is about to exit.
    """
    finished = threading.Event()

    def slow_build(_run: RunInfo) -> dict[str, object]:
        time.sleep(0.2)
        finished.set()
        return {}

    monkeypatch.setattr(telemetry, "build_payload", slow_build)
    monkeypatch.setattr(httpx2, "post", lambda *_a, **_k: object())

    start = time.monotonic()
    thread = telemetry.emit(_sample_run(), enabled=True, block=False)
    elapsed = time.monotonic() - start

    assert thread is not None
    assert elapsed < 0.1  # noqa: PLR2004 - well under slow_build's 0.2s sleep
    assert not finished.is_set()  # the builder is still running on the background thread
    thread.join(timeout=2.0)
    assert finished.is_set()


def test_emit_swallows_send_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A network failure inside the beacon never propagates to the caller."""

    def boom(*_a: object, **_k: object) -> object:
        msg = "network down"
        raise httpx2.ConnectError(msg)

    monkeypatch.setattr(httpx2, "post", boom)
    # block=True joins the worker; the test passes simply by not raising.
    telemetry.emit(_sample_run(), enabled=True, block=True)
