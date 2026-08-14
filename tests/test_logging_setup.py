"""Tests for setup_logging."""

import sys
from pathlib import Path

import pytest
from loguru import logger

from photo_tagger.logging_setup import setup_logging


def test_setup_logging_creates_log_folder(tmp_path: Path) -> None:
    """File logging creates the log folder and writes at least one file."""
    folder = tmp_path / "logs"
    setup_logging(file_log_level="DEBUG", console_log_level="OFF", log_folder=folder)
    logger.info("hello")
    logger.complete()
    assert folder.exists()
    files = list(folder.glob("*-photo_tagger.log"))
    assert files, "expected a log file to be created"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="0o600 is a POSIX permission concept; Windows ACLs do not map onto it",
)
def test_setup_logging_creates_the_log_file_owner_only(tmp_path: Path) -> None:
    """
    The log file is created 0600 so DEBUG-level records are not world-readable.

    DEBUG logs carry the `extra` context (file paths, provider URLs); on a shared multi-user
    machine, default OS permissions would let any other local user read them.
    """
    folder = tmp_path / "logs"
    setup_logging(file_log_level="DEBUG", console_log_level="OFF", log_folder=folder)
    logger.info("hello")
    logger.complete()
    log_file = next(iter(folder.glob("*-photo_tagger.log")))
    expected_mode = 0o600
    assert log_file.stat().st_mode & 0o777 == expected_mode


def test_setup_logging_survives_chmod_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hardening the log file's permissions is best-effort: a failure must not abort setup."""

    def _exploding_chmod(self: Path, *_a: object, **_k: object) -> None:
        msg = "unsupported filesystem"
        raise OSError(msg)

    monkeypatch.setattr(Path, "chmod", _exploding_chmod)
    folder = tmp_path / "logs"
    setup_logging(file_log_level="DEBUG", console_log_level="OFF", log_folder=folder)  # no raise
    logger.info("hello")
    logger.complete()
    assert list(folder.glob("*-photo_tagger.log"))


def test_setup_logging_writes_json_records(tmp_path: Path) -> None:
    """File logs are NDJSON: one parseable record per line, extra fields included."""
    import json  # noqa: PLC0415 - test-local parser.

    folder = tmp_path / "logs"
    setup_logging(file_log_level="DEBUG", console_log_level="OFF", log_folder=folder)
    logger.info("hello", answer=42)
    logger.complete()
    log_file = next(iter(folder.glob("*-photo_tagger.log")))
    record = json.loads(log_file.read_text(encoding="utf-8").splitlines()[0])
    assert record["record"]["message"] == "hello"
    assert record["record"]["extra"]["answer"] == 42  # noqa: PLR2004  # fixture value


def test_module_import_caps_the_console_at_info() -> None:
    """
    Importing logging_setup replaces loguru's DEBUG default sink with an INFO console.

    Runs in a subprocess because the import side effect fires only on first import, which in the
    test process happened long ago (and later tests reconfigure the sinks anyway).
    """
    import subprocess  # noqa: PLC0415 - test-local, fixed argv.
    import sys  # noqa: PLC0415

    code = (
        "from loguru import logger\n"
        "import photo_tagger.logging_setup\n"
        "logger.debug('hidden-debug')\n"
        "logger.info('visible-info')\n"
    )
    result = subprocess.run(  # noqa: S603 - fixed argv, our own interpreter.
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "hidden-debug" not in result.stderr
    assert "visible-info" in result.stderr


def test_setup_logging_off_disables_handlers(tmp_path: Path) -> None:
    """OFF on both sinks leaves loguru with no handlers attached."""
    folder = tmp_path / "logs"
    setup_logging(file_log_level="OFF", console_log_level="OFF", log_folder=folder)
    # No file handler => the folder is never created.
    assert not folder.exists()


def test_setup_logging_adds_console_handler_without_file(tmp_path: Path) -> None:
    """A console level other than OFF attaches a stderr sink and skips the log folder."""
    folder = tmp_path / "logs"
    setup_logging(file_log_level="OFF", console_log_level="INFO", log_folder=folder)
    logger.info("on console only")
    logger.complete()
    # Console-only logging must not create the file log folder.
    assert not folder.exists()
