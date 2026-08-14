"""Loguru configuration shared by the CLI and any embedding caller."""

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger


if TYPE_CHECKING:
    from photo_tagger.config import LogLevel


_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name:<8}:{function:<25}:{line:>4} | "
    "{message:<40} | "
    "{extra}"
)

_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <7}</level> | "
    "<level>{message:<40.50}</level> | "
    "<yellow>{extra}</yellow>"
)

# Loguru's built-in sink prints everything, DEBUG included, to stderr. Anything logged before
# setup_logging() runs (config loading happens at import time) would leak at DEBUG, so replace
# that sink with an INFO console the moment this module is imported. setup_logging() then
# reconfigures per the user's flags.
logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True, format=_CONSOLE_FORMAT)


def setup_logging(
    file_log_level: LogLevel = "DEBUG",
    console_log_level: LogLevel = "INFO",
    log_folder: Path = Path("logs"),
) -> None:
    """
    Configure Loguru for both console and file logging.

    The file sink is serialized: each line is one JSON record (message, level, timestamp, and the
    structured ``extra`` fields), so the logs can be filtered and parsed with jq and friends instead
    of regexes.

    Args:
        file_log_level: Log level for file (use 'OFF' to disable)
        console_log_level: Log level for console (use 'OFF' to disable)
        log_folder: Directory where log files are stored
    """
    logger.remove()
    if file_log_level != "OFF":
        log_folder.mkdir(parents=True, exist_ok=True)
        log_file = log_folder / Path(
            datetime.now(tz=UTC).strftime("%Y%m%d%H%M%S-photo_tagger.log"),
        )
        logger.add(
            log_file,
            level=file_log_level,
            format=_FILE_FORMAT,
            serialize=True,
            rotation="500 MB",
            retention="10 days",
            compression="zip",
        )
        # DEBUG-level records carry the `extra` context (file paths, provider URLs); restrict to
        # the owner so they are not world-readable on a shared multi-user machine. add() opens
        # (and thus creates) the file immediately, so it exists to chmod by this point. A no-op
        # on Windows, which has no POSIX permission bits; a rotated file at 500MB does not inherit
        # this, but this project's logs are nowhere near that size in normal use. Purely
        # hardening, so a failure (e.g. an unsupported filesystem) must not abort logging setup.
        try:
            log_file.chmod(0o600)
        except OSError as exc:
            logger.warning("log_file_chmod_failed", file=str(log_file), error=str(exc))
    if console_log_level != "OFF":
        logger.add(
            sys.stderr,
            level=console_log_level,
            colorize=True,
            format=_CONSOLE_FORMAT,
        )
