#!/usr/bin/env python3
"""
Photo Tagger: CLI app to describe photos and add keywords using AI.

Optionally non-destructive: create/update XMP sidecar files with Lightroom-compatible tags.
Alternatively, pass --embed-in-photo to write metadata directly into the original file.
Unfortunately, Lightroom uses XMP sidecar files only for proprietary raw formats (e.g., CR3, NEF).
For JPEG, DNG and other formats, you'll often prefer embedding the metadata directly into the file.
This can be done with ExifTool manually as well:
    exiftool -tagsFromFile image.xmp -all:all image.jpg

Requirements:
 - Exiftool installed and available in PATH.
 - Ollama or LM Studio server running and containing a vision-language model.
"""

import contextlib
import importlib
import json
import os
import sys
import tempfile
import threading
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Protocol

from cyclopts import App, Parameter, validators
from loguru import logger

from photo_tagger import __version__, i18n, telemetry
from photo_tagger.ai import create_agent
from photo_tagger.cache import build_cache_namespace, open_cache
from photo_tagger.cli_options import (
    DEFAULT_EXTENSIONS,
    DEFAULT_RECURSIVE,
    DEFAULT_WORKERS,
    ArtifactConfig,
    ConfigFileSource,
    DisplayConfig,
    FilterConfig,
    InferenceConfig,
    LogConfig,
    OutputConfig,
    ProviderConfig,
    TelemetryConfig,
    to_processing_options,
)
from photo_tagger.config import DEFAULT_USER_PROMPT
from photo_tagger.config_file import configured_exiftool_path, load_config
from photo_tagger.csv_report import CsvReportWriter, ReportRow
from photo_tagger.diagnostics import render_report, run_checks
from photo_tagger.discovery import (
    apply_date_filter,
    apply_skip_file,
    apply_skip_tagged,
    make_skip_list_appender,
    resolve_image_batch,
)
from photo_tagger.errors import PhotoTaggerError
from photo_tagger.locking import FileLock, LockHeldError
from photo_tagger.logging_setup import setup_logging
from photo_tagger.metadata import prompt_with_hint, select_camera_fields, select_location
from photo_tagger.pipeline import BatchTotals, ImageOutcome, ProcessingOptions, run_batch
from photo_tagger.progress import batch_progress

# Runtime import (not type-only): cyclopts evaluates the Annotated[ProviderName, ...] field
# on the doctor command to validate the --provider choices, so it must exist at definition time.
from photo_tagger.providers import ProviderName  # noqa: TC001


if TYPE_CHECKING:
    from collections.abc import Callable


# Any TOML config is layered onto flags the user does not pass by the ConfigFileSource hook at
# parse time, giving per-field precedence (CLI flag > config file > built-in). The hook re-reads
# the file per invocation, so nothing config-dependent is captured at import.
app = App(name="photo-tagger", version=__version__, config=ConfigFileSource())


# Built-in default option groups, hoisted to module scope so the function-default expressions on
# `tag` are simple name lookups, which keeps ruff's B008 (no function call in a default argument)
# satisfied. Do NOT fold the config file into these instances: cyclopts rebuilds a group from its
# class defaults whenever any of the group's flags is passed, which would drop the config values
# of every sibling field in that group.
_DEFAULT_PROVIDER = ProviderConfig()
_DEFAULT_OUTPUT = OutputConfig()
_DEFAULT_INFERENCE = InferenceConfig()
_DEFAULT_LOG = LogConfig()
_DEFAULT_DISPLAY = DisplayConfig()
_DEFAULT_ARTIFACTS = ArtifactConfig()
_DEFAULT_FILTER = FilterConfig()
_DEFAULT_TELEMETRY = TelemetryConfig()


def _apply_exiftool_path(path: str | None) -> None:
    """Bridge a config ``exiftool_path`` into ``PHOTO_TAGGER_EXIFTOOL`` (an exported var wins)."""
    if path:
        os.environ.setdefault("PHOTO_TAGGER_EXIFTOOL", path)


@app.command
def doctor(
    *,
    provider: Annotated[
        ProviderName,
        Parameter(name=("--provider",), help="Backend provider to check"),
    ] = _DEFAULT_PROVIDER.provider_name,
    model: Annotated[
        str,
        Parameter(name=("--model", "-m"), help="Model name expected to be served"),
    ] = _DEFAULT_PROVIDER.model_name,
    url: Annotated[
        str | None,
        Parameter(name=("--url", "-u"), help="Provider API base URL"),
    ] = _DEFAULT_PROVIDER.api_base_url,
    api_key: Annotated[
        str | None,
        Parameter(name=("--api-key", "-k"), help="Provider API key (prefer env vars)"),
    ] = _DEFAULT_PROVIDER.api_key,
) -> None:
    """
    Check that ExifTool and the model provider are reachable, then exit.

    Prints a short checklist and exits 0 when everything is in order, 1 if any check fails. Run this
    first when a tagging run cannot start: it isolates a missing ExifTool, an unreachable provider,
    or a model name typo from the rest of the pipeline. Honors the same config file and env vars as
    ``tag``.
    """
    # Resolve the exiftool path (and its own warning, e.g. a CWD config redirecting the binary)
    # before silencing loguru, or that warning never reaches the user in the one command meant
    # to surface exactly this kind of misconfiguration.
    _apply_exiftool_path(configured_exiftool_path())
    # Silence loguru so only the checklist reaches the terminal; failures are
    # captured in the report itself, not the logs.
    logger.remove()
    results = run_checks(provider, model, api_base_url=url, api_key=api_key)
    if not render_report(results):
        raise SystemExit(1)


@app.command
def gui() -> None:
    """
    Launch the desktop GUI (requires the optional ``[gui]`` extra).

    The GUI is a review-before-write frontend over the same pipeline as the ``tag`` command: add
    photos or folders to a checkable list, generate proposals with the model, review and edit each
    photo's title, description, and keywords side by side with the existing metadata, then save.
    Install it with ``pip install 'photo-tagger[gui]'``.

    PySide6 is imported lazily here so the base CLI never depends on Qt; a missing dependency is
    reported with an install hint rather than a traceback.
    """
    try:
        gui_module = importlib.import_module("photo_tagger.gui")
    except ImportError as exc:
        # Only a genuinely missing Qt module earns the install hint. Anything else (a broken
        # shiboken build, an import error inside our own gui code) must surface as itself, or the
        # hint sends the user reinstalling an extra that is not the problem.
        missing = (exc.name or "").split(".")[0]
        if missing not in ("PySide6", "shiboken6"):
            raise
        sys.stderr.write(
            "The desktop GUI needs PySide6, which is not installed.\n"
            "Install the optional extra with:\n"
            "    pip install 'photo-tagger[gui]'\n",
        )
        raise SystemExit(1) from exc
    raise SystemExit(gui_module.launch())


def _read_prompt_file(prompt_file: Path | None) -> str:
    """Return the contents of *prompt_file* (stripped) or DEFAULT_USER_PROMPT when None."""
    if prompt_file is None:
        return DEFAULT_USER_PROMPT
    try:
        text = prompt_file.read_text(encoding="utf-8").strip()
    except OSError as exc:
        logger.error("prompt_file_read_failed", file=str(prompt_file), error=str(exc))
        raise SystemExit(1) from exc
    if not text:
        logger.error("prompt_file_empty", file=str(prompt_file))
        raise SystemExit(1)
    logger.info("prompt_file_loaded", file=str(prompt_file), chars=len(text))
    return text


class _TextSink(Protocol):
    """
    Minimal text-output interface the NDJSON emitter relies on.

    ``sys.stdout`` and ``io.StringIO`` both satisfy this without any extra glue, so tests can pass a
    buffer and the production path uses the real stream.
    """

    def write(self, s: str, /) -> int:
        raise NotImplementedError  # pragma: no cover - structural Protocol, never called

    def flush(self) -> None:
        raise NotImplementedError  # pragma: no cover - structural Protocol, never called


class _NDJSONEmitter:
    """
    Thread-safe ``on_image_result`` callback that writes NDJSON to a stream.

    Workers call this concurrently. The lock prevents two lines being interleaved in stdout under
    ``--workers > 1``. Each line is a complete JSON object that downstream consumers can parse with
    one ``json.loads`` per readline.
    """

    __slots__ = ("_lock", "_stream")

    _lock: threading.Lock
    _stream: _TextSink

    def __init__(self, stream: _TextSink) -> None:
        """Wrap *stream* (any object exposing ``write`` and ``flush``)."""
        self._stream = stream
        self._lock = threading.Lock()

    def __call__(self, outcome: ImageOutcome) -> None:
        """Serialize *outcome* to a single NDJSON line and flush."""
        payload = {
            "file": str(outcome.file),
            "status": "ok" if outcome.success else "failed",
            "from_cache": outcome.from_cache,
            "retry": outcome.retry,
            "title": outcome.title,
            "description": outcome.description,
            "keywords": outcome.keywords,
            "input_tokens": outcome.input_tokens,
            "output_tokens": outcome.output_tokens,
            "total_tokens": outcome.total_tokens,
            "seconds": outcome.seconds,
        }
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        with self._lock:
            self._stream.write(line)
            self._stream.flush()


def _open_csv_report(path: Path | None) -> CsvReportWriter | None:
    """
    Open the CSV report at *path*, or log and skip the report on failure.

    Returns ``None`` when *path* is ``None`` or the file cannot be opened (parent dir unwritable,
    filesystem full). A report we cannot write should never block tagging, so the run proceeds
    without one, mirroring how the cache degrades.
    """
    if path is None:
        return None
    try:
        writer = CsvReportWriter(path)
    except OSError as exc:
        logger.error("csv_report_open_failed", file=str(path), error=str(exc))
        return None
    logger.info("csv_report_opened", file=str(path))
    return writer


def _outcome_to_report_row(outcome: ImageOutcome) -> ReportRow:
    """Flatten a pipeline :class:`ImageOutcome` into a CSV :class:`ReportRow`."""
    model, lens, captured = select_camera_fields(outcome.camera_info)
    city, country = select_location(outcome.location_tags)
    return ReportRow(
        file=str(outcome.file),
        filename=outcome.file.name,
        status="ok" if outcome.success else "failed",
        title=outcome.title or "",
        description=outcome.description or "",
        keywords=list(outcome.written_keywords),
        hierarchical_keywords=list(outcome.hierarchical_keywords),
        existing_keywords=list(outcome.existing_keywords),
        camera_model=model or "",
        lens_model=lens or "",
        capture_date=captured or "",
        gps_position=outcome.gps_position or "",
        city=city or "",
        country=country or "",
        input_tokens=outcome.input_tokens,
        output_tokens=outcome.output_tokens,
        total_tokens=outcome.total_tokens,
        seconds=outcome.seconds,
        from_cache=outcome.from_cache,
        retry=outcome.retry,
    )


class _CsvImageResultSink:
    """``on_image_result`` callback that appends each outcome to the CSV report."""

    __slots__ = ("_writer",)

    def __init__(self, writer: CsvReportWriter) -> None:
        """Wrap an open :class:`CsvReportWriter`."""
        self._writer = writer

    def __call__(self, outcome: ImageOutcome) -> None:
        """Convert *outcome* to a report row and stream it to the file."""
        self._writer.write(_outcome_to_report_row(outcome))


def _combine_image_result_callbacks(
    *callbacks: Callable[[ImageOutcome], None] | None,
) -> Callable[[ImageOutcome], None] | None:
    """
    Fan one ImageOutcome out to every non-None *callback*, in order.

    Lets ``--json`` and ``--csv-file`` both observe each photo from the single ``on_image_result``
    hook. Returns the lone callback when only one is active, or ``None`` when none are, so the
    common single-sink path adds no wrapper.
    """
    active = [cb for cb in callbacks if cb is not None]
    if not active:
        return None
    if len(active) == 1:
        return active[0]

    def _fan_out(outcome: ImageOutcome) -> None:
        for callback in active:
            callback(outcome)

    return _fan_out


def _parse_filter_date(value: str | None, *, flag: str) -> datetime | None:
    """
    Parse an ISO 8601 string from *flag* into a timezone-aware datetime, or None.

    A naive timestamp like ``2024-01-01`` is read as **local** time (matching ``git log --since``
    and ``find -newer`` conventions). The system local zone is attached, and the value is returned
    aware so downstream comparisons with UTC mtimes do the right thing.
    """
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        logger.error("date_filter_parse_failed", flag=flag, value=value, error=str(exc))
        raise SystemExit(1) from exc
    if parsed.tzinfo is not None:
        return parsed
    # datetime.astimezone() on a naive value attaches the system local zone
    # (per the stdlib docs), turning it into a tz-aware value without shifting
    # the wall-clock fields. That is exactly the "user meant their local time"
    # semantics we want here.
    return parsed.astimezone()


def _atomic_write_text(target: Path, text: str) -> None:
    """
    Write *text* to *target* via a temp file + rename.

    Avoids leaving a half-written file on disk if the process is killed or the filesystem fills mid-
    write. The temp file lives in the same directory so the rename is a same-filesystem op, which
    POSIX guarantees is atomic.

    Uses ``tempfile.mkstemp`` instead of a PID-based name so the temp path is unpredictable and
    opened with ``O_EXCL``, preventing symlink-based attacks in shared directories.

    Creates the parent directory as needed so callers can point at a fresh location like
    ``reports/run.json`` without pre-mkdir.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fd = -1  # fdopen owns the descriptor now and closes it with the with-block.
            fh.write(text)
        tmp_path.replace(target)
    except BaseException:
        if fd != -1:
            # fdopen itself failed, so the raw descriptor is still open. Close it before the
            # unlink: Windows refuses to delete a file that still has an open handle.
            with contextlib.suppress(OSError):
                os.close(fd)
        tmp_path.unlink(missing_ok=True)
        raise


def _write_summary_file(  # noqa: PLR0913 - distinct optional fields are clearer as kwargs.
    summary_file: Path | None,
    totals: BatchTotals | None,
    *,
    started_at: datetime,
    model_name: str,
    provider_name: str,
    user_prompt_chars: int,
) -> None:
    """
    Serialize *totals* to *summary_file* as JSON.

    Errors are logged, never raised.
    """
    if summary_file is None or totals is None:
        return
    payload: dict[str, object] = {
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(tz=UTC).isoformat(),
        "provider": provider_name,
        "model": model_name,
        "user_prompt_chars": user_prompt_chars,
        **asdict(totals),
    }
    try:
        _atomic_write_text(summary_file, json.dumps(payload, indent=2) + "\n")
    except OSError as exc:
        logger.error("summary_file_write_failed", file=str(summary_file), error=str(exc))
        return
    logger.info("summary_file_written", file=str(summary_file))


def _maybe_str(value: Path | None) -> str | None:
    """Return ``str(value)`` or ``None`` so loguru extras keep a clean type."""
    return str(value) if value is not None else None


def _maybe_show_telemetry_notice(*, enabled: bool) -> None:
    """
    Print the one-time telemetry disclosure to stderr on the first run telemetry is active.

    Only shown when telemetry will actually send (config/flag on and no environment opt-out), so a
    user who has disabled it never sees the notice. Written straight to stderr, not the loguru sink,
    so it reaches the terminal regardless of the configured log level.
    """
    if not telemetry.should_send(config_enabled=enabled):
        return
    notice = telemetry.first_run_notice()
    if notice is not None:
        sys.stderr.write(notice + "\n")


def _log_startup(  # noqa: PLR0913 - the log line names every config explicitly.
    *,
    inputs: list[Path] | None,
    image_extensions: str,
    recursive: bool,
    workers: int,
    filter_: FilterConfig,
    display: DisplayConfig,
    artifacts: ArtifactConfig,
    provider: ProviderConfig,
    options: ProcessingOptions,
    output_language: str,
    hint: str | None,
    log: LogConfig,
    telemetry_enabled: bool,
) -> None:
    """Single-shot info log so the run's full configuration is captured up-front."""
    logger.info(
        "starting_photo_tagger",
        inputs=[str(p) for p in (inputs or [])],
        extensions=image_extensions,
        model=provider.model_name,
        provider=provider.provider_name,
        api_base_url=provider.api_base_url,
        api_key_present=bool(provider.api_key),
        recursive=recursive,
        workers=workers,
        prompt_file=_maybe_str(artifacts.prompt_file),
        summary_file=_maybe_str(artifacts.summary_file),
        csv_file=_maybe_str(artifacts.csv_file),
        cache_file=_maybe_str(artifacts.cache_file),
        lock_file=_maybe_str(artifacts.lock_file),
        json_output=display.json_output,
        progress_bar=display.progress_bar,
        telemetry=telemetry_enabled,
        skip_from=_maybe_str(artifacts.skip_from),
        append_to_skip_file=_maybe_str(artifacts.append_to_skip_file),
        skip_tagged=filter_.skip_tagged,
        newer_than=filter_.newer_than,
        older_than=filter_.older_than,
        preserve_keywords=options.preserve_existing_kw,
        write_description=options.write_description,
        write_title=options.write_title,
        write_keywords=options.write_keywords,
        backup_xmp=options.backup_xmp,
        use_sidecar=options.use_sidecar,
        dry_run=options.dry_run,
        max_keywords=options.max_new_keywords,
        temperature=options.temperature,
        max_tokens=options.max_tokens,
        timeout_seconds=options.timeout_seconds,
        frequency_penalty=options.frequency_penalty,
        jpeg_dimensions=options.jpeg_dimensions,
        jpeg_quality=options.jpeg_quality,
        output_language=output_language,
        hint=hint,
        retries=provider.retries,
        log_folder=str(log.log_folder),
    )


@app.default
def tag(  # noqa: PLR0913 - cyclopts entry point; each arg is a CLI flag group.
    inputs: Annotated[
        list[Path] | None,
        Parameter(
            name=("--input", "-i"),
            validator=validators.Path(exists=True),
            help="One or more paths: files and/or directories (repeat this option)",
        ),
    ] = None,
    *,
    image_extensions: Annotated[
        str,
        Parameter(
            name=("--ext", "--extensions"),
            help="Comma-separated image file extensions to process (case insensitive)",
        ),
    ] = DEFAULT_EXTENSIONS,
    recursive: Annotated[
        bool,
        Parameter(
            name=("--recursive", "-r"),
            help="Process files in subdirectories recursively",
        ),
    ] = DEFAULT_RECURSIVE,
    workers: Annotated[
        int,
        Parameter(
            name=("--workers", "-w"),
            help=(
                "Number of photos to process concurrently. Defaults to 1 (serial). The model "
                "server is the bottleneck; raising this past what your provider can serve in "
                "parallel will not help"
            ),
        ),
    ] = DEFAULT_WORKERS,
    filter_: Annotated[FilterConfig, Parameter(name="*")] = _DEFAULT_FILTER,
    display: Annotated[DisplayConfig, Parameter(name="*")] = _DEFAULT_DISPLAY,
    artifacts: Annotated[ArtifactConfig, Parameter(name="*")] = _DEFAULT_ARTIFACTS,
    provider: Annotated[ProviderConfig, Parameter(name="*")] = _DEFAULT_PROVIDER,
    output: Annotated[OutputConfig, Parameter(name="*")] = _DEFAULT_OUTPUT,
    inference: Annotated[InferenceConfig, Parameter(name="*")] = _DEFAULT_INFERENCE,
    log: Annotated[LogConfig, Parameter(name="*")] = _DEFAULT_LOG,
    telemetry_config: Annotated[TelemetryConfig, Parameter(name="*")] = _DEFAULT_TELEMETRY,
) -> None:
    """
    Tag images with AI and write Lightroom-compatible metadata (sidecar or embedded).

    Requirements:
    - ExifTool installed and on PATH.
    - Vision-language model API reachable (e.g., Ollama server).

    Inputs:
    - One or more --input/-i paths (files and/or directories; repeatable).
    - Files are processed as is. Directories use --ext (add --recursive for subfolders).
    - You can mix files and directories; order is preserved, duplicates skipped.

    Behavior:
    - Loads image (RAW supported), converts to in-memory JPEG, queries the model.
    - Generates title, description, and keywords; merges with existing XMP by default
        (use --overwrite-keywords to replace).
    - Writes metadata to an XMP sidecar (default) or embeds it directly when --embed-in-photo
        is used. Use --no-write-title/--no-write-description to skip fields; --no-backup-xmp
        to avoid backups.

    Skipping:
    - --skip-from FILE: skip files listed in FILE (one name or path per line).
    - --append-to-skip-file FILE: append each successfully processed file's path to FILE so a
        later run with --skip-from FILE resumes where this one stopped.
    - --skip-tagged: skip files that already have keywords, a description, or a title in
        the image or its XMP sidecar.

    Dry runs:
    - --dry-run: query the model and log the generated title, description, and keywords
        for each image but do not write any metadata. Useful for prompt iteration.

    Performance:
    - --workers N: process N photos concurrently using a thread pool. Each worker opens
        its own ExifToolHelper. The model server is the dominant bottleneck.
    - --max-keywords N: cap the number of AI-generated keywords kept before merging.

    Customization:
    - --prompt-file PATH: replace the default user prompt with the file's contents.
        Existing photo metadata (keywords, GPS, location) is still appended automatically.

    Exit status: returns 1 if no inputs are given, discovery finds no matching images, or any
        file fails. Returns 0 if discovery finds images but every one of them is subsequently
        excluded by --skip-from, --skip-tagged, or the date filters: that is "nothing left to
        do", not an error, and scripted incremental reruns rely on it not failing the batch.

    Examples:
        photo-tagger -i ./photos/IMG_0001.CR3

        photo-tagger \
            --extensions cr3,jpg \
            --provider lmstudio \
            --url http://localhost:1234/v1 \
            --recursive \
            --skip-from processed.txt \
            --append-to-skip-file processed.txt \
            Pictures/Camera

        photo-tagger -i Pictures/Mixed --skip-tagged
    """
    setup_logging(
        file_log_level=log.file_log_level,
        console_log_level=log.console_log_level,
        log_folder=log.log_folder,
    )
    _apply_exiftool_path(configured_exiftool_path())

    with contextlib.ExitStack() as stack:
        if artifacts.lock_file is not None:
            try:
                stack.enter_context(FileLock(artifacts.lock_file))
            except LockHeldError as exc:
                logger.error(
                    "lock_held_by_other_process",
                    file=str(artifacts.lock_file),
                    error=str(exc),
                )
                raise SystemExit(1) from exc
            except OSError as exc:
                # Parent dir not writable, filesystem full, etc. Surface a clean
                # message instead of letting the traceback land in the user's tty.
                logger.error(
                    "lock_file_open_failed",
                    file=str(artifacts.lock_file),
                    error=str(exc),
                )
                raise SystemExit(1) from exc

        try:
            _tag_inside_lock(
                inputs=inputs,
                image_extensions=image_extensions,
                recursive=recursive,
                workers=workers,
                filter_=filter_,
                display=display,
                artifacts=artifacts,
                provider=provider,
                output=output,
                inference=inference,
                log=log,
                telemetry_config=telemetry_config,
            )
        except PhotoTaggerError as exc:
            raise SystemExit(1) from exc


def _tag_inside_lock(  # noqa: PLR0913 - mirrors tag()'s flag groups one-for-one.
    *,
    inputs: list[Path] | None,
    image_extensions: str,
    recursive: bool,
    workers: int,
    filter_: FilterConfig,
    display: DisplayConfig,
    artifacts: ArtifactConfig,
    provider: ProviderConfig,
    output: OutputConfig,
    inference: InferenceConfig,
    log: LogConfig,
    telemetry_config: TelemetryConfig,
) -> None:
    """Body of ``tag`` that runs once the optional file lock has been acquired."""
    _maybe_show_telemetry_notice(enabled=telemetry_config.enabled)
    options = to_processing_options(output, inference)
    newer_than = _parse_filter_date(filter_.newer_than, flag="--newer-than")
    older_than = _parse_filter_date(filter_.older_than, flag="--older-than")
    _log_startup(
        inputs=inputs,
        image_extensions=image_extensions,
        recursive=recursive,
        workers=workers,
        filter_=filter_,
        display=display,
        artifacts=artifacts,
        provider=provider,
        options=options,
        output_language=inference.output_language,
        hint=inference.hint,
        log=log,
        telemetry_enabled=telemetry_config.enabled,
    )

    image_files = apply_skip_file(
        resolve_image_batch(inputs, image_extensions, recursive=recursive),
        artifacts.skip_from,
    )
    image_files = apply_date_filter(
        image_files,
        newer_than=newer_than,
        older_than=older_than,
    )
    image_files = apply_skip_tagged(image_files, skip_tagged=filter_.skip_tagged)
    if not image_files:
        logger.info("no_files_to_process_after_skipping")
        return

    # A --hint rides inside the user prompt, so the cache namespace below picks it up too:
    # a hinted run never replays results generated without the hint (and vice versa).
    user_prompt = prompt_with_hint(_read_prompt_file(artifacts.prompt_file), inference.hint)
    agent = create_agent(
        provider.provider_name,
        provider.model_name,
        api_base_url=provider.api_base_url,
        api_key=provider.api_key,
        retries=provider.retries,
        output_language=inference.output_language,
    )
    # Fold the prompt + language + sampling/JPEG settings into the cache namespace so a
    # different configuration writes to a fresh slice instead of replaying
    # stale entries generated under earlier settings.
    cache_namespace = build_cache_namespace(
        provider.model_name,
        user_prompt=user_prompt,
        temperature=inference.temperature,
        max_tokens=inference.max_tokens,
        frequency_penalty=inference.frequency_penalty,
        jpeg_dimensions=inference.jpeg_dimensions,
        jpeg_quality=inference.jpeg_quality,
        output_language=inference.output_language,
    )
    cache = open_cache(artifacts.cache_file, namespace=cache_namespace)
    started_at = datetime.now(tz=UTC)

    def _on_complete(totals: BatchTotals) -> None:
        # Runs once before run_batch raises SystemExit, so the summary file is written and the
        # telemetry beacon fired whether the batch succeeded fully or only partially.
        _write_summary_file(
            artifacts.summary_file,
            totals,
            started_at=started_at,
            model_name=provider.model_name,
            provider_name=provider.provider_name,
            user_prompt_chars=len(user_prompt),
        )
        telemetry.emit(
            telemetry.RunInfo(
                interface="cli",
                provider=provider.provider_name,
                model=provider.model_name,
                batch_size=totals.total_files,
                duration_seconds=(datetime.now(tz=UTC) - started_at).total_seconds(),
                output_language=inference.output_language,
                ui_language=i18n.current_language(),
                file_types=telemetry.file_types_summary(image_files),
                success_count=totals.success,
                failure_count=len(totals.failed_files),
                cache_hits=totals.cache_hits,
                retry_successes=totals.retry_successes,
                workers=totals.workers,
                total_tokens=totals.total_tokens,
                inference_seconds=totals.inference_seconds,
                dry_run=totals.dry_run,
                failure_kinds=telemetry.failure_kinds_summary(totals.failure_kinds),
            ),
            enabled=telemetry_config.enabled,
            block=True,
        )

    csv_writer = _open_csv_report(artifacts.csv_file)
    ndjson_emitter = _NDJSONEmitter(sys.stdout) if display.json_output else None
    csv_sink = _CsvImageResultSink(csv_writer) if csv_writer is not None else None
    on_image_result = _combine_image_result_callbacks(ndjson_emitter, csv_sink)
    try:
        with batch_progress(len(image_files), enabled=display.progress_bar) as progress:
            run_batch(
                image_files,
                agent,
                options,
                on_success=make_skip_list_appender(artifacts.append_to_skip_file),
                on_complete=_on_complete,
                user_prompt=user_prompt,
                workers=max(1, workers),
                progress=progress,
                cache=cache,
                on_image_result=on_image_result,
            )
    finally:
        if cache is not None:
            cache.close()
        if csv_writer is not None:
            csv_writer.close()


def _crash_telemetry_enabled(tokens: list[str]) -> bool:
    """
    Best-effort telemetry opt-out resolution for the crash path.

    A crash may happen before (or during) CLI parsing, so the parsed --no-telemetry flag is not
    available; scan the raw *tokens* for it and read the config file's [telemetry] table. The
    environment opt-outs are enforced inside emit_crash itself.
    """
    if "--no-telemetry" in tokens:
        return False
    table = load_config().get("telemetry", {})
    return bool(table.get("enabled", True)) if isinstance(table, dict) else True


def main(argv: list[str] | None = None) -> None:
    """
    Console entry point: run the CLI, reporting an unhandled crash before re-raising.

    Expected exits (SystemExit from clean error handling, Ctrl-C) pass through untouched; anything
    else is a genuine crash, so an anonymous beacon (exception type and in-app code location only,
    never the message) is sent before the traceback surfaces as usual.
    """
    tokens = sys.argv[1:] if argv is None else argv
    try:
        app(tokens)
    except SystemExit, KeyboardInterrupt:
        raise
    except Exception as exc:
        telemetry.emit_crash(
            exc,
            interface="cli",
            enabled=_crash_telemetry_enabled(tokens),
            block=True,
        )
        raise


if __name__ == "__main__":  # pragma: no cover - module entry point, not exercised by tests
    main()
