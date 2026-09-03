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
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Protocol

from cyclopts import App, Parameter, validators
from loguru import logger
from rich.console import Console

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
    VocabularyBuildConfig,
    to_processing_options,
    to_trim_rules,
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
from photo_tagger.errors import DiscoveryError, PhotoTaggerError
from photo_tagger.locking import FileLock, LockHeldError
from photo_tagger.logging_setup import setup_logging
from photo_tagger.metadata import prompt_with_hint, select_camera_fields, select_location
from photo_tagger.pipeline import BatchTotals, ImageOutcome, ProcessingOptions, run_batch
from photo_tagger.progress import batch_progress

# Runtime import (not type-only): cyclopts evaluates the Annotated[ProviderName, ...] field
# on the doctor command to validate the --provider choices, so it must exist at definition time.
from photo_tagger.providers import ProviderName  # noqa: TC001
from photo_tagger.sessions import plan_sessions
from photo_tagger.undo import (
    DELETED,
    RESTORED,
    UndoError,
    UndoResult,
    latest_journal,
    list_journals,
    open_journal,
    read_journal,
    undo_run,
)
from photo_tagger.vocabulary import load_vocabulary, prompt_with_vocabulary
from photo_tagger.vocabulary_build import (
    KeywordCensus,
    TrimRules,
    census_from_export,
    census_from_photos,
    render_drop_report,
    render_vocabulary,
    trim,
)
from photo_tagger.vocabulary_organize import OrganizeStats, organize


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


def _vocabulary_header(
    source: str,
    kept: int,
    dropped: int,
    rules: TrimRules,
    stats: OrganizeStats | None = None,
) -> str:
    """Explain at the top of the generated file where it came from and how to change it."""
    cap = rules.max_terms if rules.max_terms is not None else "no cap"
    lines = [
        f"# photo-tagger vocabulary: {kept} keywords kept, {dropped} dropped.",
        f"# Source: {source}.",
        (
            f"# Rules: used at least {rules.min_uses}x, at most {cap} terms, "
            f"digits {'kept' if rules.allow_digits else 'dropped'}."
        ),
    ]
    if stats is not None:
        lines += [
            (
                f"# Organized by {stats.model_name}: {stats.grouped} keyword(s) folded into a "
                f"synonym, {stats.categorized} filed under a category."
            ),
            f"# Categories (written to your photos as parents): {', '.join(stats.categories)}.",
        ]
    lines += [
        "#",
        "# Edit freely: one keyword per line, 'Parent|Child' for a hierarchy, indentation for a",
        "# tree, {braces} for a synonym. A line starting with '# ' is a comment.",
    ]
    return "\n".join(lines) + "\n"


def _build_census(
    inputs: list[Path] | None,
    from_export: Path | None,
    *,
    image_extensions: str,
    recursive: bool,
) -> tuple[KeywordCensus, str]:
    """Count keywords from the photos, from an export, or from both; also name the source."""
    census = KeywordCensus()
    sources: list[str] = []
    if from_export is not None:
        census = census_from_export(from_export.read_text(encoding="utf-8"))
        sources.append(f"keyword export {from_export.name} (counts are tree occurrences)")
    if inputs:
        image_files = resolve_image_batch(inputs, image_extensions, recursive=recursive)
        photo_census = census_from_photos(image_files)
        for term, count in photo_census.uses.items():
            census.uses[term] += count
        for term, chains in photo_census.chains.items():
            census.chains.setdefault(term, Counter()).update(chains)
        census.photos = photo_census.photos
        sources.append(f"{photo_census.photos} photo(s)")
    return census, " and ".join(sources)


@app.command
def vocabulary(  # noqa: PLR0913 - inputs, output, and the option groups are all distinct concerns.
    inputs: Annotated[
        list[Path] | None,
        Parameter(
            name=("--input", "-i"),
            help="Photos or folders to read existing keywords from (repeat this option)",
        ),
    ] = None,
    *,
    output: Annotated[
        Path,
        Parameter(
            name=("--output", "-o"),
            validator=validators.Path(file_okay=True, dir_okay=False),
            help="Where to write the vocabulary file",
        ),
    ],
    from_export: Annotated[
        Path | None,
        Parameter(
            name=("--from-export",),
            validator=validators.Path(exists=True, file_okay=True, dir_okay=False),
            help=(
                "Read a Lightroom keyword export (.txt or .csv) instead of, or as well as, the "
                "photos. Counts from an export are tree occurrences, not photos"
            ),
        ),
    ] = None,
    image_extensions: Annotated[
        str,
        Parameter(name=("--ext", "--extensions"), help="Extensions to scan for (case insensitive)"),
    ] = DEFAULT_EXTENSIONS,
    recursive: Annotated[
        bool,
        Parameter(name=("--recursive", "-r"), help="Scan subdirectories too"),
    ] = DEFAULT_RECURSIVE,
    build: Annotated[VocabularyBuildConfig, Parameter(name="*")] = VocabularyBuildConfig(),  # noqa: B008
    provider: Annotated[ProviderConfig, Parameter(name="*")] = _DEFAULT_PROVIDER,
) -> None:
    """
    Build a controlled vocabulary from the keywords a library already uses.

    ``--vocabulary-strict`` is what stops a catalog sprawling, and it needs a keyword file worth
    enforcing. This writes one: point it at your photos and it reads the keywords they already carry
    (through exiftool, so any application that writes XMP or IPTC works, not only Lightroom), counts
    how often each is used, and keeps the ones that earn their place.

    ``--from-export`` reads a Lightroom keyword export instead, for a catalog that is not on this
    machine. Its counts are occurrences in the keyword tree rather than photos, which is a weaker
    signal; prefer the photos when you have them.

    ``--organize`` adds a model pass over the keywords that survived, for the two things counting
    cannot settle: which of them are synonyms of each other, and what hierarchy they should have. It
    never decides what to keep and never invents a keyword.

    Nothing is written to your photos or your catalog. The output is a file to review and edit, plus
    an optional ``--report`` naming every keyword that was dropped and why.
    """
    _apply_exiftool_path(configured_exiftool_path())
    if not inputs and from_export is None:
        logger.error("vocabulary_no_source")
        raise SystemExit(1)

    console = Console()
    rules = to_trim_rules(build)
    try:
        census, source = _build_census(
            inputs,
            from_export,
            image_extensions=image_extensions,
            recursive=recursive,
        )
    except (DiscoveryError, OSError) as exc:
        logger.error("vocabulary_source_unreadable", error=str(exc))
        raise SystemExit(1) from exc

    if not census.uses:
        console.print("[yellow]No keywords found: nothing to build a vocabulary from.[/yellow]")
        raise SystemExit(1)

    result = trim(census, rules)
    stats: OrganizeStats | None = None
    if build.organize:
        console.print(f"Organizing {len(result.kept)} keyword(s) with {provider.model_name}...")
        try:
            result, stats = organize(
                result,
                provider_name=provider.provider_name,
                model_name=provider.model_name,
                api_base_url=provider.api_base_url,
                api_key=provider.api_key,
                workers=build.organize_workers,
            )
        except PhotoTaggerError as exc:
            # The trimmed list is already worth writing, but silently downgrading to it would hide
            # that the organize pass the user asked for never ran.
            logger.error("vocabulary_organize_failed", error=str(exc))
            console.print(f"[red]{exc}[/red]")
            raise SystemExit(1) from exc

    header = _vocabulary_header(source, len(result.kept), len(result.dropped), rules, stats)
    try:
        output.write_text(
            render_vocabulary(result, header=header, flat=build.flat),
            encoding="utf-8",
        )
        if build.report_file is not None:
            build.report_file.write_text(render_drop_report(result), encoding="utf-8")
    except OSError as exc:
        logger.error("vocabulary_write_failed", error=str(exc))
        raise SystemExit(1) from exc

    console.print(f"Read {len(census.uses)} keyword(s) from {source}.")
    console.print(f"[green]Wrote {len(result.kept)} keyword(s) to {output}[/green]")
    if build.report_file is not None:
        console.print(f"Dropped {len(result.dropped)}; see {build.report_file}")
    console.print(
        "\nReview the file, then tag with it:\n"
        f"  photo-tagger -i PHOTOS --vocabulary {output} --vocabulary-strict",
    )


def _render_journal_list(console: Console) -> None:
    """Print the recorded runs, newest first, with how many files each one wrote."""
    journals = list_journals()
    if not journals:
        console.print("No recorded runs to undo.")
        return
    console.print("Recorded runs (newest first):\n")
    for path in journals:
        try:
            entries = len(read_journal(path))
        except UndoError:  # pragma: no cover - listing must survive one unreadable journal
            entries = 0
        console.print(f"  {path.name}  {entries} file(s)  {path}")


# Undo outcomes that mean the file is back as it was. Anything else needs the user's attention,
# so it decides the exit code.
_UNDO_OK_ACTIONS = frozenset({RESTORED, DELETED})


def _render_undo_results(results: list[UndoResult], console: Console, *, dry_run: bool) -> bool:
    """Print one line per entry and return True when every one of them was put back."""
    verb = "Would undo" if dry_run else "Undoing"
    console.print(f"{verb} {len(results)} write(s)\n")
    for result in results:
        ok = result.action in _UNDO_OK_ACTIONS
        colour = "green" if ok else "yellow"
        mark = f"[{colour}]{result.action:<9}[/{colour}]"
        detail = f"  ({result.detail})" if result.detail else ""
        console.print(f"  {mark}  {result.target}{detail}")
    skipped = [result for result in results if result.action not in _UNDO_OK_ACTIONS]
    if skipped:
        console.print(f"\n[yellow]{len(skipped)} entry/entries left alone.[/yellow]")
    else:
        console.print("\n[green]Every recorded write was put back.[/green]")
    return not skipped


@app.command
def undo(
    *,
    run: Annotated[
        Path | None,
        Parameter(
            name=("--run",),
            validator=validators.Path(exists=True, file_okay=True, dir_okay=False),
            help="Undo this journal instead of the most recent run",
        ),
    ] = None,
    show_list: Annotated[
        bool,
        Parameter(name=("--list",), help="List the recorded runs and exit"),
    ] = False,
    dry_run: Annotated[
        bool,
        Parameter(name=("--dry-run",), help="Report what would be put back, without touching it"),
    ] = False,
    force: Annotated[
        bool,
        Parameter(
            name=("--force",),
            help=(
                "Revert files that changed after the run wrote them. Without this they are left "
                "alone, because a change means someone edited the file since"
            ),
        ),
    ] = False,
) -> None:
    """
    Put back what the last tagging run wrote.

    Every run records the files it writes (unless ``--no-undo-log`` was passed), so a batch tagged
    with the wrong prompt or the wrong vocabulary can be reverted in one command: sidecars the run
    created are deleted, and files it overwrote are restored from ExifTool's ``*_original`` backup.

    A file that changed since the run is left alone unless ``--force`` says otherwise, and a file
    written with ``--no-backup-xmp`` cannot be restored at all: there is no copy of what it held.
    Runs from the desktop GUI are not recorded.

    Exit status: 1 when there is nothing to undo or when any entry was left alone; 0 when every
    recorded write was put back.
    """
    console = Console()
    if show_list:
        _render_journal_list(console)
        return

    journal_path = run or latest_journal()
    if journal_path is None:
        console.print("No recorded runs to undo.")
        raise SystemExit(1)

    try:
        records = read_journal(journal_path)
    except UndoError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1) from exc
    if not records:
        console.print(f"{journal_path} records no writes.")
        raise SystemExit(1)

    console.print(f"Undoing run {journal_path.name}")
    results = undo_run(records, force=force, dry_run=dry_run)
    if not _render_undo_results(results, console, dry_run=dry_run):
        raise SystemExit(1)


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
    session_gap_minutes: float,
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
        vocabulary_terms=len(options.vocabulary.terms) if options.vocabulary else 0,
        vocabulary_strict=options.vocabulary_strict,
        session_gap_minutes=session_gap_minutes,
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
    try:
        setup_logging(
            file_log_level=log.file_log_level,
            console_log_level=log.console_log_level,
            log_folder=log.log_folder,
        )
    except OSError as exc:
        # No sink is guaranteed to be active yet (setup_logging removes the default one before
        # it can fail), so this cannot rely on the logger; write directly to stderr like the gui
        # command's own pre-logging error path does.
        sys.stderr.write(f"Could not set up logging at {log.log_folder}: {exc}\n")
        raise SystemExit(1) from exc
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
    # Raises VocabularyError (a PhotoTaggerError) on an unusable file, which `tag` turns into a
    # clean exit 1. Loading it up front means a typo in the path fails before any model call.
    vocabulary = (
        load_vocabulary(output.vocabulary, output_language=inference.output_language)
        if output.vocabulary is not None
        else None
    )
    options = to_processing_options(output, inference, vocabulary=vocabulary)
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
        session_gap_minutes=output.session_gap_minutes,
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

    # A --hint and the --vocabulary listing ride inside the user prompt, so the cache namespace
    # below picks them up too: a hinted run never replays results generated without the hint (and
    # vice versa), and the same holds for a vocabulary.
    user_prompt = prompt_with_vocabulary(
        prompt_with_hint(_read_prompt_file(artifacts.prompt_file), inference.hint),
        vocabulary,
    )
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
    # A dry run writes nothing, so there is nothing for undo to put back.
    journal = open_journal(started_at, enabled=artifacts.undo_log and not output.dry_run)
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
                session_plan=plan_sessions(
                    image_files,
                    gap_minutes=output.session_gap_minutes,
                ),
                journal=journal,
            )
    finally:
        if cache is not None:
            cache.close()
        if csv_writer is not None:
            csv_writer.close()
        if journal is not None and journal.entries:
            logger.info("undo_journal_written", file=str(journal.path), entries=journal.entries)


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
