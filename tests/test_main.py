"""
End-to-end wiring tests for the cyclopts CLI in photo_tagger.main.

The agent (network calls) and pipeline (long-running work) are mocked; the goal is to prove that
flag values reach the right collaborators with the right shape, and that the short-circuit / skip
code paths run end to end without raising. Real work is exercised elsewhere by the per-module unit
tests.
"""

import contextlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from photo_tagger import (
    main as main_module,
    telemetry,
)
from photo_tagger.cli_options import load_defaults
from photo_tagger.pipeline import BatchTotals, ImageOutcome


if TYPE_CHECKING:
    from pathlib import Path


def _make_jpeg(path: Path) -> Path:
    """Write a small placeholder file so cyclopts' --input validator accepts the path."""
    path.write_bytes(b"\xff\xd8stub")
    return path


def _run_app(args: list[str]) -> None:
    """Invoke the cyclopts app while absorbing the SystemExit it always raises."""
    with contextlib.suppress(SystemExit):
        main_module.app(args)


def _patches(captured: dict[str, Any]) -> Any:  # noqa: ANN401 - context manager juggling.
    """Patch every IO collaborator main.tag invokes; capture the values for assertions."""

    def fake_run_batch(
        image_files: list[Path],
        agent: object,
        options: object,
        **kwargs: object,
    ) -> object:
        captured["image_files"] = list(image_files)
        captured["options"] = options
        captured["on_success"] = kwargs.get("on_success")
        captured["on_complete"] = kwargs.get("on_complete")
        captured["user_prompt"] = kwargs.get("user_prompt")
        captured["workers"] = kwargs.get("workers")
        captured["on_image_result"] = kwargs.get("on_image_result")
        captured["cache"] = kwargs.get("cache")
        return None

    return (
        patch.object(main_module, "setup_logging"),
        patch.object(main_module, "create_agent", return_value=object()),
        patch.object(main_module, "run_batch", side_effect=fake_run_batch),
    )


def test_cli_passes_inputs_and_dry_run_through_to_pipeline(tmp_path: Path) -> None:
    """A minimal invocation feeds the resolved file list and dry_run flag to run_batch."""
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--dry-run"])

    assert captured["image_files"] == [image.resolve()]
    assert captured["options"].dry_run is True
    # Default skip_tagged path means no on_success appender created.
    assert captured["on_success"] is None


def test_cli_creates_appender_when_append_to_skip_file_provided(tmp_path: Path) -> None:
    """Passing --append-to-skip-file installs an on_success callback on run_batch."""
    image = _make_jpeg(tmp_path / "img.cr3")
    skip_file = tmp_path / "processed.txt"
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--append-to-skip-file", str(skip_file)])

    assert callable(captured["on_success"])

    # The callback writes a single line per call. Exercise it once to be sure.
    captured["on_success"](image)
    assert skip_file.read_text(encoding="utf-8").splitlines() == [str(image)]


def test_cli_skip_tagged_filters_before_pipeline(tmp_path: Path) -> None:
    """--skip-tagged removes already-tagged paths before run_batch is even called."""
    keep = _make_jpeg(tmp_path / "keep.cr3")
    drop = _make_jpeg(tmp_path / "drop.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with (
        setup,
        create_agent,
        run_batch,
        patch("photo_tagger.discovery.find_tagged_images", return_value={drop.resolve()}),
    ):
        _run_app(["--input", str(keep), "--input", str(drop), "--skip-tagged"])

    assert captured["image_files"] == [keep.resolve()]


def test_cli_short_circuits_when_skip_filters_remove_everything(tmp_path: Path) -> None:
    """When all inputs are skipped, run_batch is not called and the CLI exits cleanly."""
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with (
        setup,
        create_agent,
        run_batch,
        patch("photo_tagger.discovery.find_tagged_images", return_value={image.resolve()}),
    ):
        _run_app(["--input", str(image), "--skip-tagged"])

    assert "image_files" not in captured  # run_batch never invoked


def test_cli_exits_when_no_inputs_passed() -> None:
    """No --input is a hard error so accidental empty runs surface immediately."""
    setup, create_agent, run_batch = _patches({})
    with setup, create_agent, run_batch, pytest.raises(SystemExit):
        main_module.app([])


_EXPECTED_WORKERS = 3


def test_cli_workers_and_prompt_file_reach_run_batch(tmp_path: Path) -> None:
    """--workers and --prompt-file both flow into the run_batch call."""
    image = _make_jpeg(tmp_path / "img.cr3")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Describe like a wildlife photographer.\n", encoding="utf-8")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(
            [
                "--input",
                str(image),
                "--workers",
                str(_EXPECTED_WORKERS),
                "--prompt-file",
                str(prompt),
            ],
        )

    assert captured["workers"] == _EXPECTED_WORKERS
    assert captured["user_prompt"] == "Describe like a wildlife photographer."


def test_cli_hint_lands_in_the_user_prompt(tmp_path: Path) -> None:
    """--hint rides inside the user prompt, so the cache namespace changes with it too."""
    from photo_tagger.config import DEFAULT_USER_PROMPT  # noqa: PLC0415

    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--hint", "The animal is a deer"])

    prompt = captured["user_prompt"]
    assert "Photographer's note about this photo: The animal is a deer" in prompt
    assert prompt.startswith(DEFAULT_USER_PROMPT)


def _outcome(file: Path, *, success: bool = True, from_cache: bool = False) -> ImageOutcome:
    """Build a representative ImageOutcome for NDJSON-emitter tests."""
    return ImageOutcome(
        file=file,
        success=success,
        from_cache=from_cache,
        retry=False,
        title="A Title",
        description="A description.",
        keywords=["Beach", "Sunset"],
        input_tokens=42,
        output_tokens=7,
        total_tokens=49,
        seconds=1.5,
    )


def _rich_outcome(file: Path) -> ImageOutcome:
    """Build an ImageOutcome with the EXIF/merged-keyword fields the CSV report reads."""
    return ImageOutcome(
        file=file,
        success=True,
        from_cache=False,
        retry=False,
        title="A Title",
        description="A description.",
        keywords=["Beach", "Sunset"],
        input_tokens=42,
        output_tokens=7,
        total_tokens=49,
        seconds=1.5,
        written_keywords=["Beach", "Sunset"],
        hierarchical_keywords=["Nature|Beach"],
        existing_keywords=["Old"],
        camera_info={"EXIF:Model": "Canon EOS R5", "EXIF:LensModel": "RF 100mm"},
        location_tags={"XMP-photoshop:City": "Hamburg", "XMP-photoshop:Country": "Germany"},
        gps_position="53 N, 9 E",
    )


def test_ndjson_emitter_writes_one_line_per_outcome(tmp_path: Path) -> None:
    """Each call to the emitter writes exactly one JSON line that round-trips through json.loads."""
    buf = io.StringIO()
    emitter = main_module._NDJSONEmitter(buf)  # noqa: SLF001
    emitter(_outcome(tmp_path / "a.cr3", success=True, from_cache=False))
    emitter(_outcome(tmp_path / "b.cr3", success=False, from_cache=True))

    lines = buf.getvalue().splitlines()
    expected_lines = 2
    assert len(lines) == expected_lines
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["status"] == "ok"
    assert first["from_cache"] is False
    assert first["title"] == "A Title"
    assert first["keywords"] == ["Beach", "Sunset"]
    assert second["status"] == "failed"
    assert second["from_cache"] is True


def test_ndjson_emitter_is_thread_safe(tmp_path: Path) -> None:
    """Concurrent emitters never interleave a partial line."""
    buf = io.StringIO()
    emitter = main_module._NDJSONEmitter(buf)  # noqa: SLF001
    paths = [tmp_path / f"img{i:03d}.cr3" for i in range(60)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda p: emitter(_outcome(p)), paths))

    lines = buf.getvalue().splitlines()
    # Each line parses cleanly: proof that nothing interleaved.
    decoded = [json.loads(line) for line in lines]
    assert sorted(d["file"] for d in decoded) == sorted(str(p) for p in paths)


def test_cli_json_flag_installs_ndjson_emitter(tmp_path: Path) -> None:
    """--json wires an _NDJSONEmitter onto run_batch's on_image_result callback."""
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--json"])

    emitter = captured["on_image_result"]
    assert isinstance(emitter, main_module._NDJSONEmitter)  # noqa: SLF001


def test_cli_default_does_not_install_ndjson_emitter(tmp_path: Path) -> None:
    """Without --json the on_image_result callback stays None."""
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image)])

    assert captured["on_image_result"] is None


# ---------------------------------------------------------------------------
# --csv-file report
# ---------------------------------------------------------------------------


def test_outcome_to_report_row_maps_fields(tmp_path: Path) -> None:
    """An ImageOutcome maps onto the report row, EXIF dicts resolved to scalar columns."""
    rendered = main_module._outcome_to_report_row(_rich_outcome(tmp_path / "img.cr3")).as_dict()  # noqa: SLF001
    assert rendered["filename"] == "img.cr3"
    assert rendered["status"] == "ok"
    assert rendered["title"] == "A Title"
    assert rendered["keywords"] == "Beach; Sunset"
    assert rendered["hierarchical_keywords"] == "Nature|Beach"
    assert rendered["existing_keywords"] == "Old"
    assert rendered["camera_model"] == "Canon EOS R5"
    assert rendered["lens_model"] == "RF 100mm"
    assert rendered["gps_position"] == "53 N, 9 E"
    assert rendered["city"] == "Hamburg"
    assert rendered["country"] == "Germany"
    assert rendered["from_cache"] == "false"


def test_combine_image_result_callbacks_handles_zero_one_and_many(tmp_path: Path) -> None:
    """Combining returns None for no sinks, the lone sink for one, and a fan-out for many."""
    seen: list[tuple[str, ImageOutcome]] = []

    def first(outcome: ImageOutcome) -> None:
        seen.append(("first", outcome))

    def second(outcome: ImageOutcome) -> None:
        seen.append(("second", outcome))

    assert main_module._combine_image_result_callbacks(None, None) is None  # noqa: SLF001
    assert main_module._combine_image_result_callbacks(None, first) is first  # noqa: SLF001

    combined = main_module._combine_image_result_callbacks(first, second)  # noqa: SLF001
    assert combined is not None
    outcome = _rich_outcome(tmp_path / "img.cr3")
    combined(outcome)
    assert seen == [("first", outcome), ("second", outcome)]


def test_open_csv_report_returns_none_for_none_path() -> None:
    """No --csv-file means no writer, so nothing to open or close."""
    assert main_module._open_csv_report(None) is None  # noqa: SLF001


def test_open_csv_report_degrades_on_open_error(tmp_path: Path) -> None:
    """A writer that cannot open is logged and skipped, never raised."""
    with patch.object(main_module, "CsvReportWriter", side_effect=OSError("denied")):
        assert main_module._open_csv_report(tmp_path / "report.csv") is None  # noqa: SLF001


def test_cli_csv_file_writes_streamed_report(tmp_path: Path) -> None:
    """--csv-file produces a header plus one row per photo the pipeline reports."""
    import csv as csv_module  # noqa: PLC0415 - test-local parser.

    image = _make_jpeg(tmp_path / "img.cr3")
    csv_path = tmp_path / "report.csv"

    def fake_run_batch(
        image_files: list[Path],
        agent: object,
        options: object,
        **kwargs: object,
    ) -> object:
        callback = kwargs.get("on_image_result")
        assert callback is not None
        callback(_rich_outcome(image))  # type: ignore[operator]
        return None

    with (
        patch.object(main_module, "setup_logging"),
        patch.object(main_module, "create_agent", return_value=object()),
        patch.object(main_module, "run_batch", side_effect=fake_run_batch),
    ):
        _run_app(["--input", str(image), "--csv-file", str(csv_path)])

    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv_module.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["filename"] == "img.cr3"
    assert rows[0]["title"] == "A Title"
    assert rows[0]["keywords"] == "Beach; Sunset"
    assert rows[0]["camera_model"] == "Canon EOS R5"
    assert rows[0]["city"] == "Hamburg"
    assert rows[0]["status"] == "ok"


def test_cli_csv_file_installs_csv_sink_and_writes_header(tmp_path: Path) -> None:
    """--csv-file alone wires a _CsvImageResultSink and emits the header even with no rows."""
    image = _make_jpeg(tmp_path / "img.cr3")
    csv_path = tmp_path / "report.csv"
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--csv-file", str(csv_path)])

    assert isinstance(captured["on_image_result"], main_module._CsvImageResultSink)  # noqa: SLF001
    assert csv_path.exists()


def test_cli_csv_and_json_install_fan_out(tmp_path: Path) -> None:
    """--json with --csv-file fans each outcome out to both sinks."""
    image = _make_jpeg(tmp_path / "img.cr3")
    csv_path = tmp_path / "report.csv"
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--json", "--csv-file", str(csv_path)])

    callback = captured["on_image_result"]
    assert callable(callback)
    # Neither single sink: the combiner wrapped both behind one callable.
    assert not isinstance(callback, main_module._NDJSONEmitter)  # noqa: SLF001
    assert not isinstance(callback, main_module._CsvImageResultSink)  # noqa: SLF001


def test_cli_newer_than_filters_input_batch(tmp_path: Path) -> None:
    """--newer-than parses the timestamp and drops files older than the bound."""
    import os  # noqa: PLC0415 - test-local import.
    from datetime import UTC, datetime, timedelta  # noqa: PLC0415 - test-local import.

    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}
    boundary = datetime(2024, 1, 1, tzinfo=UTC)
    old_ts = (boundary - timedelta(days=10)).timestamp()
    os.utime(image, (old_ts, old_ts))

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--newer-than", "2024-01-01"])

    # The lone file is older than the bound so run_batch should never be invoked.
    assert "image_files" not in captured


def test_cli_rejects_malformed_newer_than(tmp_path: Path) -> None:
    """--newer-than with a non-ISO string exits before scheduling work."""
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch, pytest.raises(SystemExit):
        main_module.app(["--input", str(image), "--newer-than", "not-a-date"])


def test_parse_filter_date_treats_naive_as_local_time() -> None:
    """A naive ISO date attaches the system local zone, not UTC."""
    from datetime import datetime  # noqa: PLC0415 - test-local import.

    parsed = main_module._parse_filter_date("2024-01-01T00:00:00", flag="--newer-than")  # noqa: SLF001
    assert parsed is not None
    assert parsed.tzinfo is not None
    # Wall-clock fields are preserved verbatim: the user wrote midnight local.
    naive_expected = datetime(2024, 1, 1, 0, 0, 0)  # noqa: DTZ001 - naive on purpose.
    assert parsed.replace(tzinfo=None) == naive_expected
    # Attached offset matches the system's local offset for *that* wall-clock
    # time (which may differ from "now" across DST boundaries). Comparing
    # against datetime.astimezone() of the same naive moment guards against
    # the parser silently falling back to UTC.
    assert parsed.utcoffset() == naive_expected.astimezone().utcoffset()


def test_parse_filter_date_preserves_explicit_timezone() -> None:
    """An ISO string that already carries a timezone is passed through unchanged."""
    parsed = main_module._parse_filter_date("2024-01-01T00:00:00+00:00", flag="--newer-than")  # noqa: SLF001
    assert parsed is not None
    offset = parsed.utcoffset()
    assert offset is not None
    assert offset.total_seconds() == 0


def test_parse_filter_date_returns_none_for_none() -> None:
    """Passing None short-circuits without raising."""
    assert main_module._parse_filter_date(None, flag="--newer-than") is None  # noqa: SLF001


def test_cli_skips_cache_when_open_fails(tmp_path: Path) -> None:
    """--cache-file with an unusable target still lets the batch run (no cache)."""
    image = _make_jpeg(tmp_path / "img.cr3")
    cache_path = tmp_path / "cache.sqlite3"
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with (
        setup,
        create_agent,
        run_batch,
        patch("photo_tagger.cache.InferenceCache", side_effect=OSError("denied")),
    ):
        _run_app(["--input", str(image), "--cache-file", str(cache_path)])

    # run_batch was invoked, and the cache kwarg fell through as None.
    assert captured.get("image_files") == [image.resolve()]
    assert captured.get("cache") is None


def test_cli_cache_file_opened_is_passed_and_closed(tmp_path: Path) -> None:
    """A usable --cache-file opens a cache, hands it to run_batch, and closes it in the finally."""
    image = _make_jpeg(tmp_path / "img.cr3")
    cache_path = tmp_path / "cache.sqlite3"
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--cache-file", str(cache_path)])

    # The cache opened (reached run_batch) and the finally closed it without error.
    assert captured["cache"] is not None
    assert cache_path.exists()


def test_cli_lock_file_blocks_second_run(tmp_path: Path) -> None:
    """A second --lock-file invocation while another holds the lock exits with code 1."""
    from photo_tagger.locking import FileLock  # noqa: PLC0415 - test-local import.

    image = _make_jpeg(tmp_path / "img.cr3")
    lock_path = tmp_path / "run.lock"
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    # Hold the lock in this thread, then invoke the CLI which should fail to acquire.
    with FileLock(lock_path), setup, create_agent, run_batch, pytest.raises(SystemExit):
        main_module.app(["--input", str(image), "--lock-file", str(lock_path)])

    # run_batch never ran because the CLI bailed out at lock acquisition.
    assert "image_files" not in captured


def test_cli_lock_file_open_failure_exits(tmp_path: Path) -> None:
    """An OSError while opening the lock file (e.g. unwritable dir) exits with code 1."""
    image = _make_jpeg(tmp_path / "img.cr3")
    lock_path = tmp_path / "run.lock"
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with (
        setup,
        create_agent,
        run_batch,
        patch.object(main_module, "FileLock", side_effect=OSError("disk full")),
        pytest.raises(SystemExit),
    ):
        main_module.app(["--input", str(image), "--lock-file", str(lock_path)])

    assert "image_files" not in captured


_EXPECTED_TOTAL_TOKENS = 49


def test_cli_summary_file_written_on_completion(tmp_path: Path) -> None:
    """--summary-file receives a JSON payload with run totals after the batch finishes."""
    from photo_tagger.pipeline import BatchTotals  # noqa: PLC0415 - test-local import.

    image = _make_jpeg(tmp_path / "img.cr3")
    summary = tmp_path / "summary.json"
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    fake_totals = BatchTotals(
        total_files=1,
        success=1,
        successful_files=[image.name],
        input_tokens=42,
        output_tokens=7,
        total_tokens=_EXPECTED_TOTAL_TOKENS,
        inference_calls=1,
    )

    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--summary-file", str(summary)])
        # Simulate run_batch's on_complete callback firing with realistic totals.
        on_complete = captured["on_complete"]
        assert on_complete is not None
        on_complete(fake_totals)

    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["total_tokens"] == _EXPECTED_TOTAL_TOKENS
    assert payload["successful_files"] == [image.name]
    assert payload["model"]  # provider/model fields are populated.
    assert "started_at" in payload
    assert "finished_at" in payload


def test_cli_summary_file_creates_missing_parent_dir(tmp_path: Path) -> None:
    """--summary-file pointing into a not-yet-existing folder is created transparently."""
    from photo_tagger.pipeline import BatchTotals  # noqa: PLC0415 - test-local import.

    image = _make_jpeg(tmp_path / "img.cr3")
    nested = tmp_path / "reports" / "today" / "summary.json"
    assert not nested.parent.exists()
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    fake_totals = BatchTotals(total_files=1, success=1, successful_files=[image.name])

    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--summary-file", str(nested)])
        captured["on_complete"](fake_totals)

    assert nested.exists()
    payload = json.loads(nested.read_text(encoding="utf-8"))
    assert payload["total_files"] == 1


# ---------------------------------------------------------------------------
# _read_prompt_file edge cases
# ---------------------------------------------------------------------------


def test_read_prompt_file_exits_on_empty_file(tmp_path: Path) -> None:
    """A prompt file that contains only whitespace is rejected with SystemExit."""
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("   \n  ", encoding="utf-8")
    with pytest.raises(SystemExit):
        main_module._read_prompt_file(prompt)  # noqa: SLF001


def test_read_prompt_file_exits_on_read_error(tmp_path: Path) -> None:
    """An unreadable prompt file triggers SystemExit."""
    prompt = tmp_path / "missing.txt"
    with pytest.raises(SystemExit):
        main_module._read_prompt_file(prompt)  # noqa: SLF001


def test_read_prompt_file_returns_default_when_none() -> None:
    """Passing None returns the built-in default prompt."""
    from photo_tagger.config import DEFAULT_USER_PROMPT  # noqa: PLC0415

    assert main_module._read_prompt_file(None) == DEFAULT_USER_PROMPT  # noqa: SLF001


# ---------------------------------------------------------------------------
# _write_summary_file edge cases
# ---------------------------------------------------------------------------


def test_write_summary_file_noop_when_path_is_none(tmp_path: Path) -> None:
    """summary_file=None is the normal no-write path."""
    from datetime import UTC, datetime  # noqa: PLC0415

    # Should not raise or create any file.
    main_module._write_summary_file(  # noqa: SLF001
        None,
        BatchTotals(),
        started_at=datetime.now(tz=UTC),
        model_name="m",
        provider_name="p",
        user_prompt_chars=0,
    )


def test_write_summary_file_noop_when_totals_is_none(tmp_path: Path) -> None:
    """``totals=None`` is the early-exit path when the batch never ran."""
    from datetime import UTC, datetime  # noqa: PLC0415

    dest = tmp_path / "out.json"
    main_module._write_summary_file(  # noqa: SLF001
        dest,
        None,
        started_at=datetime.now(tz=UTC),
        model_name="m",
        provider_name="p",
        user_prompt_chars=0,
    )
    assert not dest.exists()


def test_write_summary_file_swallows_write_error(tmp_path: Path) -> None:
    """An OSError during write is logged, not raised."""
    from datetime import UTC, datetime  # noqa: PLC0415

    # Point at a directory path so write fails.
    dest = tmp_path / "dir_not_file"
    dest.mkdir()
    dest = dest / "nested" / "summary.json"
    with patch.object(main_module, "_atomic_write_text", side_effect=OSError("boom")):
        # Must not raise.
        main_module._write_summary_file(  # noqa: SLF001
            dest,
            BatchTotals(),
            started_at=datetime.now(tz=UTC),
            model_name="m",
            provider_name="p",
            user_prompt_chars=0,
        )


# ---------------------------------------------------------------------------
# _atomic_write_text edge cases
# ---------------------------------------------------------------------------


def test_atomic_write_text_cleans_up_on_write_failure(tmp_path: Path) -> None:
    """If the write to the temp file fails, the temp file is removed and the error re-raised."""
    target = tmp_path / "output.json"

    # Patch os.fdopen to raise after mkstemp creates the temp file.
    with (
        patch("photo_tagger.main.os.fdopen", side_effect=OSError("disk full")),
        pytest.raises(OSError, match="disk full"),
    ):
        main_module._atomic_write_text(target, "content")  # noqa: SLF001

    # Target must not exist, and no temp files should be left behind.
    assert not target.exists()
    leftovers = list(tmp_path.glob(f".{target.name}.*"))
    assert leftovers == []


def test_atomic_write_text_cleans_up_when_rename_fails(tmp_path: Path) -> None:
    """
    A failure after the temp file was fully written (the rename) also removes it.

    Unlike the fdopen failure above, the descriptor is already closed here, so this exercises the
    branch that skips the extra close before unlinking.
    """
    target = tmp_path / "output.json"

    with (
        patch("photo_tagger.main.Path.replace", side_effect=OSError("read-only filesystem")),
        pytest.raises(OSError, match="read-only"),
    ):
        main_module._atomic_write_text(target, "content")  # noqa: SLF001

    assert not target.exists()
    assert list(tmp_path.glob(f".{target.name}.*")) == []


# ---------------------------------------------------------------------------
# doctor command
# ---------------------------------------------------------------------------


def test_doctor_exits_zero_when_all_checks_pass() -> None:
    """A clean checklist returns normally (cyclopts treats no SystemExit as exit 0)."""
    from photo_tagger.diagnostics import CheckResult  # noqa: PLC0415 - test-local import.

    ok = [CheckResult("ExifTool", ok=True, detail="/usr/bin/exiftool")]
    with patch.object(main_module, "run_checks", return_value=ok):
        # No SystemExit means success; _run_app would swallow one if raised.
        main_module.doctor(provider="lmstudio", model="m", url=None, api_key=None)


def test_doctor_exits_one_when_a_check_fails() -> None:
    """A failing check makes the command raise SystemExit(1)."""
    from photo_tagger.diagnostics import CheckResult  # noqa: PLC0415 - test-local import.

    bad = [CheckResult("ExifTool", ok=False, detail="missing")]
    with (
        patch.object(main_module, "run_checks", return_value=bad),
        pytest.raises(SystemExit) as exc_info,
    ):
        main_module.doctor(provider="lmstudio", model="m", url=None, api_key=None)
    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# gui command (lazy PySide6 import)
# ---------------------------------------------------------------------------


def test_gui_reports_missing_pyside6(capsys: pytest.CaptureFixture[str]) -> None:
    """Without the optional extra, the command exits 1 with a pip install hint."""

    def raise_import_error(_name: str) -> object:
        msg = "No module named 'PySide6'"
        raise ImportError(msg, name="PySide6")

    with (
        patch("photo_tagger.main.importlib.import_module", raise_import_error),
        pytest.raises(SystemExit) as exc_info,
    ):
        main_module.gui()
    assert exc_info.value.code == 1
    assert "photo-tagger[gui]" in capsys.readouterr().err


def test_gui_propagates_non_qt_import_errors(capsys: pytest.CaptureFixture[str]) -> None:
    """
    An ImportError from inside the gui module chain is not "PySide6 missing".

    Regression test: a broken transitive dependency used to be reported with the pip install
    hint, sending the user to reinstall an extra that was never the problem.
    """

    def raise_import_error(_name: str) -> object:
        msg = "cannot import name 'Broken' from 'somewhere.else'"
        raise ImportError(msg, name="somewhere.else")

    with (
        patch("photo_tagger.main.importlib.import_module", raise_import_error),
        pytest.raises(ImportError, match=r"somewhere\.else"),
    ):
        main_module.gui()
    assert "photo-tagger[gui]" not in capsys.readouterr().err


def test_gui_launches_when_available() -> None:
    """When PySide6 is present, the command delegates to gui.launch() and exits with its code."""
    fake_gui = type("FakeGui", (), {"launch": staticmethod(lambda: 0)})
    with (
        patch("photo_tagger.main.importlib.import_module", return_value=fake_gui),
        pytest.raises(SystemExit) as exc_info,
    ):
        main_module.gui()
    assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# telemetry wiring
# ---------------------------------------------------------------------------


def _run_batch_firing_complete(
    image_files: list[Path],
    _agent: object,
    _options: object,
    **kwargs: object,
) -> object:
    """Stand-in for run_batch that drives the on_complete callback so the beacon fires."""
    on_complete = kwargs.get("on_complete")
    if on_complete is not None:
        on_complete(BatchTotals(total_files=len(image_files)))  # type: ignore[operator]
    return None


def test_cli_emits_telemetry_on_completion(tmp_path: Path) -> None:
    """A successful run reports interface, provider, model, and batch size to telemetry."""
    image = _make_jpeg(tmp_path / "img.cr3")
    emitted: dict[str, Any] = {}

    def fake_emit(run: Any, *, enabled: bool, block: bool = False) -> None:  # noqa: ANN401
        emitted["run"] = run
        emitted["enabled"] = enabled
        emitted["block"] = block

    with (
        patch.object(main_module, "setup_logging"),
        patch.object(main_module, "create_agent", return_value=object()),
        patch.object(main_module, "run_batch", side_effect=_run_batch_firing_complete),
        patch.object(telemetry, "emit", side_effect=fake_emit),
    ):
        _run_app(["--input", str(image), "--provider", "ollama", "--model", "my-vlm"])

    assert emitted["enabled"] is True
    assert emitted["block"] is True
    assert emitted["run"].interface == "cli"
    assert emitted["run"].provider == "ollama"
    assert emitted["run"].model == "my-vlm"
    assert emitted["run"].batch_size == 1


def test_cli_no_telemetry_flag_disables_and_silences_notice(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--no-telemetry passes enabled=False to emit and suppresses the first-run notice."""
    image = _make_jpeg(tmp_path / "img.cr3")
    emitted: dict[str, Any] = {}

    def fake_emit(_run: Any, *, enabled: bool, block: bool = False) -> None:  # noqa: ANN401
        emitted["enabled"] = enabled

    with (
        patch.object(main_module, "setup_logging"),
        patch.object(main_module, "create_agent", return_value=object()),
        patch.object(main_module, "run_batch", side_effect=_run_batch_firing_complete),
        patch.object(telemetry, "emit", side_effect=fake_emit),
    ):
        _run_app(["--input", str(image), "--no-telemetry"])

    assert emitted["enabled"] is False
    assert "anonymous usage stats" not in capsys.readouterr().err


def test_cli_telemetry_notice_shown_only_on_first_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With telemetry on (the default), the disclosure prints once and stays quiet after."""
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image)])
        first_run_err = capsys.readouterr().err
        _run_app(["--input", str(image)])
        second_run_err = capsys.readouterr().err

    assert "anonymous usage stats" in first_run_err
    assert "anonymous usage stats" not in second_run_err


def _point_config_at(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    """Write *text* to a TOML file and point PHOTO_TAGGER_CONFIG at it for this test."""
    cfg = tmp_path / "config.toml"
    cfg.write_text(text, encoding="utf-8")
    monkeypatch.setenv("PHOTO_TAGGER_CONFIG", str(cfg))


def test_config_file_fills_flags_the_user_did_not_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config values reach the pipeline when the corresponding flags are absent."""
    _point_config_at(
        tmp_path,
        monkeypatch,
        "workers = 2\n\n[inference]\nmax_tokens = 500\n\n[output]\nbackup_xmp = false\n",
    )
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image)])

    assert captured["options"].max_tokens == 500  # noqa: PLR2004
    assert captured["options"].backup_xmp is False
    assert captured["workers"] == 2  # noqa: PLR2004


def test_config_file_survives_sibling_cli_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Passing one flag from a group must not reset the group's other fields to built-ins.

    Regression test: config defaults used to be baked into the default group instances, and
    cyclopts rebuilds a group from class defaults whenever any of its flags is passed, silently
    dropping the config values of every sibling field.
    """
    _point_config_at(tmp_path, monkeypatch, "[inference]\nmax_tokens = 500\n")
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--temperature", "0.9"])

    assert captured["options"].temperature == 0.9  # noqa: PLR2004
    assert captured["options"].max_tokens == 500  # noqa: PLR2004


def test_cli_flag_overrides_config_file_for_the_same_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit flag beats the config file for that field."""
    _point_config_at(tmp_path, monkeypatch, "[inference]\nmax_tokens = 500\n")
    image = _make_jpeg(tmp_path / "img.cr3")
    captured: dict[str, Any] = {}

    setup, create_agent, run_batch = _patches(captured)
    with setup, create_agent, run_batch:
        _run_app(["--input", str(image), "--max-tokens", "800"])

    assert captured["options"].max_tokens == 800  # noqa: PLR2004


def test_cli_config_overrides_translates_field_names_to_option_names() -> None:
    """Renamed fields map to their CLI spelling; unknown keys and tables are dropped."""
    from photo_tagger.cli_options import cli_config_overrides  # noqa: PLC0415

    flat = cli_config_overrides(
        {
            "provider": {"model_name": "my-vlm", "api_base_url": "http://h:1/v1", "nope": 1},
            "output": {"use_sidecar": False},
            "telemetry": {"enabled": False},
            "display": {"progress_bar": False, "json_output": True},
            "workers": 4,
            "recursive": True,
            "exiftool_path": "/x/exiftool",
            "future_table": {"key": "value"},
        },
    )

    assert flat == {
        "model": "my-vlm",
        "url": "http://h:1/v1",
        "write-sidecar": False,
        "telemetry": False,
        "progress": False,
        "json": True,
        "workers": 4,
        "recursive": True,
    }


def test_main_reports_unhandled_crashes_and_re_raises(tmp_path: Path) -> None:
    """
    A crash escaping the CLI fires one anonymous crash beacon and still surfaces the traceback.

    SystemExit (clean error handling) must never be reported as a crash.
    """
    image = _make_jpeg(tmp_path / "img.cr3")
    crashes: list[dict[str, Any]] = []

    def fake_emit_crash(exc: BaseException, **kwargs: Any) -> None:  # noqa: ANN401
        crashes.append({"exc": exc, **kwargs})

    with (
        patch.object(main_module, "setup_logging"),
        patch.object(main_module, "create_agent", return_value=object()),
        patch.object(main_module, "run_batch", side_effect=RuntimeError("boom")),
        patch.object(telemetry, "emit_crash", side_effect=fake_emit_crash),
        patch("sys.argv", ["photo-tagger", "--input", str(image)]),
        pytest.raises(RuntimeError, match="boom"),
    ):
        main_module.main()

    assert len(crashes) == 1
    assert isinstance(crashes[0]["exc"], RuntimeError)
    assert crashes[0]["interface"] == "cli"


def test_main_does_not_report_clean_exits_as_crashes() -> None:
    """SystemExit from normal error handling passes through without a crash beacon."""
    crashes: list[object] = []
    with (
        patch.object(telemetry, "emit_crash", side_effect=lambda *a, **_k: crashes.append(a)),
        patch.object(main_module, "app", side_effect=SystemExit(1)),
        pytest.raises(SystemExit),
    ):
        main_module.main()
    assert crashes == []


def test_crash_telemetry_enabled_honors_argv_flag() -> None:
    """A --no-telemetry anywhere on the command line disables the crash beacon too."""
    with patch("sys.argv", ["photo-tagger", "-i", "x", "--no-telemetry"]):
        assert main_module._crash_telemetry_enabled() is False  # noqa: SLF001
    with patch("sys.argv", ["photo-tagger", "-i", "x"]):
        assert main_module._crash_telemetry_enabled() is True  # noqa: SLF001


def test_suite_is_isolated_from_developer_config() -> None:
    """
    Conftest points PHOTO_TAGGER_CONFIG at an empty file, so defaults are the built-ins.

    Guards that isolation: a real ~/.config/photo-tagger/config.toml must not leak into the suite.
    (A developer config setting a relative ``summary_file`` once wrote a stray file into the repo
    when the on_complete tests fired.)
    """
    defaults = load_defaults()
    assert defaults.artifacts.summary_file is None
    assert defaults.telemetry.enabled is True


def test_load_defaults_reads_exiftool_path() -> None:
    """A config-file exiftool_path surfaces on Defaults; absent means None."""
    assert load_defaults({"exiftool_path": "/x/exiftool"}).exiftool_path == "/x/exiftool"
    assert load_defaults({}).exiftool_path is None


def test_apply_exiftool_path_bridges_to_env_without_overriding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fill an unset env var from config; never override an exported one; no-op on None."""
    import os  # noqa: PLC0415 - test-local

    monkeypatch.delenv("PHOTO_TAGGER_EXIFTOOL", raising=False)
    main_module._apply_exiftool_path("/cfg/exiftool")  # noqa: SLF001
    assert os.environ["PHOTO_TAGGER_EXIFTOOL"] == "/cfg/exiftool"

    monkeypatch.setenv("PHOTO_TAGGER_EXIFTOOL", "/env/exiftool")
    main_module._apply_exiftool_path("/cfg/exiftool")  # noqa: SLF001
    assert os.environ["PHOTO_TAGGER_EXIFTOOL"] == "/env/exiftool"

    monkeypatch.delenv("PHOTO_TAGGER_EXIFTOOL", raising=False)
    main_module._apply_exiftool_path(None)  # noqa: SLF001
    assert "PHOTO_TAGGER_EXIFTOOL" not in os.environ
