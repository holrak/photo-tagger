"""Tests for grouping a batch into shoots and harmonizing each shoot's keywords."""

import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from photo_tagger.sessions import (
    SessionPlan,
    _most_common_chain,
    build_session_vocabulary,
    plan_sessions,
)
from photo_tagger.vocabulary import Vocabulary


# EXIF capture times are camera clock readings, so every timestamp here is naive on purpose.
_DAY_START = datetime(2026, 5, 1, 9, 0)  # noqa: DTZ001


def _touch(path: Path, *, mtime: datetime | None = None) -> Path:
    """Create a file, optionally stamping its mtime so the mtime fallback is testable."""
    path.write_text("x")
    if mtime is not None:
        stamp = mtime.timestamp()
        os.utime(path, (stamp, stamp))
    return path


def _at(minute: int) -> datetime:
    """Return a capture time *minute* minutes after a fixed start of day."""
    return _DAY_START + timedelta(minutes=minute)


def test_plan_sessions_is_off_for_a_zero_gap(tmp_path: Path) -> None:
    """A gap of zero means the feature is disabled, so there is no plan at all."""
    photo = _touch(tmp_path / "a.cr3")
    assert plan_sessions([photo], gap_minutes=0) is None
    assert plan_sessions([photo], gap_minutes=-5) is None
    assert plan_sessions([], gap_minutes=30) is None


def test_plan_sessions_splits_on_a_capture_time_gap(tmp_path: Path) -> None:
    """Frames close in time are one shoot; a long pause starts the next."""
    first = _touch(tmp_path / "a.cr3")
    second = _touch(tmp_path / "b.cr3")
    third = _touch(tmp_path / "c.cr3")
    times = {first: _at(0), second: _at(5), third: _at(120)}

    with patch("photo_tagger.sessions.read_capture_times", return_value=times):
        plan = plan_sessions([first, second, third], gap_minutes=30)

    assert plan is not None
    assert plan.sessions == [[first, second], [third]]
    assert plan.index_of == {first: 0, second: 0, third: 1}


def test_plan_sessions_orders_photos_by_capture_time(tmp_path: Path) -> None:
    """The batch is reordered into shooting order, whatever order it arrived in."""
    late = _touch(tmp_path / "late.cr3")
    early = _touch(tmp_path / "early.cr3")
    times = {late: _at(10), early: _at(1)}

    with patch("photo_tagger.sessions.read_capture_times", return_value=times):
        plan = plan_sessions([late, early], gap_minutes=30)

    assert plan is not None
    assert plan.sessions == [[early, late]]


def test_plan_sessions_falls_back_to_file_mtime(tmp_path: Path) -> None:
    """A photo with no EXIF capture time is grouped by when the file was last written."""
    scanned = _touch(tmp_path / "scan.jpg", mtime=_at(0))
    later = _touch(tmp_path / "later.jpg", mtime=_at(180))

    with patch("photo_tagger.sessions.read_capture_times", return_value={}):
        plan = plan_sessions([scanned, later], gap_minutes=30)

    assert plan is not None
    assert plan.sessions == [[scanned], [later]]


def test_plan_sessions_isolates_photos_with_no_readable_timestamp(tmp_path: Path) -> None:
    """A file we cannot stat gets its own group instead of joining a real shoot."""
    real = _touch(tmp_path / "real.cr3", mtime=_at(0))
    missing = tmp_path / "gone.cr3"

    with patch("photo_tagger.sessions.read_capture_times", return_value={}):
        plan = plan_sessions([real, missing], gap_minutes=30)

    assert plan is not None
    assert plan.sessions == [[missing], [real]]


def test_session_plan_returns_a_vocabulary_only_once_remembered(tmp_path: Path) -> None:
    """Before its session is harmonized a photo has no vocabulary to snap onto."""
    photo = tmp_path / "a.cr3"
    plan = SessionPlan(sessions=[[photo]], index_of={photo: 0})

    assert plan.vocabulary_for(photo) is None
    assert plan.vocabulary_for(tmp_path / "other.cr3") is None

    plan.remember(0, Vocabulary.from_entries(["Osprey"]))
    vocabulary = plan.vocabulary_for(photo)
    assert vocabulary is not None
    assert vocabulary.match("ospreys") == "Osprey"


def test_build_session_vocabulary_picks_the_majority_spelling() -> None:
    """Two spellings of one concept collapse onto the one the session used most."""
    vocabulary = build_session_vocabulary(
        [["Osprey"], ["Osprey"], ["ospreys"], ["Golden Hour"]],
    )
    assert vocabulary.match("ospreys") == "Osprey"
    assert vocabulary.match("Golden Hour") == "Golden Hour"


def test_build_session_vocabulary_picks_the_majority_hierarchy() -> None:
    """A leaf filed under two parents ends up under the one most of the shoot used."""
    vocabulary = build_session_vocabulary(
        [
            ["Osprey<Bird<Animal"],
            ["Osprey<Bird<Animal"],
            ["Osprey<Raptor<Wildlife"],
        ],
    )
    assert vocabulary.chain_for("Osprey") == ["Animal", "Bird", "Osprey"]


def test_build_session_vocabulary_harmonizes_a_whole_session() -> None:
    """Snapping each photo back through the result makes the session agree with itself."""
    photos = [
        ["Osprey<Bird<Animal", "Reflection"],
        ["ospreys<Raptor", "Reflections"],
    ]
    vocabulary = build_session_vocabulary(photos)

    assert [vocabulary.snap(keywords).keywords for keywords in photos] == [
        ["Osprey<Bird<Animal", "Reflection"],
        ["Osprey<Bird<Animal", "Reflection"],
    ]


def test_build_session_vocabulary_breaks_spelling_ties_alphabetically() -> None:
    """A tie cannot be left to insertion order, or two runs disagree on the same input."""
    assert build_session_vocabulary([["Sunset"], ["sunset"]]).match("SUNSET") == "Sunset"
    assert build_session_vocabulary([["sunset"], ["Sunset"]]).match("SUNSET") == "Sunset"


def test_build_session_vocabulary_prefers_the_deeper_chain_on_a_tie() -> None:
    """With no majority, the hierarchy that says more wins."""
    vocabulary = build_session_vocabulary([["Osprey<Bird"], ["Osprey<Bird<Animal"]])
    assert vocabulary.chain_for("Osprey") == ["Animal", "Bird", "Osprey"]


def test_build_session_vocabulary_ignores_unusable_keywords() -> None:
    """Blank, punctuation-only, and separator-only keywords contribute nothing."""
    assert not build_session_vocabulary([["", "   ", "<<<", "..."]])


def test_most_common_chain_is_deterministic_for_equal_depth_ties() -> None:
    """Equal count and equal depth fall back to the chain itself, not insertion order."""
    from collections import Counter  # noqa: PLC0415 - only this test needs the type

    counter: Counter[tuple[str, ...]] = Counter(
        {("Wildlife", "Osprey"): 1, ("Animal", "Osprey"): 1},
    )
    assert _most_common_chain(counter) == ("Wildlife", "Osprey")
