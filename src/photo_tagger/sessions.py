"""
Group a batch into shoots and make each shoot's keywords agree with itself.

Every photo is analyzed on its own, so forty frames of the same bird can come back as "Osprey" here
and "Ospreys" there, under "Bird<Animal" on one frame and "Raptor<Wildlife" on the next. Lightroom
then shows four keywords where there is one subject.

The fix has two halves. :func:`plan_sessions` splits the batch where the capture time jumps by more
than the configured gap, which is what a "shoot" actually is. :func:`build_session_vocabulary` then
turns everything the model said about one session into a :class:`~photo_tagger.vocabulary.
Vocabulary`, letting the majority spelling and the majority hierarchy win, and the pipeline snaps
every photo in the session onto it before writing.

Deriving the vocabulary from the batch's own output (rather than nudging the model with a prompt) is
what makes this deterministic: the result does not depend on which photo happened to finish first.
"""

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from loguru import logger

from photo_tagger.keywords import parse_hierarchical_keyword
from photo_tagger.metadata import read_capture_times
from photo_tagger.vocabulary import Vocabulary, loose_key


if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from exiftool import ExifToolHelper  # type: ignore[attr-defined]


# Sorts before every real capture time, so photos with no readable timestamp land in one group of
# their own instead of silently joining whichever shoot they happen to sit next to. Naive, like
# every other timestamp here: these are camera clock readings, not instants on a timeline.
_UNKNOWN_TIME = datetime.min  # noqa: DTZ901


@dataclass(slots=True)
class SessionPlan:
    """
    The batch split into shoots, plus the vocabulary each shoot settled on.

    The pipeline fills in the vocabularies as it goes: a session's is only known once every photo in
    it has been analyzed. The retry pass reads them back, so a photo that needed a second attempt
    still lands on the same terms as the rest of its shoot.
    """

    sessions: list[list[Path]] = field(default_factory=list)
    index_of: dict[Path, int] = field(default_factory=dict)
    vocabularies: dict[int, Vocabulary] = field(default_factory=dict)

    def remember(self, index: int, vocabulary: Vocabulary) -> None:
        """Record the vocabulary session *index* settled on."""
        self.vocabularies[index] = vocabulary

    def vocabulary_for(self, path: Path) -> Vocabulary | None:
        """Return the vocabulary of *path*'s session, or None before it has been computed."""
        index = self.index_of.get(path)
        return None if index is None else self.vocabularies.get(index)


def _timestamps(paths: list[Path], *, et: ExifToolHelper | None) -> dict[Path, datetime]:
    """
    Return a capture time for every path, falling back to file mtime.

    EXIF DateTimeOriginal is the camera's own local time and mtime is converted to local time too,
    so the two are comparable: only the gaps between them matter here. A path we cannot stat at all
    gets :data:`_UNKNOWN_TIME`.
    """
    times = read_capture_times(paths, et=et)
    resolved: dict[Path, datetime] = {}
    for path in paths:
        if (captured := times.get(path)) is not None:
            resolved[path] = captured
            continue
        try:
            # Naive local time on purpose, to stay comparable with EXIF's camera-local stamps.
            resolved[path] = datetime.fromtimestamp(path.stat().st_mtime)  # noqa: DTZ006
        except OSError as exc:
            logger.warning("session_timestamp_failed", file=str(path), error=str(exc))
            resolved[path] = _UNKNOWN_TIME
    return resolved


def plan_sessions(
    image_files: list[Path],
    *,
    gap_minutes: float,
    et: ExifToolHelper | None = None,
) -> SessionPlan | None:
    """
    Split *image_files* into shoots separated by more than *gap_minutes* of idle time.

    Returns None when *gap_minutes* is zero or negative (the feature is off) or when there is
    nothing to group, so callers can treat "no plan" as "run the batch as usual". Photos are
    reordered into capture order, which is the order a session's vocabulary is built from.
    """
    if gap_minutes <= 0 or not image_files:
        return None

    times = _timestamps(image_files, et=et)
    ordered = sorted(image_files, key=lambda path: (times[path], str(path)))
    gap_seconds = gap_minutes * 60.0

    sessions: list[list[Path]] = []
    current: list[Path] = []
    previous: datetime | None = None
    for path in ordered:
        captured = times[path]
        if previous is not None and (captured - previous).total_seconds() > gap_seconds:
            sessions.append(current)
            current = []
        current.append(path)
        previous = captured
    # `ordered` is never empty here (an empty batch returned None above), so there is always a
    # final session to close.
    sessions.append(current)

    plan = SessionPlan(
        sessions=sessions,
        index_of={path: index for index, session in enumerate(sessions) for path in session},
    )
    logger.info(
        "sessions_planned",
        sessions=len(sessions),
        photos=len(ordered),
        gap_minutes=gap_minutes,
        largest=max((len(s) for s in sessions), default=0),
    )
    return plan


def _count_keywords(
    keyword_lists: Iterable[Iterable[str]],
) -> tuple[Counter[str], dict[str, Counter[str]], dict[str, Counter[tuple[str, ...]]]]:
    """
    Tally how a session used each concept.

    Returns per-concept totals, the spellings seen for each concept, and the hierarchies seen for
    each concept. A concept is a loose key (case-, punctuation-, and plural-insensitive), so
    "Reflections" and "reflection" are counted as one thing with two spellings.
    """
    totals: Counter[str] = Counter()
    spellings: dict[str, Counter[str]] = {}
    chains: dict[str, Counter[tuple[str, ...]]] = {}
    for keywords in keyword_lists:
        for keyword in keywords:
            _, parts = parse_hierarchical_keyword(keyword)
            if not parts:
                continue
            leaf = parts[-1]
            concept = loose_key(leaf)
            if not concept:
                continue
            totals[concept] += 1
            spellings.setdefault(concept, Counter())[leaf] += 1
            if len(parts) > 1:
                chains.setdefault(concept, Counter())[tuple(parts)] += 1
    return totals, spellings, chains


def _most_common_spelling(counter: Counter[str]) -> str:
    """
    Return the spelling used most often, breaking ties alphabetically.

    Ties are the norm in a short session (two photos, two spellings), so they cannot be left to
    insertion order if the run is to be reproducible.
    """
    return min(counter.items(), key=lambda item: (-item[1], item[0]))[0]


def _most_common_chain(counter: Counter[tuple[str, ...]]) -> tuple[str, ...]:
    """Return the hierarchy used most often, breaking ties toward the deeper chain."""
    return max(counter.items(), key=lambda item: (item[1], len(item[0]), item[0]))[0]


def build_session_vocabulary(keyword_lists: Iterable[Iterable[str]]) -> Vocabulary:
    """
    Derive one session's vocabulary from the keywords its photos generated.

    For every concept the session mentioned, the spelling used most often becomes the canonical one
    and the hierarchy used most often becomes its chain. Concepts are emitted in descending
    frequency so that when two of them share a parent, the busier one's spelling of that parent
    wins.
    """
    totals, spellings, chains = _count_keywords(keyword_lists)
    entries: list[str] = []
    for concept, _count in totals.most_common():
        leaf = _most_common_spelling(spellings[concept])
        if (seen := chains.get(concept)) is not None:
            chain = list(_most_common_chain(seen))
            # The chain carries whatever spelling of the leaf came with it, which is not
            # necessarily the session's majority one.
            chain[-1] = leaf
            entries.append("|".join(chain))
        else:
            entries.append(leaf)
    return Vocabulary.from_entries(entries)
