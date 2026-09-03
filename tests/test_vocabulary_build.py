"""Tests for building a vocabulary out of a library's existing keywords."""

from pathlib import Path
from typing import Any

import pytest

from photo_tagger.errors import DiscoveryError
from photo_tagger.models import KeywordSet
from photo_tagger.vocabulary import Vocabulary, parse_keyword_lines
from photo_tagger.vocabulary_build import (
    DROP_DIGITS,
    DROP_OVER_LIMIT,
    DROP_PUNCTUATION,
    DROP_RARE,
    DROP_TOO_LONG,
    DROP_TOO_MANY_WORDS,
    DROP_VARIANT,
    KeywordCensus,
    TrimRules,
    census_from_export,
    census_from_photos,
    render_drop_report,
    render_vocabulary,
    trim,
    vocabulary_header,
)
from photo_tagger.vocabulary_organize import OrganizeStats


def _census(**uses: int) -> KeywordCensus:
    """Build a flat census straight from term/count pairs."""
    census = KeywordCensus()
    for term, count in uses.items():
        census.add([term.replace("_", " ")], weight=count)
    return census


def test_census_credits_every_level_of_a_hierarchy() -> None:
    """A photo tagged Animal|Bird|Osprey uses all three keywords, not just the leaf."""
    census = KeywordCensus()
    census.add(["Animal", "Bird", "Osprey"])
    census.add(["Animal", "Bird", "Mallard"])
    assert dict(census.uses) == {"Animal": 2, "Bird": 2, "Osprey": 1, "Mallard": 1}
    assert census.best_chain("Osprey") == ("Animal", "Bird", "Osprey")
    assert census.best_chain("Animal") == ()


def test_census_best_chain_prefers_the_most_used_hierarchy() -> None:
    """A leaf filed two ways takes the placement the library used most."""
    census = KeywordCensus()
    census.add(["Animal", "Bird", "Osprey"], weight=5)
    census.add(["Wildlife", "Raptor", "Osprey"], weight=2)
    assert census.best_chain("Osprey") == ("Animal", "Bird", "Osprey")


def test_census_from_photos_counts_one_vote_per_photo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Flat and hierarchical spellings of one keyword on one photo count once."""
    photos = {
        Path("a.jpg"): KeywordSet(
            subject=["Osprey", "Golden Hour"],
            hierarchical=["Animal|Bird|Osprey"],
        ),
        Path("b.jpg"): KeywordSet(subject=["Golden Hour"]),
    }

    def fake_read(paths: Any, **_: Any) -> dict[Path, KeywordSet]:  # noqa: ANN401
        return {path: photos[path] for path in paths}

    monkeypatch.setattr("photo_tagger.vocabulary_build.read_keyword_sets", fake_read)
    census = census_from_photos(list(photos))

    assert census.photos == len(photos)
    assert dict(census.uses) == {"Animal": 1, "Bird": 1, "Osprey": 1, "Golden Hour": 2}
    assert census.best_chain("Osprey") == ("Animal", "Bird", "Osprey")


def test_census_from_photos_reports_a_failed_read_instead_of_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    A broken exiftool must not read as "your library has no keywords".

    read_keyword_sets maps every path it is given, so an empty result means the read never ran.
    Counting it as zero sent the user to "No keywords found: nothing to build a vocabulary from",
    with the real cause only in the log.
    """

    def failed_read(_paths: Any, **_: Any) -> dict[Path, KeywordSet]:  # noqa: ANN401
        return {}

    monkeypatch.setattr("photo_tagger.vocabulary_build.read_keyword_sets", failed_read)

    with pytest.raises(DiscoveryError, match="exiftool"):
        census_from_photos([Path("a.jpg")])


def test_census_from_export_counts_tree_occurrences() -> None:
    """An export has no usage, so a term filed in several places scores several."""
    census = census_from_export("Animal\n\tBird\nWildlife\n\tBird\n\t\tOsprey {Sea Hawk}\n")
    assert dict(census.uses) == {"Animal": 1, "Wildlife": 1, "Bird": 2, "Osprey": 1, "Sea Hawk": 1}


def test_trim_drops_rare_terms_and_keeps_the_rest() -> None:
    """The core rule: a keyword used once is a one-off, not a vocabulary term."""
    result = trim(_census(Bird=10, Osprey=3, Fluke=1), TrimRules(min_uses=2))
    assert result.kept == ["Bird", "Osprey"]
    assert [(d.term, d.reason) for d in result.dropped] == [("Fluke", DROP_RARE)]


@pytest.mark.parametrize(
    ("term", "reason"),
    [
        ("19.5V", DROP_DIGITS),
        ("50% Off", DROP_DIGITS),
        ("Bird/Raptor", DROP_PUNCTUATION),
        ("Man Riding A Bicycle", DROP_TOO_MANY_WORDS),
        ("Supercalifragilisticexpialidocious Bird", DROP_TOO_LONG),
    ],
)
def test_trim_drops_terms_that_are_not_keyword_shaped(term: str, reason: str) -> None:
    """Measurements, paths, and whole sentences are not subjects, however often they occur."""
    census = KeywordCensus()
    census.add([term], weight=99)
    census.add(["Bird"], weight=99)
    result = trim(census)
    assert result.kept == ["Bird"]
    assert [(d.term, d.reason) for d in result.dropped] == [(term, reason)]


def test_trim_keeps_digits_when_asked() -> None:
    """A catalog that really uses "35Mm" can say so."""
    census = KeywordCensus()
    census.add(["35Mm"], weight=5)
    assert trim(census, TrimRules(allow_digits=True)).kept == ["35Mm"]


def test_trim_collapses_variants_onto_the_most_used_spelling() -> None:
    """One concept, one spelling: the catalog's own usage picks the winner."""
    result = trim(_census(Animal=20, Animals=5, animal=2))
    assert result.kept == ["Animal"]
    variants = [(d.term, d.reason, d.detail) for d in result.dropped]
    assert variants == [
        ("Animals", DROP_VARIANT, "kept Animal"),
        ("animal", DROP_VARIANT, "kept Animal"),
    ]


def test_trim_variant_losers_still_match_through_the_vocabulary() -> None:
    """Dropping a spelling costs nothing: matching folds case and plurals anyway."""
    result = trim(_census(Animal=20, Animals=5))
    vocab = Vocabulary.from_entries(result.kept)
    assert vocab.match("Animals") == "Animal"


def test_trim_caps_the_list_at_max_terms_by_usage() -> None:
    """The cap removes the least-used survivors, and says so in the report."""
    result = trim(_census(Bird=10, Osprey=8, Mallard=6, Heron=4), TrimRules(max_terms=2))
    assert result.kept == ["Bird", "Osprey"]
    assert [(d.term, d.reason) for d in result.dropped] == [
        ("Mallard", DROP_OVER_LIMIT),
        ("Heron", DROP_OVER_LIMIT),
    ]


def test_trim_is_deterministic_regardless_of_census_order() -> None:
    """Same library, same file: ties break alphabetically rather than by insertion order."""
    forward = trim(_census(Bird=5, Heron=5, Osprey=5))
    backward = trim(_census(Osprey=5, Heron=5, Bird=5))
    assert forward.kept == backward.kept == ["Bird", "Heron", "Osprey"]


def test_render_vocabulary_writes_paths_for_terms_with_a_hierarchy() -> None:
    """A known hierarchy is rendered as a path the vocabulary parser reads straight back."""
    census = KeywordCensus()
    census.add(["Animal", "Bird", "Osprey"], weight=4)
    census.add(["Landscape"], weight=4)
    rendered = render_vocabulary(trim(census))
    assert rendered == "Animal\nAnimal|Bird\nAnimal|Bird|Osprey\nLandscape\n"

    reloaded = Vocabulary.from_entries(parse_keyword_lines(rendered))
    assert reloaded.chain_for("Osprey") == ["Animal", "Bird", "Osprey"]


def test_render_vocabulary_does_not_readmit_a_dropped_parent() -> None:
    """A parent the rules rejected must not sneak back in through its child's path."""
    census = KeywordCensus()
    census.add(["2024 Trip", "Bird"], weight=9)
    rendered = render_vocabulary(trim(census))
    assert rendered == "Bird\n"


def test_render_vocabulary_prefixes_a_header_when_given_one() -> None:
    """The generated file explains itself; comment lines are ignored on read."""
    rendered = render_vocabulary(trim(_census(Bird=5)), header="# built by photo-tagger")
    assert rendered == "# built by photo-tagger\nBird\n"
    assert Vocabulary.from_entries(parse_keyword_lines(rendered)).terms == ("Bird",)


def test_render_drop_report_lists_every_drop_with_its_reason() -> None:
    """The report is what makes --min-uses tunable instead of a guess."""
    report = render_drop_report(trim(_census(Bird=10, Fluke=1)))
    lines = report.splitlines()
    assert lines[0] == "keyword,uses,reason,detail"
    assert lines[1] == "Fluke,1,rare,used 1x"


def test_census_merge_adds_up_two_sources() -> None:
    """An export and the photos themselves are two counts of one library, so they add up."""
    photos = KeywordCensus()
    photos.add(["Animal", "Bird"], weight=3)
    photos.photos = 12
    export = KeywordCensus()
    export.add(["Animal", "Bird"], weight=2)
    export.add(["Sunset"], weight=4)

    export.merge(photos)

    assert dict(export.uses) == {"Animal": 5, "Bird": 5, "Sunset": 4}
    assert export.best_chain("Bird") == ("Animal", "Bird")
    assert export.photos == photos.photos


def test_vocabulary_header_reports_the_source_and_the_rules() -> None:
    """The generated file says where it came from, so its thresholds can be revisited."""
    header = vocabulary_header("410 photo(s)", 120, 8, TrimRules(min_uses=3, max_terms=500))

    assert header.startswith("# photo-tagger vocabulary: 120 keywords kept, 8 dropped.\n")
    assert "# Source: 410 photo(s).\n" in header
    assert "used at least 3x, at most 500 terms, digits dropped" in header
    assert header.endswith("{braces} for a synonym. A line starting with '# ' is a comment.\n")


def test_vocabulary_header_names_an_uncapped_list() -> None:
    """No cap reads as words, not as a stray None."""
    header = vocabulary_header("an export", 3, 0, TrimRules(max_terms=None, allow_digits=True))
    assert "at most no cap terms, digits kept" in header


def test_vocabulary_header_records_what_the_model_organized() -> None:
    """The organize pass invents parent keywords, so the file lists them."""
    stats = OrganizeStats(model_name="qwen-vl", categories=["Animal", "Place"], grouped=4)
    header = vocabulary_header("410 photo(s)", 120, 8, TrimRules(), stats)

    assert "# Organized by qwen-vl: 4 keyword(s) folded into a synonym" in header
    assert "# Categories (written to your photos as parents): Animal, Place." in header
