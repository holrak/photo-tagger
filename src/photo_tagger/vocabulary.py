"""
Controlled vocabulary: the fixed term list generated keywords are snapped onto.

A photographer who already curates a Lightroom keyword hierarchy does not want an AI seeding it with
near-duplicates ("Osprey" next to "Sea Hawk", "Bird of Prey" next to "Raptor"). A vocabulary names
the terms that are allowed: every generated keyword is matched against it and either rewritten to
the catalog's own spelling or, in strict mode, dropped.

Two things build a :class:`Vocabulary`: :func:`load_vocabulary` reads the user's keyword file, and
:mod:`photo_tagger.sessions` derives one from a shoot's own output so photos of the same subject
agree with each other. Both then run through the same matcher.

Matching is deliberately layered, cheapest first: exact (case-insensitive), then a loose key that
ignores punctuation and a trailing plural, then a bounded fuzzy pass for typos and small variants.
"""

import csv
import difflib
import io
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self

from loguru import logger

from photo_tagger.errors import PhotoTaggerError
from photo_tagger.keywords import parse_hierarchical_keyword


if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class VocabularyError(PhotoTaggerError):
    """The vocabulary file is missing, unreadable, or contains no usable terms."""


# Past this size the file is the wrong file rather than a big catalog. It is generous on purpose:
# a Lightroom catalog that has been tagged by a tool rather than by hand runs to six figures of
# keywords, and refusing one of those would be refusing the very catalog a vocabulary is for.
# Whatever the limit, checking it early beats reading a gigabyte of something else into memory.
_MAX_FILE_CHARS = 5_000_000

# Similarity a fuzzy candidate must reach to count as the same term, and the rule that it must
# start with the same letter. Both exist to absorb typos and spacing variants ("Ospray",
# "Wild Life") without folding genuinely different concepts together. The ratio is length-aware by
# construction: one wrong letter in "Osprey" scores 0.83 and passes, while the same edit in "Duck"
# scores 0.75 and does not, which is the behavior we want on short words. The first-letter rule
# covers what the ratio alone misses, where an extra leading letter makes another real word
# ("Eagle" and "Beagle" score 0.91).
_FUZZY_CUTOFF = 0.82

# difflib compares against every candidate, so a huge vocabulary would pay O(terms) per keyword.
# Past this size only the exact and loose lookups run, which are dict hits.
_FUZZY_MAX_TERMS = 5000

# How many entries the prompt block lists before it truncates. The block is sent with every photo,
# so it is capped to keep the token cost bounded on a large catalog.
_PROMPT_MAX_ENTRIES = 120

# A comment needs a space after the hash. Lightroom catalogs are full of hashtag-shaped keywords
# ("#Diversity", "#1"), and dropping those as comments loses real terms; nobody writes a comment
# without the space.
_COMMENT_RE = re.compile(r"^#(\s|$)")

_SYNONYM_RE = re.compile(r"\{([^{}]*)\}")
_NON_ALPHANUMERIC_RE = re.compile(r"[^\w\s]+", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")


def _loose_key(term: str) -> str:
    """
    Return a comparison key that ignores case, punctuation, spacing, and a trailing plural.

    Examples:
        >>> _loose_key("Bird-of-Prey")
        'bird of prey'
        >>> _loose_key("Ospreys")
        'osprey'
    """
    cleaned = _NON_ALPHANUMERIC_RE.sub(" ", term.casefold())
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return _singularize(cleaned)


def _singularize(text: str) -> str:
    """
    Strip a naive English plural ending from the last word of *text*.

    Only ever used to build comparison keys, never to produce output, so an over-eager strip on a
    non-English term costs nothing unless the vocabulary happens to contain the stripped form too.

    Examples:
        >>> _singularize("berries")
        'berry'
        >>> _singularize("churches")
        'church'
    """
    if text.endswith("ies") and len(text) > 4:  # noqa: PLR2004 - "ies" plus a stem character
        return text[:-3] + "y"
    if text.endswith("es") and text[:-2].endswith(("ch", "sh", "ss", "x", "z")):
        return text[:-2]
    if text.endswith("s") and not text.endswith(("ss", "us", "is")):
        return text[:-1]
    return text


def _clean_segment(segment: str) -> str:
    """
    Strip whitespace and Lightroom's "do not export" brackets from one chain segment.

    Examples:
        >>> _clean_segment("  [Private]  ")
        'Private'
    """
    stripped = segment.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        stripped = stripped[1:-1].strip()
    return stripped


@dataclass(slots=True)
class _Entry:
    """One parsed vocabulary line: its root-to-leaf chain and any synonyms of the leaf."""

    chain: list[str]
    synonyms: list[str] = field(default_factory=list)


def _split_synonyms(line: str) -> tuple[str, list[str]]:
    """
    Pull ``{synonym}`` groups off *line*, returning the bare name and the synonyms.

    Examples:
        >>> _split_synonyms("Osprey {Sea Hawk} {Fish Hawk}")
        ('Osprey', ['Sea Hawk', 'Fish Hawk'])
    """
    synonyms = [syn.strip() for syn in _SYNONYM_RE.findall(line) if syn.strip()]
    return _SYNONYM_RE.sub("", line).strip(), synonyms


def _explicit_chain(name: str) -> list[str] | None:
    """
    Read a one-line hierarchy path, or None when *name* is a plain term.

    Accepts Lightroom's ``Animal|Bird|Osprey`` and the model's leaf-first ``Osprey<Bird<Animal``.

    Examples:
        >>> _explicit_chain("Animal|Bird|Osprey")
        ['Animal', 'Bird', 'Osprey']
        >>> _explicit_chain("Osprey<Bird<Animal")
        ['Animal', 'Bird', 'Osprey']
        >>> _explicit_chain("Osprey") is None
        True
    """
    if "<" in name or ">" in name:
        _, parts = parse_hierarchical_keyword(name)
        return [seg for seg in (_clean_segment(p) for p in parts) if seg] or None
    if "|" in name:
        return [seg for seg in (_clean_segment(p) for p in name.split("|")) if seg] or None
    return None


def _indent_width(line: str) -> int:
    """Return the number of leading whitespace characters in *line*."""
    return len(line) - len(line.lstrip())


def _flag_prefix_len(row: list[str]) -> int:
    """
    Count the leading single-letter option columns of a CSV row.

    The last field is never counted, so a row whose keyword is itself one letter still has a
    keyword.
    """
    length = 0
    while length < len(row) - 1 and len(row[length].strip()) <= 1:
        length += 1
    return length


def _lightroom_csv_keywords(text: str) -> str | None:
    """
    Lift the keyword column out of a Lightroom CSV keyword export, or return None.

    Lightroom's *Metadata > Export Keywords* offers two shapes of the same list. The ``.txt`` is
    the indented list :func:`_parse_lines` reads as-is; the ``.csv`` wraps it in four option columns
    ("Include On Export" and friends) and hides the indentation inside the last field. Without this,
    the CSV parses into terms like ``Y,Y,Y,N,Osprey`` that match nothing, and the whole hierarchy is
    lost: a vocabulary that silently snaps no keyword at all, which strict mode turns into a run
    that drops every one.

    Detection reads the shape rather than the header text, which Lightroom translates: every data
    row must open with the same run of single-letter flags. Column *count* is no help, because
    Lightroom does not quote a keyword that contains a comma ("Gdansk, Poland" arrives as two
    fields), so everything from the flags to the end of the row is joined back together.
    """
    if "," not in text.lstrip().partition("\n")[0]:
        return None
    try:
        rows = [row for row in csv.reader(io.StringIO(text)) if any(f.strip() for f in row)]
    except csv.Error:
        return None
    if len(rows) < 2:  # noqa: PLR2004 - a header and one keyword is the smallest real export
        return None

    prefixes = [_flag_prefix_len(row) for row in rows]
    # The header names its columns, so it has no flag prefix. Drop it only when it looks like one.
    if prefixes[0] == 0 and all(length > 0 for length in prefixes[1:]):
        rows, prefixes = rows[1:], prefixes[1:]
    if (columns := min(prefixes)) == 0:
        return None

    # A quoted field can hold a line break, which would otherwise split one keyword into two.
    keywords = [",".join(row[columns:]).replace("\n", " ") for row in rows]
    return "\n".join(keywords) if any(keyword.strip() for keyword in keywords) else None


def _parse_lines(text: str) -> list[_Entry]:
    """
    Parse a vocabulary file into entries.

    Handles both shapes a photographer is likely to have: Lightroom's indented keyword export (one
    keyword per line, children indented under their parent, ``{braces}`` for synonyms and
    ``[brackets]`` for keywords marked "do not export") and a plain list where each line is a term
    or a full path. Blank lines are ignored, and so is a ``#`` comment: a hash *followed by a
    space*, which leaves hashtag-style keywords like ``#Diversity`` as the keywords they are.

    Indentation is compared by raw width rather than counted in levels, so tabs and any consistent
    number of spaces both describe the same tree.
    """
    entries: list[_Entry] = []
    stack: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        stripped = raw_line.lstrip()
        if not stripped or _COMMENT_RE.match(stripped):
            continue
        indent = _indent_width(raw_line)
        name, synonyms = _split_synonyms(raw_line)
        if path := _explicit_chain(name):
            # A full path on one line stands alone; it never parents the lines below it.
            entries.append(_Entry(path, synonyms))
            continue
        term = _clean_segment(name)
        if not term:
            # A line holding nothing but synonyms belongs to the keyword above it.
            if synonyms and entries:
                entries[-1].synonyms.extend(synonyms)
            continue
        while stack and stack[-1][0] >= indent:
            stack.pop()
        chain = [parent for _, parent in stack] + [term]
        stack.append((indent, term))
        entries.append(_Entry(chain, synonyms))
    return entries


@dataclass(slots=True, frozen=True)
class SnapResult:
    """
    What :meth:`Vocabulary.snap` made of one photo's keywords.

    ``keywords`` is the list to feed the merge step, in the same flat-or-``Leaf<Parent`` language
    the model emits. ``mapped`` records the rewrites (original to canonical) and ``dropped`` the
    terms strict mode discarded, so the run can report both instead of silently changing the output.
    """

    keywords: list[str] = field(default_factory=list)
    mapped: dict[str, str] = field(default_factory=dict)
    dropped: list[str] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class Vocabulary:
    """
    An ordered set of allowed terms, their hierarchies, and their lookup aliases.

    Build it with :meth:`from_entries` (or :func:`load_vocabulary`); the fields are derived indexes
    and are not meant to be assembled by hand.
    """

    terms: tuple[str, ...] = ()
    # Canonical (casefolded) leaf to its full root-to-leaf chain, longest chain per leaf.
    chains: dict[str, list[str]] = field(default_factory=dict)
    # Exact casefolded term or synonym to the canonical term.
    exact: dict[str, str] = field(default_factory=dict)
    # Punctuation- and plural-insensitive key to the canonical term.
    loose: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_entries(cls, entries: Iterable[str | _Entry]) -> Self:
        """
        Build a vocabulary from paths, plain terms, or already-parsed entries.

        Strings may be ``Animal|Bird|Osprey``, ``Osprey<Bird<Animal``, or a bare ``Osprey``. Every
        segment of a path becomes a term of its own, because Lightroom stores each level as a
        keyword. The first spelling seen for a term wins, so callers control canonical casing by
        ordering their input.
        """
        terms: list[str] = []
        chains: dict[str, list[str]] = {}
        exact: dict[str, str] = {}
        loose: dict[str, str] = {}

        def register(term: str) -> str:
            key = term.casefold()
            if (canonical := exact.get(key)) is not None:
                return canonical
            exact[key] = term
            loose.setdefault(_loose_key(term), term)
            terms.append(term)
            return term

        for entry in entries:
            parsed = entry if isinstance(entry, _Entry) else _entry_from_string(entry)
            if parsed is None:
                continue
            canonical_chain = [register(segment) for segment in parsed.chain]
            leaf_key = canonical_chain[-1].casefold()
            current = chains.get(leaf_key)
            if current is None or len(canonical_chain) > len(current):
                chains[leaf_key] = canonical_chain
            for synonym in parsed.synonyms:
                # Synonyms only ever alias an existing term; they are not terms themselves, so a
                # generated "Sea Hawk" comes back as the catalog's "Osprey".
                exact.setdefault(synonym.casefold(), canonical_chain[-1])
                loose.setdefault(_loose_key(synonym), canonical_chain[-1])

        return cls(terms=tuple(terms), chains=chains, exact=exact, loose=loose)

    def __bool__(self) -> bool:
        """Report whether the vocabulary holds any term."""
        return bool(self.terms)

    def match(self, term: str) -> str | None:
        """
        Return the canonical spelling of *term*, or None when the vocabulary has no such concept.

        Examples:
            >>> vocab = Vocabulary.from_entries(["Animal|Bird|Osprey"])
            >>> vocab.match("ospreys")
            'Osprey'
            >>> vocab.match("Tractor") is None
            True
        """
        stripped = term.strip()
        if not stripped:
            return None
        if (hit := self.exact.get(stripped.casefold())) is not None:
            return hit
        key = _loose_key(stripped)
        if (hit := self.loose.get(key)) is not None:
            return hit
        if not key or len(self.terms) > _FUZZY_MAX_TERMS:
            return None
        candidates = [other for other in self.loose if other[:1] == key[:1]]
        close = difflib.get_close_matches(key, candidates, n=1, cutoff=_FUZZY_CUTOFF)
        return self.loose[close[0]] if close else None

    def chain_for(self, term: str) -> list[str] | None:
        """Return the canonical root-to-leaf chain whose leaf is *term*, if it has one."""
        chain = self.chains.get(term.casefold())
        return list(chain) if chain and len(chain) > 1 else None

    def _snap_one(self, keyword: str) -> str | None:
        """
        Map one generated keyword (flat or a ``Leaf<Parent`` chain) onto the vocabulary.

        Only the leaf is matched: the model's own parents are discarded in favour of the catalog's
        hierarchy, which is the point of having a vocabulary at all.
        """
        _, parts = parse_hierarchical_keyword(keyword)
        if not parts:
            return None
        canonical = self.match(parts[-1])
        if canonical is None:
            return None
        if (chain := self.chain_for(canonical)) is not None:
            # Back to the leaf-first form the merge step parses.
            return "<".join(reversed(chain))
        return canonical

    def snap(self, keywords: Iterable[str], *, strict: bool = False) -> SnapResult:
        """
        Rewrite *keywords* onto the vocabulary, dropping the unmatched ones when *strict*.

        Matched keywords come back carrying the vocabulary's hierarchy, so the merge step writes the
        catalog's chain rather than whatever parents the model invented. Unmatched keywords pass
        through untouched unless *strict* is set.
        """
        out: list[str] = []
        seen: set[str] = set()
        mapped: dict[str, str] = {}
        dropped: list[str] = []
        for keyword in keywords:
            original = keyword.strip()
            if not original:
                continue
            snapped = self._snap_one(original)
            if snapped is None:
                if strict:
                    dropped.append(original)
                    continue
                snapped = original
            elif snapped != original:
                mapped[original] = snapped
            if (key := snapped.casefold()) not in seen:
                seen.add(key)
                out.append(snapped)
        return SnapResult(keywords=out, mapped=mapped, dropped=dropped)

    def _display_paths(self) -> list[str]:
        """
        Render one ``Parent > Child`` line per term, dropping lines a longer one already shows.

        Without the prefix pass an indented catalog lists "Animal", then "Animal > Bird", then
        "Animal > Bird > Osprey": three lines to say what the last one says on its own.
        """
        paths = [self.chains.get(term.casefold(), [term]) for term in self.terms]
        covered = {" > ".join(chain[:depth]) for chain in paths for depth in range(1, len(chain))}
        return [line for chain in paths if (line := " > ".join(chain)) not in covered]

    def prompt_section(self, *, max_entries: int = _PROMPT_MAX_ENTRIES) -> str:
        """
        Render the vocabulary as a prompt block, or "" when it is empty.

        Hierarchies are shown parent-first because that is how a photographer reads their own
        keyword tree. The list is capped: the block ships with every photo, so a large catalog is
        truncated rather than allowed to dominate the context.
        """
        if not self.terms:
            return ""
        paths = self._display_paths()
        lines = paths[:max_entries]
        remaining = len(paths) - len(lines)
        if remaining > 0:
            lines.append(f"... and {remaining} more")
        listing = "\n".join(f"- {line}" for line in lines)
        return (
            "Controlled Vocabulary. These are the only keyword names this catalog uses. Whenever "
            "a term below fits what you see, write it exactly as spelled here instead of a "
            "synonym or a variant of it. Hierarchies are shown parent > child; output them the "
            "usual way, leaf-first with '<'.\n" + listing
        )


def _entry_from_string(raw: str) -> _Entry | None:
    """Parse one ``A|B|C``, ``C<B<A``, or plain-term string into an entry."""
    name, synonyms = _split_synonyms(raw)
    if chain := _explicit_chain(name):
        return _Entry(chain, synonyms)
    term = _clean_segment(name)
    return _Entry([term], synonyms) if term else None


def load_vocabulary(path: Path) -> Vocabulary:
    """
    Read a vocabulary file and index it.

    Accepts either Lightroom keyword-list export, ``.txt`` or ``.csv`` (see
    :func:`_lightroom_csv_keywords`), or a plain list of terms and paths (see :func:`_parse_lines`).
    Raises :class:`VocabularyError` when the file cannot be read or holds no usable term, because
    silently continuing with an empty vocabulary would drop every keyword in strict mode.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.error("vocabulary_read_failed", file=str(path), error=str(exc))
        msg = f"Could not read vocabulary file {path}: {exc}"
        raise VocabularyError(msg) from exc
    except UnicodeDecodeError as exc:
        logger.error("vocabulary_decode_failed", file=str(path), error=str(exc))
        msg = f"Vocabulary file {path} is not valid UTF-8 text"
        raise VocabularyError(msg) from exc

    if len(text) > _MAX_FILE_CHARS:
        logger.error("vocabulary_file_too_large", file=str(path), chars=len(text))
        msg = f"Vocabulary file {path} is larger than {_MAX_FILE_CHARS} characters"
        raise VocabularyError(msg)

    if (keyword_column := _lightroom_csv_keywords(text)) is not None:
        logger.debug("vocabulary_csv_export_detected", file=str(path))
        text = keyword_column

    vocabulary = Vocabulary.from_entries(_parse_lines(text))
    if not vocabulary:
        logger.error("vocabulary_empty", file=str(path))
        msg = f"Vocabulary file {path} contains no keywords"
        raise VocabularyError(msg)

    logger.info(
        "vocabulary_loaded",
        file=str(path),
        terms=len(vocabulary.terms),
        hierarchies=len(vocabulary.chains),
    )
    if len(vocabulary.terms) > _FUZZY_MAX_TERMS:
        # Both limits are silent by design, which is fine at a few hundred terms and misleading at
        # fifty thousand: the vocabulary looks like it is doing more work than it is.
        logger.warning(
            "vocabulary_very_large",
            file=str(path),
            terms=len(vocabulary.terms),
            fuzzy_matching_disabled_above=_FUZZY_MAX_TERMS,
            terms_listed_in_prompt=_PROMPT_MAX_ENTRIES,
        )
    return vocabulary
