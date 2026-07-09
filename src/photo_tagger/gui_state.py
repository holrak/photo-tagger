"""
Qt-free helpers and per-photo state for the desktop GUI.

Kept separate from :mod:`photo_tagger.gui` so the GUI's logic (expanding dropped paths, grouping
files for the tree, parsing the editable keyword field, building the keyword set to write) is plain
Python the test suite covers normally, with no display server and no dependency on the optional
PySide6 extra. ``gui.py`` is then just the widget and event-loop shell that wires these helpers to
Qt.
"""

import os
import subprocess  # nosec B404 - only used to read the user's own login-shell PATH (see below)
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import tomlkit

from photo_tagger.config import DEFAULT_OUTPUT_LANGUAGE
from photo_tagger.csv_report import ReportRow
from photo_tagger.discovery import parse_extensions, resolve_image_files
from photo_tagger.i18n import AUTO, _, ngettext
from photo_tagger.keywords import dedupe_keywords, merge_keywords
from photo_tagger.metadata import (
    FIELD_DESCRIPTION,
    FIELD_KEYWORDS,
    FIELD_TITLE,
    select_camera_fields,
    select_location,
)
from photo_tagger.models import KeywordSet


if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


# Per-photo status values, shown as an icon/word in the tree.
PENDING = "pending"  # added, not yet generated
WORKING = "working"  # generation in flight
READY = "ready"  # a proposal is available to review
SAVED = "saved"  # written to the file
FAILED = "failed"  # generation or save failed

# A broad, common default for the GUI's folder-scan extensions. Each distinct extension is listed
# because matching is case-insensitive but not variant-aware (jpg does not cover jpeg).
DEFAULT_GUI_EXTENSIONS = "jpg,jpeg,png,dng,cr3,nef,arw,heic,heif,tif,tiff,webp"

# Pre-filled choices for the metadata-language combo. English names on purpose: the value is
# spliced into the (English) system prompt as-is, and the combo stays editable, so any language
# the model understands can still be typed.
OUTPUT_LANGUAGE_SUGGESTIONS = (
    DEFAULT_OUTPUT_LANGUAGE,
    "Brazilian Portuguese",
    "Dutch",
    "French",
    "German",
    "Italian",
    "Japanese",
    "Korean",
    "Portuguese",
    "Russian",
    "Simplified Chinese",
    "Spanish",
)

# Substrings that hint a model is vision-capable, used to surface likely picks first.
_VISION_HINTS = (
    "vl",
    "vision",
    "llava",
    "moondream",
    "minicpm-v",
    "bakllava",
    "cogvlm",
    "internvl",
    "pixtral",
    "gemma3",
    "smolvlm",
    "-v-",
)


@dataclass(slots=True)
class PhotoItem:
    """
    Mutable per-photo state shared between the file tree and the detail pane.

    ``existing_*`` holds what was read off the file; ``title``/``description``/ ``keywords`` are the
    editable working copy that :func:`keywords_to_save` and the Save action write. The working copy
    is seeded from a :class:`Proposal` (or from the existing values when the user opens a file
    without generating).

    The trailing read-context fields (``camera_info``/``location_tags``/``gps_position``) and the
    token/timing counters are captured when a proposal is generated and feed the CSV export; they
    stay empty for photos that were never generated.
    """

    path: Path
    selected: bool = True
    status: str = PENDING
    error: str = ""
    loaded: bool = False
    # Which indicator fields (title/description/keywords) the file already carries. None means
    # the background metadata scan has not reported yet; a set (possibly empty) means it has.
    known_fields: set[str] | None = None
    existing_title: str | None = None
    existing_description: str | None = None
    existing_keywords: KeywordSet = field(default_factory=KeywordSet)
    existing_sources: list[str] = field(default_factory=list)
    sources_read: bool = False
    has_proposal: bool = False
    from_cache: bool = False
    title: str = ""
    description: str = ""
    keywords: list[str] = field(default_factory=list)
    camera_info: dict[str, str] = field(default_factory=dict)
    location_tags: dict[str, str] = field(default_factory=dict)
    gps_position: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class Proposal:
    """One file's AI proposal plus the existing metadata and read context alongside it."""

    path: Path
    existing_title: str | None
    existing_description: str | None
    existing_keywords: KeywordSet
    title: str
    description: str
    keywords: list[str]
    camera_info: dict[str, str] = field(default_factory=dict)
    location_tags: dict[str, str] = field(default_factory=dict)
    gps_position: str | None = None
    from_cache: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    seconds: float = 0.0


def expand_inputs(
    paths: Iterable[Path],
    image_extensions: str,
    *,
    recursive: bool,
) -> list[Path]:
    """
    Expand dropped files and folders into a de-duplicated list of image files.

    Reuses the CLI's discovery: directories are walked and extension-filtered, explicit files are
    kept as is, and order is preserved. An empty extension string yields no files rather than
    raising.
    """
    ext_set = parse_extensions(image_extensions)
    if not ext_set:
        return []
    return resolve_image_files(list(paths), ext_set, recursive=recursive)


def new_paths(existing: Iterable[Path], found: Iterable[Path]) -> list[Path]:
    """Return entries of *found* not already present in *existing* (order preserved)."""
    seen = set(existing)
    out: list[Path] = []
    for path in found:
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out


def group_by_parent(paths: Iterable[Path]) -> list[tuple[Path, list[Path]]]:
    """Group file paths under their parent directory, first-seen order preserved."""
    groups: dict[Path, list[Path]] = {}
    for path in paths:
        groups.setdefault(path.parent, []).append(path)
    return list(groups.items())


def parse_keyword_lines(text: str) -> list[str]:
    """
    Parse the editable keyword field (one keyword per line) into a clean list.

    Repeats are collapsed case-insensitively so the change counts stay honest; the model (and a
    pasting user) sometimes repeats a keyword.
    """
    return dedupe_keywords(text.splitlines())


def keywords_to_text(keywords: list[str]) -> str:
    """Render keywords one per line for the editable text field."""
    return "\n".join(keywords)


def keywords_to_save(
    existing: KeywordSet,
    edited_keywords: list[str],
    *,
    overwrite: bool,
) -> KeywordSet:
    """
    Build the :class:`KeywordSet` to write from the edited keywords.

    Merges with the existing keywords unless *overwrite* is set, in which case the existing keywords
    are dropped first. Hierarchical entries (``Duck<Bird<Animal``) are parsed by
    :func:`merge_keywords` exactly as the CLI does.
    """
    base = KeywordSet() if overwrite else existing
    return merge_keywords(base, edited_keywords)


def apply_proposal(item: PhotoItem, proposal: Proposal) -> None:
    """Fill *item*'s existing metadata and seed its editable copy from *proposal*."""
    item.existing_title = proposal.existing_title
    item.existing_description = proposal.existing_description
    item.existing_keywords = proposal.existing_keywords
    item.loaded = True
    item.title = proposal.title
    item.description = proposal.description
    item.keywords = list(proposal.keywords)
    item.camera_info = proposal.camera_info
    item.location_tags = proposal.location_tags
    item.gps_position = proposal.gps_position
    item.input_tokens = proposal.input_tokens
    item.output_tokens = proposal.output_tokens
    item.total_tokens = proposal.total_tokens
    item.seconds = proposal.seconds
    item.has_proposal = True
    item.from_cache = proposal.from_cache
    item.status = READY
    item.error = ""


def photo_item_to_report_row(item: PhotoItem, *, overwrite: bool) -> ReportRow:
    """
    Flatten a GUI :class:`PhotoItem` into a CSV :class:`ReportRow`.

    The keyword columns reflect what a Save would write: the working keywords merged with (or, when
    *overwrite*, replacing) the existing ones, exactly as :func:`keywords_to_save` computes for the
    Save action. The EXIF and token columns are whatever was captured at generation time, and stay
    blank for a photo that was added but never generated.
    """
    to_write = keywords_to_save(item.existing_keywords, item.keywords, overwrite=overwrite)
    model, lens, captured = select_camera_fields(item.camera_info)
    city, country = select_location(item.location_tags)
    return ReportRow(
        file=str(item.path),
        filename=item.path.name,
        status=item.status,
        error=item.error,
        title=item.title,
        description=item.description,
        keywords=list(to_write.subject),
        hierarchical_keywords=list(to_write.hierarchical),
        existing_keywords=list(item.existing_keywords.subject),
        existing_title=item.existing_title or "",
        existing_description=item.existing_description or "",
        camera_model=model or "",
        lens_model=lens or "",
        capture_date=captured or "",
        gps_position=item.gps_position or "",
        city=city or "",
        country=country or "",
        input_tokens=item.input_tokens,
        output_tokens=item.output_tokens,
        total_tokens=item.total_tokens,
        seconds=item.seconds,
        from_cache=item.from_cache if item.has_proposal else None,
    )


def paths_matching_fields(
    presence: dict[Path, set[str]],
    required: set[str],
    *,
    match_all: bool,
) -> set[Path]:
    """
    Pick the paths whose present fields satisfy *required*.

    With *match_all* (AND), a path matches only when it has every required field, so ``{title,
    description}`` selects photos that already carry both. Without it (OR), having any one required
    field is enough, which is how the broad "any metadata" criterion works. An empty *required* set
    matches nothing, so a caller cannot accidentally select every photo by passing no fields.
    """
    if not required:
        return set()
    matched: set[Path] = set()
    for path, present in presence.items():
        overlap = present & required
        if (overlap == required) if match_all else bool(overlap):
            matched.add(path)
    return matched


def deselect_paths(items: dict[str, PhotoItem], paths: Iterable[Path]) -> int:
    """
    Uncheck the items whose path is in *paths*; return how many actually changed.

    This is how the GUI "skips" photos: rather than dropping them from the list like the CLI does,
    it just deselects them, so the user still sees what was skipped and can re-check any of it.
    Items already unchecked (or not in the list) are left alone and not counted, so the returned
    tally is the number of newly-skipped photos.
    """
    changed = 0
    for path in paths:
        item = items.get(str(path))
        if item is not None and item.selected:
            item.selected = False
            changed += 1
    return changed


@dataclass(slots=True)
class FolderNode:
    """A folder in the file tree, with its display label, subfolders, and files."""

    path: Path
    label: str
    folders: list[FolderNode]
    files: list[Path]


def _raw_dirs(paths: list[Path]) -> dict[Path, FolderNode]:
    """Build a directory -> node map linking every file's ancestor chain."""
    nodes: dict[Path, FolderNode] = {}
    suborder: dict[Path, list[Path]] = {}

    def ensure(directory: Path) -> None:
        if directory in nodes:
            return
        nodes[directory] = FolderNode(path=directory, label=directory.name, folders=[], files=[])
        suborder[directory] = []
        parent = directory.parent
        if parent != directory:  # stop at the filesystem root, whose parent is itself
            ensure(parent)
            suborder[parent].append(directory)

    for file in paths:
        ensure(file.parent)
        nodes[file.parent].files.append(file)
    for parent, subs in suborder.items():
        nodes[parent].folders = [nodes[s] for s in subs]
    return nodes


def _first_significant(node: FolderNode) -> FolderNode:
    """Descend through single-child, file-less folders to the first meaningful node."""
    while not node.files and len(node.folders) == 1:
        node = node.folders[0]
    return node


def _display_node(node: FolderNode, parent_path: Path | None) -> FolderNode:
    """Re-label *node* relative to its display parent and collapse its child chains."""
    label = str(node.path) if parent_path is None else str(node.path.relative_to(parent_path))
    folders = [_display_node(_first_significant(child), node.path) for child in node.folders]
    return FolderNode(path=node.path, label=label, folders=folders, files=list(node.files))


def build_tree(paths: Iterable[Path]) -> list[FolderNode]:
    """
    Group file paths into a nested folder tree for the GUI.

    Leading single-child directory chains are collapsed (``a/b/c`` shows as one node when only ``c``
    holds files), subfolders nest under their parent, and disjoint roots become separate top-level
    nodes. Each node's ``label`` is its path relative to its display parent (the absolute path for a
    top-level node).
    """
    paths = list(paths)
    if not paths:
        return []
    nodes = _raw_dirs(paths)
    tops: list[FolderNode] = []
    for root in (d for d in nodes if d.parent == d):
        node = _first_significant(nodes[root])
        if node.path.parent == node.path and not node.files and len(node.folders) > 1:
            # The filesystem root is just a container for disjoint trees; promote each branch.
            tops.extend(_first_significant(child) for child in node.folders)
        else:
            tops.append(node)
    return [_display_node(top, None) for top in tops]


def paths_under(paths: Iterable[Path], folder: Path) -> list[Path]:
    """Return the paths that live under *folder* (at any depth), order preserved."""
    return [path for path in paths if path.is_relative_to(folder)]


def rank_vision_models(model_ids: Iterable[str]) -> list[str]:
    """
    Order model ids with likely vision-capable ones first, keeping all of them.

    The provider listings do not reliably flag modality, so this is a name heuristic (``vl``,
    ``vision``, ``llava``, ...) used only to surface probable picks; nothing is hidden, so a model
    the heuristic misses is still selectable.
    """
    model_ids = list(model_ids)
    likely = [m for m in model_ids if any(hint in m.lower() for hint in _VISION_HINTS)]
    likely_set = set(likely)
    others = [m for m in model_ids if m not in likely_set]
    return likely + others


# A nested name -> children mapping used to fold cumulative hierarchy paths into one tree.
type _Tree = dict[str, "_Tree"]


def chain_to_display(path: str) -> str:
    """
    Convert a Lightroom root-first ``A|B|C`` path to the GUI's leaf-first ``C<B<A`` notation.

    The editable keyword field already speaks the '<' form (it is what the model emits and what the
    CLI documents), so every read-only view uses it too instead of leaking the on-disk '|'.
    """
    return "<".join(reversed(path.split("|")))


def format_existing_keywords(keywords: KeywordSet) -> str:
    """
    Render existing keywords one per line, in the same ``<`` notation as the editable field.

    Each hierarchy shows once, as its deepest chain (``Duck<Bird<Animal``); the intermediate flat
    copies Lightroom also stores (Animal, Bird) are folded into it. Flat keywords that belong to no
    hierarchy follow. This mirrors what a user would type to reproduce the same metadata.
    """
    chains = [entry for entry in keywords.hierarchical if "|" in entry]
    deepest = [
        entry
        for entry in chains
        if not any(other.startswith(entry + "|") for other in chains if other != entry)
    ]
    covered = {segment.casefold() for entry in chains for segment in entry.split("|")}
    lines = [chain_to_display(entry) for entry in deepest]
    lines += [kw for kw in keywords.subject if kw.casefold() not in covered]
    return "\n".join(lines)


def _walk_tree(node: _Tree, prefix: str, lines: list[str]) -> None:
    """Append *node*'s children to *lines* with ``tree``-style guide characters."""
    entries = list(node.items())
    for index, (name, child) in enumerate(entries):
        last = index == len(entries) - 1
        lines.append(prefix + ("└─ " if last else "├─ ") + name)
        _walk_tree(child, prefix + ("   " if last else "│  "), lines)


def hierarchy_tree_text(paths: Iterable[str]) -> str:
    """
    Render Lightroom ``A|B|C`` paths as a ``tree``-style diagram with branch guides.

    Cumulative paths ("A|B", "A|B|C") collapse into one branch, so the view shows the taxonomy shape
    rather than repeating every prefix line. Roots sit at column zero and children hang off
    ``├─``/``└─`` connectors like the CLI ``tree`` command, which reads far better than plain two-
    space indentation.
    """
    root: _Tree = {}
    for path in paths:
        node = root
        for segment in path.split("|"):
            node = node.setdefault(segment, {})

    lines: list[str] = []
    for name, child in root.items():
        lines.append(name)
        _walk_tree(child, "", lines)
    return "\n".join(lines)


def hierarchy_preview(
    existing: KeywordSet,
    edited_keywords: list[str],
    *,
    overwrite: bool,
) -> str:
    """Render the keyword tree that saving the edited keywords would produce."""
    return hierarchy_tree_text(
        keywords_to_save(existing, edited_keywords, overwrite=overwrite).hierarchical,
    )


# Diff states for a keyword when comparing the existing flat subjects to what a save writes.
ADDED = "added"
REMOVED = "removed"
UNCHANGED = "unchanged"


def keyword_diff(
    existing: KeywordSet,
    edited_keywords: list[str],
    *,
    overwrite: bool,
) -> list[tuple[str, str]]:
    """
    Compare existing flat keywords to the result of saving the edited keywords.

    Returns ``(keyword, state)`` pairs where state is :data:`ADDED`, :data:`REMOVED`, or
    :data:`UNCHANGED`. The keywords that will be written come first in write order, then any that
    would be dropped (only possible with *overwrite*). Comparison is case-insensitive.
    """
    result = keywords_to_save(existing, edited_keywords, overwrite=overwrite).subject
    existing_folds = {kw.casefold() for kw in existing.subject}
    result_folds = {kw.casefold() for kw in result}
    diff = [(kw, UNCHANGED if kw.casefold() in existing_folds else ADDED) for kw in result]
    diff += [(kw, REMOVED) for kw in existing.subject if kw.casefold() not in result_folds]
    return diff


# Order photos take when the tree is sorted by Status (ascending): lifecycle, failures last.
STATUS_SORT_ORDER = (PENDING, WORKING, READY, SAVED, FAILED)


def status_sort_rank(status: str) -> int:
    """
    Rank a photo status for sorting the tree's Status column.

    Lower ranks sort first when ascending, following the lifecycle order in
    :data:`STATUS_SORT_ORDER` (pending, working, ready, saved, failed). Sorting on the raw status
    label would order it alphabetically ("failed" before "ready"), which is not useful; this gives
    the column a meaningful order that the user can flip by clicking the header.
    """
    try:
        return STATUS_SORT_ORDER.index(status)
    except ValueError:
        return len(STATUS_SORT_ORDER)


def status_summary(items: Iterable[PhotoItem]) -> str:
    """One-line counts for the status bar: selected, generated, saved, failed."""
    items = list(items)
    selected = sum(1 for i in items if i.selected)
    generated = sum(1 for i in items if i.has_proposal)
    saved = sum(1 for i in items if i.status == SAVED)
    failed = sum(1 for i in items if i.status == FAILED)
    files = ngettext("{n} file", "{n} files", len(items)).format(n=len(items))
    return _(
        "{files} · {selected} selected · {generated} generated · {saved} saved · {failed} failed",
    ).format(
        files=files,
        selected=selected,
        generated=generated,
        saved=saved,
        failed=failed,
    )


def count_generated(items: Iterable[PhotoItem]) -> int:
    """Count photos with an AI proposal: the batch size the GUI session reports to telemetry."""
    return sum(1 for item in items if item.has_proposal)


def reveal_label(platform_name: str) -> str:
    """Name the OS file browser for the context-menu action ("Reveal in Finder" on macOS)."""
    if platform_name == "darwin":
        return _("Reveal in Finder")
    if platform_name.startswith("win"):
        return _("Show in Explorer")
    return _("Show in File Manager")


def reveal_command(path: Path, platform_name: str) -> list[str] | None:
    """
    Return the argv that reveals *path* selected in the OS file browser, or None.

    None means the platform has no standard "reveal" command (Linux file managers vary), so the
    caller should fall back to opening the containing folder instead.
    """
    if platform_name == "darwin":
        return ["open", "-R", str(path)]
    if platform_name.startswith("win"):
        # Explorer's /select switch takes the path in the same argument, comma-separated.
        return ["explorer", f"/select,{path}"]
    return None


def file_type_label(path: Path) -> str:
    """
    Label a photo's Type column: its extension, plus "+xmp" when an XMP sidecar sits beside it.

    The sidecar check is a plain filesystem stat, so this is cheap enough to run for every file at
    add time (no exiftool involved).
    """
    suffix = path.suffix.lstrip(".").lower()
    return f"{suffix}+xmp" if path.with_suffix(".xmp").exists() else suffix


# Column letters for the Tagged indicator, in display order.
_FIELD_LETTERS = ((FIELD_TITLE, "T"), (FIELD_DESCRIPTION, "D"), (FIELD_KEYWORDS, "K"))


def tagged_summary(fields: set[str]) -> str:
    """
    Compress a file's present metadata fields into the Tagged column label.

    "TDK" means title, description, and keywords all exist; "-" means the scan ran and found none.
    The header tooltip spells out the letters.
    """
    letters = "".join(letter for field_name, letter in _FIELD_LETTERS if field_name in fields)
    return letters or "-"


# Badge names for the folder grid's thumbnail overlays, in the order they are drawn.
BADGE_FAILED = "failed"
BADGE_SAVED = "saved"
BADGE_UNSAVED = "unsaved"
BADGE_METADATA = "metadata"
BADGE_SIDECAR = "sidecar"


def thumb_badges(item: PhotoItem, *, has_sidecar: bool) -> list[str]:
    """
    Decide which overlay badges a photo's grid thumbnail shows.

    At most one lifecycle badge (failed beats saved beats unsaved-proposal), plus the already-has-
    metadata and sidecar markers. The GUI maps each name to a color and glyph.
    """
    badges: list[str] = []
    if item.status == FAILED:
        badges.append(BADGE_FAILED)
    elif item.status == SAVED:
        badges.append(BADGE_SAVED)
    elif item.has_proposal:
        badges.append(BADGE_UNSAVED)
    if item.known_fields:
        badges.append(BADGE_METADATA)
    if has_sidecar:
        badges.append(BADGE_SIDECAR)
    return badges


def _toml_str(value: str) -> str:
    """Quote *value* as a TOML basic string."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_bool(value: bool) -> str:  # noqa: FBT001  # the bool is the value being rendered
    """Render a TOML boolean."""
    return "true" if value else "false"


@dataclass(frozen=True, slots=True)
class GuiConfigValues:
    """
    The GUI settings that persist to the config file.

    The API key is deliberately not a field: it must never be written to disk.
    """

    provider_name: str
    model_name: str
    api_base_url: str | None
    extensions: str
    recursive: bool
    write_title: bool
    write_description: bool
    write_keywords: bool
    preserve_keywords: bool
    use_sidecar: bool
    telemetry_enabled: bool


def config_toml_text(values: GuiConfigValues) -> str:
    """
    Render *values* as a fresh TOML config file the CLI and GUI both load.

    Key names mirror the config tables ``load_defaults`` reads ([provider], [output], [telemetry],
    plus the top-level extensions/recursive). Used only when no config file exists yet; an existing
    file goes through :func:`merged_config_text` instead so nothing is lost.
    """
    lines = [
        "# Written by the Photo Tagger GUI (Settings > Save Settings as Defaults).",
        f"extensions = {_toml_str(values.extensions)}",
        f"recursive = {_toml_bool(values.recursive)}",
        "",
        "[provider]",
        f"provider_name = {_toml_str(values.provider_name)}",
        f"model_name = {_toml_str(values.model_name)}",
    ]
    if values.api_base_url:
        lines.append(f"api_base_url = {_toml_str(values.api_base_url)}")
    lines += [
        "",
        "[output]",
        f"write_title = {_toml_bool(values.write_title)}",
        f"write_description = {_toml_bool(values.write_description)}",
        f"write_keywords = {_toml_bool(values.write_keywords)}",
        f"preserve_keywords = {_toml_bool(values.preserve_keywords)}",
        f"use_sidecar = {_toml_bool(values.use_sidecar)}",
        "",
        "[telemetry]",
        f"enabled = {_toml_bool(values.telemetry_enabled)}",
        "",
    ]
    return "\n".join(lines)


def merged_config_text(existing_text: str, values: GuiConfigValues) -> str:
    """
    Update *existing_text* (a TOML config file) with *values*, preserving everything else.

    tomlkit keeps comments, ordering, and keys the GUI does not manage, so saving from the GUI never
    destroys a hand-written config. Only the GUI-managed keys are set; a blank base URL removes the
    key so the provider default applies again.
    """
    document = tomlkit.parse(existing_text)
    document["extensions"] = values.extensions
    document["recursive"] = values.recursive

    provider = document.setdefault("provider", tomlkit.table())
    provider["provider_name"] = values.provider_name
    provider["model_name"] = values.model_name
    if values.api_base_url:
        provider["api_base_url"] = values.api_base_url
    else:
        provider.pop("api_base_url", None)

    output = document.setdefault("output", tomlkit.table())
    output["write_title"] = values.write_title
    output["write_description"] = values.write_description
    output["write_keywords"] = values.write_keywords
    output["preserve_keywords"] = values.preserve_keywords
    output["use_sidecar"] = values.use_sidecar

    telemetry = document.setdefault("telemetry", tomlkit.table())
    telemetry["enabled"] = values.telemetry_enabled
    return tomlkit.dumps(document)


def config_text_with_language(existing_text: str, language: str) -> str:
    """
    Set the top-level ``language`` key in a TOML config, preserving everything else.

    Choosing :data:`~photo_tagger.i18n.AUTO` (follow the OS locale, the built-in default) removes
    the key instead of writing it, so a config never pins a language the user did not pick. tomlkit
    keeps comments and ordering intact, like :func:`merged_config_text`.
    """
    document = tomlkit.parse(existing_text)
    if language == AUTO:
        document.pop("language", None)
    else:
        document["language"] = language
    return tomlkit.dumps(document)


def config_text_with_output_language(existing_text: str, language: str) -> str:
    """
    Set ``[inference] output_language`` in a TOML config, preserving everything else.

    The key the CLI's ``--output-language`` flag reads its default from, so the GUI choice carries
    over to CLI runs too. Choosing the built-in default (English) removes the key instead of pinning
    it, mirroring :func:`config_text_with_language`; an ``[inference]`` table left empty by that
    removal is dropped as well.
    """
    document = tomlkit.parse(existing_text)
    if language.strip().casefold() == DEFAULT_OUTPUT_LANGUAGE.casefold():
        inference = document.get("inference")
        if inference is not None:
            inference.pop("output_language", None)
            if not inference:
                document.pop("inference", None)
    else:
        document.setdefault("inference", tomlkit.table())["output_language"] = language.strip()
    return tomlkit.dumps(document)


# How long to wait for the login shell to report its PATH before giving up.
_SHELL_PATH_TIMEOUT = 5.0


def login_shell_path() -> list[str]:
    """
    Return the PATH entries a login shell would set, or ``[]`` if they can't be determined.

    A Finder/Dock launch inherits a minimal PATH, so exiftool installed by any package manager
    (Homebrew, Nix, MacPorts, ...) is missing - those locations are only added by shell startup
    files. Asking the user's own login shell for its PATH recovers exiftool however it was
    installed, instead of hard-coding each manager's bin dir. Best-effort: a missing ``$SHELL``, a
    non-zero exit, or a timeout all yield ``[]``.
    """
    shell = os.environ.get("SHELL")
    if not shell:
        return []
    try:
        result = subprocess.run(  # noqa: S603  # nosec B603 - fixed argv, trusted $SHELL
            [shell, "-l", "-c", 'printf %s "$PATH"'],
            capture_output=True,
            text=True,
            timeout=_SHELL_PATH_TIMEOUT,
            check=False,
        )
    except OSError, subprocess.SubprocessError:
        return []
    return [entry for entry in result.stdout.strip().split(os.pathsep) if entry]


def ensure_path_dirs(current_path: str, dirs: Iterable[str]) -> str:
    """
    Prepend any of *dirs* not already in *current_path* to it, order preserved.

    A GUI launched from Finder or the Dock inherits a minimal ``PATH``, so Homebrew's bin
    directories (where exiftool usually lives) are missing and exiftool discovery fails even though
    it works fine from a shell. Prepending them lets a double-clicked app find exiftool the same
    way. Directories that do not exist are harmless on ``PATH``, so no filesystem check is needed.
    """
    entries = current_path.split(os.pathsep) if current_path else []
    additions = [d for d in dirs if d not in entries]
    if not additions:
        return current_path
    return os.pathsep.join([*additions, *entries])
