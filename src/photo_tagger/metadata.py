"""Read and write XMP/IPTC metadata via pyexiftool."""

import contextlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from exiftool import ExifToolHelper  # type: ignore[attr-defined]
from exiftool.exceptions import ExifToolExecuteError
from loguru import logger


if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

from photo_tagger.config import (
    CAMERA_TAGS,
    LOCATION_TAGS,
    TAG_EXIF_DATE_TIME_ORIGINAL,
    TAG_EXIF_IMAGE_DESCRIPTION,
    TAG_IPTC_KEYWORDS,
    TAG_IPTC_OBJECT_NAME,
    TAG_XMP_DESCRIPTION,
    TAG_XMP_HIERARCHICAL_SUBJECT,
    TAG_XMP_SUBJECT,
    TAG_XMP_TITLE,
    TAG_XMP_WEIGHTED_FLAT_SUBJECT,
    exiftool_executable,
)
from photo_tagger.models import KeywordSet


# Exception types that pyexiftool raises on bad inputs or subprocess failures.
# Centralized here to avoid seven copies of the same three-element tuple.
_EXIFTOOL_ERRORS = (ValueError, TypeError, ExifToolExecuteError)

# Tag and api option that make exiftool hash the image data only, skipping metadata. The result
# comes back under the "File:" group, hence the separate key constant.
_IMAGE_DATA_HASH_TAG = "ImageDataHash"
_IMAGE_DATA_HASH_KEY = "File:ImageDataHash"
_IMAGE_HASH_API = ["-api", "ImageHashType=SHA256"]


@contextlib.contextmanager
def managed_helper(et: ExifToolHelper | None) -> Generator[ExifToolHelper]:
    """
    Yield *et* if supplied, otherwise spin up and tear down a one-shot helper.

    Honors an explicit ExifTool binary (``PHOTO_TAGGER_EXIFTOOL`` / config ``exiftool_path``); with
    none configured, pyexiftool finds ``exiftool`` on PATH as before.
    """
    if et is not None:
        yield et
        return
    executable = exiftool_executable()
    helper = ExifToolHelper(executable=executable) if executable else ExifToolHelper()
    with helper as own:  # type: ignore[no-untyped-call]
        yield own


_GPS_TAG = "Composite:GPSPosition"

# Existing title/description tags, in read priority order (first non-empty wins). The GUI
# surfaces these so a user can see and edit what is already on the photo before saving.
_TITLE_TAGS: tuple[str, ...] = (TAG_XMP_TITLE, TAG_IPTC_OBJECT_NAME)
# "XMP:ImageDescription" is the family-0 spelling of XMP-tiff:ImageDescription, the mirror
# write_metadata itself produces, so our own sidecars (and other tools' tiff namespace
# descriptions) count on re-read.
_DESCRIPTION_TAGS: tuple[str, ...] = (
    TAG_XMP_DESCRIPTION,
    "XMP:ImageDescription",
    TAG_EXIF_IMAGE_DESCRIPTION,
)

# Any of these tags being non-empty marks an image as already tagged. Covers the cases
# where another tool (Lightroom, exiftool by hand, a previous photo-tagger run) wrote
# keywords, a description, or a title to either the image or its XMP sidecar.
_TAGGED_INDICATOR_TAGS: tuple[str, ...] = (
    TAG_XMP_SUBJECT,
    TAG_XMP_HIERARCHICAL_SUBJECT,
    TAG_XMP_WEIGHTED_FLAT_SUBJECT,
    TAG_IPTC_KEYWORDS,
    *_DESCRIPTION_TAGS,
    *_TITLE_TAGS,
)

# Names of the three metadata fields a caller can ask about individually (vs the coarse
# "any indicator" check above). Stable identifiers the GUI's field-aware deselect uses.
FIELD_TITLE = "title"
FIELD_DESCRIPTION = "description"
FIELD_KEYWORDS = "keywords"

# The exiftool tags that mark each field as populated. A field counts as present when any of
# its tags carries content, on either the image or its XMP sidecar.
_FIELD_PRESENCE_TAGS: dict[str, tuple[str, ...]] = {
    FIELD_TITLE: _TITLE_TAGS,
    FIELD_DESCRIPTION: _DESCRIPTION_TAGS,
    FIELD_KEYWORDS: (
        TAG_XMP_SUBJECT,
        TAG_XMP_HIERARCHICAL_SUBJECT,
        TAG_XMP_WEIGHTED_FLAT_SUBJECT,
        TAG_IPTC_KEYWORDS,
    ),
}

# Tags read from a photo when collecting "existing" keywords, paired with the
# KeywordSet field they feed. Applied in order, so XMP entries arrive before the
# IPTC fall-back.
_KEYWORD_TAG_TO_FIELD: tuple[tuple[str, str], ...] = (
    (TAG_XMP_SUBJECT, "subject"),
    (TAG_XMP_HIERARCHICAL_SUBJECT, "hierarchical"),
    (TAG_XMP_WEIGHTED_FLAT_SUBJECT, "weighted"),
    (TAG_IPTC_KEYWORDS, "subject"),
)


def metadata_targets(image_path: Path) -> list[str]:
    """
    Return the file paths that may contain metadata for an image.

    Examples:
        >>> metadata_targets(Path("/photos/image.cr3"))  # doctest: +SKIP
        ['/photos/image.cr3', '/photos/image.xmp']
    """
    targets: list[str] = []
    xmp_path = image_path.with_suffix(".xmp")
    if image_path.exists():
        targets.append(str(image_path))
    if xmp_path.exists():
        targets.append(str(xmp_path))
    return targets


def format_metadata_value(value: Any) -> str:  # noqa: ANN401
    """
    Coerce metadata values (lists, numbers) into a readable string.

    Examples:
        >>> format_metadata_value(["sky", "", None])
        'sky'
        >>> format_metadata_value(42)
        '42'
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v) for v in value if str(v).strip())
    return str(value)


def _coerce_to_list(raw_value: Any) -> list[str]:  # noqa: ANN401
    """Return a clean list of strings from a possibly-scalar exiftool field."""
    if isinstance(raw_value, (list, tuple, set)):
        return [str(item) for item in raw_value if str(item).strip()]
    return [str(raw_value)] if str(raw_value).strip() else []


def _dedup_preserving_first_case(values: list[str]) -> list[str]:
    """
    Return *values* with case-insensitive duplicates collapsed, first-seen casing kept.

    Photos often carry the same keyword in both XMP-dc:Subject and IPTC:Keywords with different
    casing (``Bird`` vs ``bird``). Exact-match dedup leaves both, and the writer faithfully replays
    the duplicate back to disk. Compare on casefold instead so the output keyword list reflects what
    Lightroom would consider distinct.
    """
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _accumulate_keyword_blocks(
    blocks: list[dict[str, Any]],
    result: KeywordSet,
) -> int:
    """Fill *result* from exiftool *blocks*; returns the count of IPTC entries seen."""
    iptc_count = 0
    for block in blocks:
        for tag, field_name in _KEYWORD_TAG_TO_FIELD:
            if tag not in block:
                continue
            values = _coerce_to_list(block[tag])
            getattr(result, field_name).extend(values)
            if tag == TAG_IPTC_KEYWORDS:
                iptc_count += len(values)
    return iptc_count


def _dedup_keyword_set(keywords: KeywordSet) -> None:
    """Collapse case-insensitive duplicates in every view of *keywords* in place."""
    keywords.subject = _dedup_preserving_first_case(keywords.subject)
    keywords.hierarchical = _dedup_preserving_first_case(keywords.hierarchical)
    keywords.weighted = _dedup_preserving_first_case(keywords.weighted)


def _value_is_present(value: Any) -> bool:  # noqa: ANN401
    """Return True if an exiftool tag value carries any non-blank content."""
    if value is None:
        return False
    if isinstance(value, (list, tuple, set)):
        return any(str(item).strip() for item in value)
    return bool(str(value).strip())


def _block_has_indicator(blocks: list[dict[str, Any]]) -> bool:
    """Return True if any indicator tag in the exiftool result blocks is populated."""
    for block in blocks:
        for tag in _TAGGED_INDICATOR_TAGS:
            if _value_is_present(block.get(tag)):
                return True
    return False


def _build_target_index(paths: list[Path]) -> tuple[list[str], dict[Path, list[Path]]]:
    """
    Flatten every image's metadata targets and map each target back to its image(s).

    The map is keyed by ``Path``, not the string sent to exiftool: exiftool echoes ``SourceFile``
    with forward slashes, so string keys would miss every entry on Windows, where ``str(Path)`` uses
    backslashes. Path equality normalizes the separator.

    A target maps to a *list* because RAW+JPEG pairs sharing a stem (``IMG_0001.CR3`` and
    ``IMG_0001.JPG``) both resolve to the same ``IMG_0001.xmp`` sidecar; a single ``Path`` value
    here would silently drop all but the last image sharing that sidecar.
    """
    all_targets: list[str] = []
    target_to_images: dict[Path, list[Path]] = {}
    for image_path in paths:
        for target in metadata_targets(image_path):
            all_targets.append(target)
            target_to_images.setdefault(Path(target), []).append(image_path)
    return all_targets, target_to_images


def _images_for_block(
    block: dict[str, Any],
    target_to_images: dict[Path, list[Path]],
) -> list[Path]:
    """Map one exiftool result block back to its source image(s) via the target index."""
    source_file = block.get("SourceFile")
    if not source_file:
        return []
    return target_to_images.get(Path(str(source_file)), [])


def _batched_get_tags(
    et: ExifToolHelper | None,
    *,
    files: list[str],
    tags: list[str],
) -> list[dict[str, Any]]:
    """
    Run one ``get_tags`` over *files*, salvaging the payload when a single file is unreadable.

    exiftool exits non-zero when any file in the batch has a format error, and pyexiftool then
    raises even though the JSON for every file (including the healthy ones) is already on stdout.
    Without the salvage, one corrupt or zero-byte photo would wipe out the whole folder's read and
    every image would report "no metadata". Re-raises when there is nothing to salvage.
    """
    with managed_helper(et) as helper:
        try:
            blocks: list[dict[str, Any]] = helper.get_tags(files=files, tags=tags)
        except ExifToolExecuteError as exc:
            try:
                blocks = json.loads(exc.stdout or "")
            except ValueError:
                raise exc from None
            if not isinstance(blocks, list):
                raise
            logger.warning(
                "exiftool_batch_partial_failure",
                files=len(files),
                error=str(exc.stderr).strip(),
            )
        return blocks


def find_tagged_images(
    image_paths: Iterable[Path],
    *,
    et: ExifToolHelper | None = None,
) -> set[Path]:
    """
    Return the subset of *image_paths* that already carry meaningful metadata.

    An image counts as tagged if either it or its XMP sidecar has any of the indicator tags
    populated (keywords, hierarchical keywords, description, or title). The check batches all target
    files into a single exiftool call and maps the result blocks back to source images, reducing the
    IPC cost from O(N) round-trips to O(1).
    """
    paths = list(image_paths)
    if not paths:
        return set()

    all_targets, target_to_images = _build_target_index(paths)
    if not all_targets:
        return set()

    tagged: set[Path] = set()
    try:
        blocks = _batched_get_tags(et, files=all_targets, tags=list(_TAGGED_INDICATOR_TAGS))
    except _EXIFTOOL_ERRORS as exc:
        logger.exception("failed_to_open_exiftool_for_tagged_check", error=str(exc))
        return set()

    for block in blocks:
        if _block_has_indicator([block]):
            tagged.update(_images_for_block(block, target_to_images))

    if tagged:
        logger.debug("tagged_images_detected", count=len(tagged))
    return tagged


def find_field_presence(
    image_paths: Iterable[Path],
    *,
    et: ExifToolHelper | None = None,
) -> dict[Path, set[str]]:
    """
    Report which of title/description/keywords each image already carries.

    Returns a mapping from every input path to the set of present field names (:data:`FIELD_TITLE`,
    :data:`FIELD_DESCRIPTION`, :data:`FIELD_KEYWORDS`); an empty set means the photo was scanned and
    carries none. When exiftool cannot be run at all the result is an *empty dict*, so callers (the
    GUI's Tagged column) can tell "could not read" from "read and found nothing" instead of wrongly
    reporting every photo as untagged. A field counts as present when any of its tags is populated
    on the image *or* its XMP sidecar, so presence accumulates across both. Like
    :func:`find_tagged_images`, this is one batched exiftool call rather than one per image.

    This is the per-field counterpart the GUI uses to deselect, say, photos that already have a
    title and description while leaving keyword-only photos selected.
    """
    paths = list(image_paths)
    presence: dict[Path, set[str]] = {path: set() for path in paths}
    if not paths:
        return presence

    all_targets, target_to_images = _build_target_index(paths)
    if not all_targets:
        return presence

    all_tags = sorted({tag for tags in _FIELD_PRESENCE_TAGS.values() for tag in tags})
    try:
        blocks = _batched_get_tags(et, files=all_targets, tags=all_tags)
    except _EXIFTOOL_ERRORS as exc:
        logger.exception("failed_to_open_exiftool_for_field_presence", error=str(exc))
        return {}

    for block in blocks:
        images = _images_for_block(block, target_to_images)
        if not images:
            continue
        present_fields = {
            field_name
            for field_name, tags in _FIELD_PRESENCE_TAGS.items()
            if any(_value_is_present(block.get(tag)) for tag in tags)
        }
        for image_path in images:
            presence[image_path].update(present_fields)
    return presence


def read_keyword_sets(
    image_paths: Iterable[Path],
    *,
    et: ExifToolHelper | None = None,
) -> dict[Path, KeywordSet]:
    """
    Read the keywords already on each of *image_paths*, image and XMP sidecar merged.

    Every input path maps to a :class:`KeywordSet`, empty when the photo carries no keyword; an
    *empty dict* means exiftool could not be run at all, which callers must tell apart from "read
    and found nothing" (see :func:`find_field_presence`). Like the other census-style reads, this is
    one batched exiftool call rather than one per image, so a whole library can be surveyed without
    paying an IPC round-trip per photo.
    """
    paths = list(image_paths)
    keywords = {path: KeywordSet() for path in paths}
    if not paths:
        return keywords

    all_targets, target_to_images = _build_target_index(paths)
    if not all_targets:
        return keywords

    tags = list(dict.fromkeys(tag for tag, _ in _KEYWORD_TAG_TO_FIELD))
    try:
        blocks = _batched_get_tags(et, files=all_targets, tags=tags)
    except _EXIFTOOL_ERRORS as exc:
        logger.exception("failed_to_open_exiftool_for_keyword_census", error=str(exc))
        return {}

    for block in blocks:
        for image_path in _images_for_block(block, target_to_images):
            _accumulate_keyword_blocks([block], keywords[image_path])
    for keyword_set in keywords.values():
        _dedup_keyword_set(keyword_set)
    return keywords


def _first_tag_value(blocks: list[dict[str, Any]], tags: tuple[str, ...]) -> str | None:
    """Return the first non-blank value across *blocks* for the first matching *tags* entry."""
    for tag in tags:
        for block in blocks:
            value = block.get(tag)
            if value not in (None, ""):
                return format_metadata_value(value)
    return None


def read_caption(
    image_path: Path,
    *,
    et: ExifToolHelper | None = None,
) -> tuple[str | None, str | None]:
    """
    Read the existing title and description from the image or its XMP sidecar.

    Returns ``(title, description)``, each ``None`` when absent. Title prefers XMP-dc:Title and
    falls back to IPTC:ObjectName; description prefers XMP-dc:Description and falls back to
    EXIF:ImageDescription. The GUI uses this to show a user what is already on a photo before it
    writes new metadata.
    """
    targets = metadata_targets(image_path)
    if not targets:
        return (None, None)

    try:
        with managed_helper(et) as helper:
            blocks = helper.get_tags(files=targets, tags=[*_TITLE_TAGS, *_DESCRIPTION_TAGS])
    except _EXIFTOOL_ERRORS as e:
        logger.exception("failed_to_read_caption", error=str(e))
        return (None, None)

    return (_first_tag_value(blocks, _TITLE_TAGS), _first_tag_value(blocks, _DESCRIPTION_TAGS))


# Human-readable labels for where existing metadata lives.
SOURCE_IMAGE = "image file"
SOURCE_SIDECAR = "XMP sidecar"


def read_metadata_sources(
    image_path: Path,
    *,
    et: ExifToolHelper | None = None,
) -> list[str]:
    """
    Report where existing metadata lives: the image file, an XMP sidecar, or both.

    Returns a list of source labels (:data:`SOURCE_IMAGE`, :data:`SOURCE_SIDECAR`) for the targets
    that actually carry an indicator tag (keywords, title, or description), so the GUI can tell the
    user whether what it shows came from the photo or its sidecar. Returns an empty list when
    neither target has any such metadata.
    """
    targets = metadata_targets(image_path)
    if not targets:
        return []

    try:
        with managed_helper(et) as helper:
            blocks = helper.get_tags(files=targets, tags=list(_TAGGED_INDICATOR_TAGS))
    except _EXIFTOOL_ERRORS as exc:
        logger.exception("failed_to_read_metadata_sources", error=str(exc))
        return []

    sources: list[str] = []
    for block in blocks:
        if not _block_has_indicator([block]):
            continue
        source_file = str(block.get("SourceFile", ""))
        label = SOURCE_SIDECAR if source_file.casefold().endswith(".xmp") else SOURCE_IMAGE
        if label not in sources:
            sources.append(label)
    return sources


@dataclass(slots=True, frozen=True)
class ImageContext:
    """
    Bundle of metadata read off a photo before the AI call.

    Replaces the older sequence of separate single-purpose reads the pipeline used to issue, which
    cost three exiftool IPC round-trips per image. The batched read fetches everything we need in
    one call.
    """

    existing_keywords: KeywordSet = field(default_factory=KeywordSet)
    location_tags: dict[str, str] = field(default_factory=dict)
    gps_position: str | None = None
    camera_info: dict[str, str] = field(default_factory=dict)
    # SHA256 of the image data only (exiftool's ImageDataHash), ignoring metadata. None when not
    # requested or when exiftool cannot hash the format. Used as a metadata-independent cache key.
    content_hash: str | None = None


def _extract_gps(blocks: list[dict[str, Any]]) -> str | None:
    """Return the first non-empty GPS position string from *blocks*, else None."""
    for block in blocks:
        value = block.get(_GPS_TAG)
        if value not in (None, ""):
            return format_metadata_value(value)
    return None


def _extract_named_tags(
    blocks: list[dict[str, Any]],
    tags: tuple[str, ...],
) -> dict[str, str]:
    """Collect non-blank string values for *tags* across all *blocks*."""
    collected: dict[str, str] = {}
    for block in blocks:
        for tag in tags:
            value = block.get(tag)
            if value not in (None, "") and tag not in collected:
                collected[tag] = format_metadata_value(value)
    return collected


def _extract_content_hash(blocks: list[dict[str, Any]]) -> str | None:
    """
    Return the image-data SHA256 from whichever block carries it, else None.

    Only the image file yields ImageDataHash; XMP sidecars have no image data, so a first-present
    scan across the blocks safely picks the image's hash.
    """
    for block in blocks:
        value = block.get(_IMAGE_DATA_HASH_KEY)
        if value:
            return str(value)
    return None


def read_image_context(
    image_path: Path,
    *,
    et: ExifToolHelper | None = None,
    include_content_hash: bool = False,
) -> ImageContext:
    """
    Fetch every read-only tag the pipeline needs in a single exiftool call.

    This is the production read path, used by both the CLI's ``process_photo`` and the GUI worker.

    When *include_content_hash* is set, the same call also reads ImageDataHash so callers can key a
    cache on the image content rather than the whole file. It is off by default because hashing the
    image data is wasted work when no cache is configured.
    """
    targets = metadata_targets(image_path)
    if not targets:
        logger.info("no_metadata_targets_found")
        return ImageContext()

    keyword_tags = [tag for tag, _ in _KEYWORD_TAG_TO_FIELD]
    all_tags = list(dict.fromkeys([*keyword_tags, *LOCATION_TAGS, *CAMERA_TAGS, _GPS_TAG]))
    params: list[str] = []
    if include_content_hash:
        all_tags.append(_IMAGE_DATA_HASH_TAG)
        params = _IMAGE_HASH_API

    try:
        with managed_helper(et) as helper:
            blocks = helper.get_tags(files=targets, tags=all_tags, params=params)
    except _EXIFTOOL_ERRORS as exc:
        logger.exception("failed_to_read_image_context", error=str(exc))
        return ImageContext()

    existing_keywords = KeywordSet()
    _accumulate_keyword_blocks(blocks, existing_keywords)
    _dedup_keyword_set(existing_keywords)

    location_tags = _extract_named_tags(blocks, LOCATION_TAGS)
    camera_info = _extract_named_tags(blocks, CAMERA_TAGS)
    gps_position = _extract_gps(blocks)

    logger.debug(
        "image_context_read",
        subject_count=len(existing_keywords.subject),
        hierarchical_count=len(existing_keywords.hierarchical),
        location_tag_count=len(location_tags),
        camera_tag_count=len(camera_info),
        gps_present=gps_position is not None,
    )
    return ImageContext(
        existing_keywords=existing_keywords,
        location_tags=location_tags,
        gps_position=gps_position,
        camera_info=camera_info,
        content_hash=_extract_content_hash(blocks) if include_content_hash else None,
    )


def _parse_exif_datetime(raw: str) -> datetime | None:
    """
    Parse EXIF's ``YYYY:MM:DD HH:MM:SS`` into a naive datetime, or None when it is unusable.

    Sub-second and offset suffixes are cut off rather than parsed: grouping photos into shoots only
    needs whole seconds, and a camera's clock is its own local time either way.
    """
    try:
        return datetime.strptime(raw.strip()[:19], "%Y:%m:%d %H:%M:%S")  # noqa: DTZ007 - camera local time
    except ValueError:
        return None


def read_capture_times(
    image_paths: Iterable[Path],
    *,
    et: ExifToolHelper | None = None,
) -> dict[Path, datetime]:
    """
    Read ``EXIF:DateTimeOriginal`` for every path in a single exiftool call.

    Returns naive datetimes in the camera's own local time, which is what a capture timestamp means
    and what makes gaps between frames comparable. Paths whose tag is missing, blank, or malformed
    are simply absent from the result, leaving the caller free to fall back to the file mtime.
    """
    paths = [path for path in image_paths if path.is_file()]
    if not paths:
        return {}

    try:
        blocks = _batched_get_tags(
            et,
            files=[str(p) for p in paths],
            tags=[TAG_EXIF_DATE_TIME_ORIGINAL],
        )
    except _EXIFTOOL_ERRORS as exc:
        logger.exception("failed_to_read_capture_times", error=str(exc))
        return {}

    by_source = {path: path for path in paths}
    times: dict[Path, datetime] = {}
    for block in blocks:
        source = block.get("SourceFile")
        raw = block.get(TAG_EXIF_DATE_TIME_ORIGINAL)
        if not source or not raw:
            continue
        image_path = by_source.get(Path(str(source)))
        parsed = _parse_exif_datetime(str(raw))
        if image_path is not None and parsed is not None:
            times[image_path] = parsed
    logger.debug("capture_times_read", requested=len(paths), found=len(times))
    return times


def _first_present(values: dict[str, str], *tags: str) -> str | None:
    """Return the first non-empty value in *values* lookup-ordered by *tags*."""
    for tag in tags:
        if (raw := values.get(tag)) and (stripped := raw.strip()):
            return stripped
    return None


def select_camera_fields(
    camera_info: dict[str, str],
) -> tuple[str | None, str | None, str | None]:
    """Return ``(model, lens, capture_date)`` from a :data:`CAMERA_TAGS` mapping."""
    return (
        camera_info.get("EXIF:Model"),
        camera_info.get("EXIF:LensModel"),
        camera_info.get(TAG_EXIF_DATE_TIME_ORIGINAL),
    )


def select_location(location_tags: dict[str, str]) -> tuple[str | None, str | None]:
    """
    Return ``(city, country)`` from a :data:`LOCATION_TAGS` mapping.

    XMP-photoshop and IPTC each define their own city/country fields, so this picks the first
    present in each category, the same fallback order :func:`_format_location` uses for the prompt
    hint and the CSV report uses for its columns.
    """
    city = _first_present(location_tags, "XMP-photoshop:City", "IPTC:City")
    country = _first_present(
        location_tags,
        "XMP-photoshop:Country",
        "IPTC:Country-PrimaryLocationName",
    )
    return city, country


def _format_location(location_tags: dict[str, str]) -> str | None:
    """
    Build a "City, Country" hint, falling back to whichever single field is set.

    Returning both means the model gets the full place name (e.g. "Barcelona, Spain") instead of
    silently losing the city under the older "first hit wins" logic.
    """
    city, country = select_location(location_tags)
    if city and country:
        return f"{city}, {country}"
    return city or country


def _camera_lines(camera_info: dict[str, str]) -> list[str]:
    """Render ``EXIF:Model`` / ``EXIF:LensModel`` / ``EXIF:DateTimeOriginal`` lines."""
    lines: list[str] = []
    if model := camera_info.get("EXIF:Model"):
        lines.append(f"- Camera: {model}")
    if lens := camera_info.get("EXIF:LensModel"):
        lines.append(f"- Lens: {lens}")
    if captured := camera_info.get(TAG_EXIF_DATE_TIME_ORIGINAL):
        lines.append(f"- Captured: {captured}")
    return lines


def prompt_with_hint(base_prompt: str, hint: str | None) -> str:
    """
    Append a photographer's note to *base_prompt*; return it unchanged when *hint* is blank.

    The note is framed as authoritative, unlike the Existing Metadata block (which the system prompt
    tells the model to distrust on conflict): a hint exists precisely because the model misread the
    image ("that animal is a deer, not a boar"), so on conflict the note must win.
    """
    hint = (hint or "").strip()
    if not hint:
        return base_prompt
    return (
        f"{base_prompt.strip()}\n\n"
        f"Photographer's note about this photo: {hint}\n"
        "The photographer knows what is in the photo. If this note conflicts with your own "
        "identification, trust the note."
    )


def build_contextual_prompt(  # noqa: PLR0913 - each kwarg renders an independent section.
    base_prompt: str,
    flat_keywords: list[str],
    location_tags: dict[str, str],
    gps_info: dict[str, str],
    max_prompt_flat_keywords: int = 4,
    *,
    camera_info: dict[str, str] | None = None,
) -> str:
    """
    Create a concise user prompt that surfaces existing photo metadata.

    *camera_info* is a mapping of ExifTool tag names (``EXIF:Model`` etc.) to their string values.
    When present, equipment and capture-date hints are appended so the model can take cues from the
    gear used (macro lens, telephoto, wide angle) and the time of year.
    """
    metadata_lines: list[str] = []

    if unique_keywords := list(dict.fromkeys(flat_keywords)):
        selected = unique_keywords[:max_prompt_flat_keywords]
        line = "- Existing Keywords: " + ", ".join(selected)
        if len(unique_keywords) > max_prompt_flat_keywords:
            line += ", ..."
        metadata_lines.append(line)

    if location_value := _format_location(location_tags):
        metadata_lines.append(f"- Location: {location_value}")

    if gps_position := (gps_info or {}).get("position"):
        metadata_lines.append(f"- GPS: {gps_position}")

    metadata_lines.extend(_camera_lines(camera_info or {}))

    sections = [base_prompt.strip()]
    if metadata_lines:
        sections.append("Existing Metadata:\n" + "\n".join(metadata_lines))
    return "\n\n".join(sections)


def _build_write_payload(
    keywords: KeywordSet,
    description: str | None,
    title: str | None,
) -> dict[str, str | list[str]]:
    """Build the exiftool tag map that write_metadata will apply."""
    payload: dict[str, str | list[str]] = {}
    if subjects := keywords.subject:
        payload["XMP-dc:Subject"] = subjects
        # Lightroom prioritizes IPTC:Keywords for JPEGs, so mirror the Subject list there.
        payload[TAG_IPTC_KEYWORDS] = subjects
    if hierarchical := keywords.hierarchical:
        payload["XMP-lr:HierarchicalSubject"] = hierarchical
    if weighted := keywords.weighted:
        payload[TAG_XMP_WEIGHTED_FLAT_SUBJECT] = weighted
    if description:
        payload["XMP-dc:Description"] = description
        # ImageDescription lives in the XMP *tiff* namespace (exiftool rejects XMP-exif for it,
        # silently: a warning plus exit 0). exiftool maps it back to IFD0 ImageDescription when a
        # sidecar is folded into the image.
        payload["XMP-tiff:ImageDescription"] = description
    if title:
        payload["XMP-dc:Title"] = title
        payload[TAG_IPTC_OBJECT_NAME] = title
    return payload


def write_target(image_path: Path, *, use_sidecar: bool) -> Path:
    """
    Return the file :func:`write_metadata` will modify for *image_path*.

    Shared so callers that need to inspect the target before the write (the undo journal, which
    must know whether the sidecar already existed) cannot drift from the writer's own choice.

    Examples:
        >>> write_target(Path("/photos/image.cr3"), use_sidecar=True).name
        'image.xmp'
        >>> write_target(Path("/photos/image.cr3"), use_sidecar=False).name
        'image.cr3'
    """
    return image_path.with_suffix(".xmp") if use_sidecar else image_path


def write_metadata(  # noqa: PLR0913 - distinct optional fields are clearer as kwargs.
    image_path: Path,
    keywords: KeywordSet,
    *,
    description: str | None = None,
    title: str | None = None,
    backup: bool = True,
    use_sidecar: bool = True,
    et: ExifToolHelper | None = None,
) -> bool:
    """
    Write tags to either the image file or an XMP sidecar in Lightroom-compatible format.

    Args:
        image_path: Path to the image file. The sidecar shares its name with a `.xmp` extension.
        keywords: A :class:`KeywordSet` of subject, hierarchical, and weighted keywords.
        description: Optional short description to write to XMP (and ImageDescription).
        title: Optional short title to write to XMP-dc:Title and IPTC:ObjectName.
        backup: If True, let ExifTool create a backup (`_original` suffix where applicable).
        use_sidecar: If True, write to a sidecar; otherwise embed in the source file.
        et: Optional already-open ExifToolHelper to reuse across a batch.

    Returns:
        True on success, False on failure.
    """
    target_path = write_target(image_path, use_sidecar=use_sidecar)
    payload = _build_write_payload(keywords, description, title)
    if not payload:
        logger.warning("no_data_to_write", file=image_path.name)
        return False

    set_kwargs: dict[str, Any] = {"tags": payload, "files": [str(target_path)]}
    if not backup:
        set_kwargs["params"] = ["-overwrite_original"]

    try:
        with managed_helper(et) as helper:
            helper.set_tags(**set_kwargs)
    except _EXIFTOOL_ERRORS as e:
        logger.exception("xmp_write_failed", error=str(e), target=str(target_path))
        return False

    logger.info(
        "metadata_written_successfully",
        target=str(target_path),
        mode="sidecar" if use_sidecar else "embedded",
        subject_keywords=len(keywords.subject),
        hierarchical_keywords=len(keywords.hierarchical),
        backup_created=backup,
    )
    return True
