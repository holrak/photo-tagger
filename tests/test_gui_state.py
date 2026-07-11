"""Tests for the Qt-free GUI helpers (no PySide6, no display required)."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from photo_tagger.gui_state import (
    ADDED,
    BADGE_FAILED,
    BADGE_METADATA,
    BADGE_SAVED,
    BADGE_SIDECAR,
    BADGE_UNSAVED,
    DEFAULT_GUI_EXTENSIONS,
    FAILED,
    FILTER_ALL,
    FILTER_FAILED,
    FILTER_GENERATED,
    FILTER_PENDING,
    FILTER_SAVED,
    FILTER_SELECTED,
    FILTER_UNTAGGED,
    OUTPUT_LANGUAGE_SUGGESTIONS,
    PENDING,
    READY,
    REMOVED,
    SAVED,
    SORT_NAME,
    SORT_STATUS,
    SORT_TAGGED,
    SORT_TYPE,
    STATUS_SORT_ORDER,
    UNCHANGED,
    WORKING,
    FolderNode,
    GuiConfigValues,
    PhotoItem,
    Proposal,
    apply_proposal,
    build_tree,
    chain_to_display,
    config_text_with_language,
    config_text_with_output_language,
    config_toml_text,
    deselect_paths,
    ensure_path_dirs,
    expand_inputs,
    fields_written,
    file_dialog_name_filters,
    file_type_label,
    filter_photos,
    format_existing_keywords,
    group_by_parent,
    hierarchy_preview,
    hierarchy_tree_text,
    keyword_diff,
    keywords_to_save,
    keywords_to_text,
    login_shell_path,
    merged_config_text,
    new_paths,
    parse_keyword_lines,
    paths_matching_fields,
    paths_under,
    photo_item_to_report_row,
    photo_matches_filter,
    photo_sort_key,
    rank_vision_models,
    reveal_command,
    reveal_label,
    sort_photos,
    status_sort_rank,
    status_summary,
    tagged_legend,
    tagged_summary,
    tagged_tooltip,
    thumb_badges,
)
from photo_tagger.i18n import activate
from photo_tagger.metadata import FIELD_KEYWORDS, FIELD_TITLE
from photo_tagger.models import KeywordSet
from photo_tagger.providers import PROVIDER_LABELS, PROVIDER_NAMES


def test_expand_inputs_walks_folders_and_keeps_files(tmp_path: Path) -> None:
    """Directories are extension-filtered and recursed; explicit files pass through."""
    (tmp_path / "a.jpg").write_text("x")
    (tmp_path / "b.cr3").write_text("x")
    (tmp_path / "skip.txt").write_text("x")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.jpg").write_text("x")

    flat = expand_inputs([tmp_path], "jpg", recursive=False)
    assert flat == [(tmp_path / "a.jpg").resolve()]

    deep = expand_inputs([tmp_path], "jpg,cr3", recursive=True)
    assert set(deep) == {
        (tmp_path / "a.jpg").resolve(),
        (tmp_path / "b.cr3").resolve(),
        (sub / "c.jpg").resolve(),
    }


def test_expand_inputs_empty_extensions_yields_nothing(tmp_path: Path) -> None:
    """A blank extension string returns no files rather than raising."""
    (tmp_path / "a.jpg").write_text("x")
    assert expand_inputs([tmp_path], "  ", recursive=False) == []


def test_new_paths_filters_already_present() -> None:
    """Only paths not already tracked are returned, in order, de-duplicated."""
    existing = [Path("/a.jpg"), Path("/b.jpg")]
    found = [Path("/b.jpg"), Path("/c.jpg"), Path("/c.jpg"), Path("/d.jpg")]
    assert new_paths(existing, found) == [Path("/c.jpg"), Path("/d.jpg")]


def test_group_by_parent_groups_and_preserves_order() -> None:
    """Files group under their parent folder in first-seen order."""
    paths = [Path("/x/a.jpg"), Path("/y/b.jpg"), Path("/x/c.jpg")]
    assert group_by_parent(paths) == [
        (Path("/x"), [Path("/x/a.jpg"), Path("/x/c.jpg")]),
        (Path("/y"), [Path("/y/b.jpg")]),
    ]


def test_parse_keyword_lines_strips_and_drops_blanks() -> None:
    """One keyword per line; whitespace trimmed and empty lines removed."""
    assert parse_keyword_lines("  Bird \n\n Sky\n   \nForest") == ["Bird", "Sky", "Forest"]


def test_keywords_to_text_round_trips() -> None:
    """Rendering then parsing returns the same list."""
    keywords = ["Bird", "Sky", "Forest"]
    assert parse_keyword_lines(keywords_to_text(keywords)) == keywords


def test_keywords_to_save_merges_with_existing() -> None:
    """Without overwrite, edited keywords merge into the existing set."""
    existing = KeywordSet(subject=["Beach"], weighted=["Beach"])
    merged = keywords_to_save(existing, ["Bird"], overwrite=False)
    assert merged.subject == ["Beach", "Bird"]


def test_keywords_to_save_overwrite_drops_existing() -> None:
    """With overwrite, the existing keywords are discarded before merging."""
    existing = KeywordSet(subject=["Beach"], weighted=["Beach"])
    merged = keywords_to_save(existing, ["Bird"], overwrite=True)
    assert merged.subject == ["Bird"]


def test_apply_proposal_seeds_the_editable_copy() -> None:
    """A proposal fills existing metadata and seeds the editable title/desc/keywords."""
    item = PhotoItem(path=Path("/a.jpg"))
    proposal = Proposal(
        path=Path("/a.jpg"),
        existing_title="Old",
        existing_description="Old caption.",
        existing_keywords=KeywordSet(subject=["Beach"]),
        title="Golden Eagle",
        description="An eagle soars.",
        keywords=["Eagle", "Sky"],
        camera_info={"EXIF:Model": "Canon EOS R5"},
        gps_position="53 N, 9 E",
        input_tokens=10,
        total_tokens=15,
        seconds=0.4,
    )
    apply_proposal(item, proposal)
    assert item.status == READY
    assert item.has_proposal is True
    assert item.loaded is True
    assert item.existing_title == "Old"
    assert item.existing_keywords.subject == ["Beach"]
    assert item.title == "Golden Eagle"
    assert item.keywords == ["Eagle", "Sky"]
    # The read context and usage carried by the proposal land on the item for the CSV export.
    assert item.camera_info == {"EXIF:Model": "Canon EOS R5"}
    assert item.gps_position == "53 N, 9 E"
    assert item.total_tokens == 15  # noqa: PLR2004 - fixture value
    # The editable copy is independent of the proposal's list.
    item.keywords.append("Extra")
    assert proposal.keywords == ["Eagle", "Sky"]


def test_photo_item_to_report_row_merges_keywords_and_reads_context() -> None:
    """The GUI row reflects a Save (merged keywords) plus existing metadata and EXIF."""
    item = PhotoItem(
        path=Path("/photos/a.jpg"),
        status=READY,
        title="Golden Eagle",
        description="An eagle soars.",
        keywords=["Eagle", "Bird<Animal"],
        existing_title="Old",
        existing_description="Old caption.",
        existing_keywords=KeywordSet(subject=["Beach"]),
        camera_info={"EXIF:Model": "Canon EOS R5", "EXIF:LensModel": "RF 100mm"},
        location_tags={"IPTC:City": "Berlin", "IPTC:Country-PrimaryLocationName": "Germany"},
        gps_position="53 N, 9 E",
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        seconds=0.4,
    )

    row = photo_item_to_report_row(item, overwrite=False).as_dict()

    assert row["filename"] == "a.jpg"
    assert row["status"] == READY
    assert row["title"] == "Golden Eagle"
    # Merge (not overwrite): the existing "Beach" survives alongside the new flat keyword.
    assert "Beach" in row["keywords"]
    assert "Eagle" in row["keywords"]
    # The "Bird<Animal" entry becomes a Lightroom hierarchy path, not a flat keyword.
    assert "Animal|Bird" in row["hierarchical_keywords"]
    assert row["existing_keywords"] == "Beach"
    assert row["existing_title"] == "Old"
    assert row["camera_model"] == "Canon EOS R5"
    assert row["lens_model"] == "RF 100mm"
    assert row["city"] == "Berlin"
    assert row["country"] == "Germany"
    assert row["total_tokens"] == "15"
    # The GUI has no cache or retry pass, so those columns stay blank.
    assert row["from_cache"] == ""
    assert row["retry"] == ""


def test_photo_item_to_report_row_overwrite_drops_existing_keywords() -> None:
    """With overwrite, the existing keywords are replaced rather than merged in."""
    item = PhotoItem(
        path=Path("/photos/a.jpg"),
        keywords=["Eagle"],
        existing_keywords=KeywordSet(subject=["Beach"]),
    )
    row = photo_item_to_report_row(item, overwrite=True).as_dict()
    assert row["keywords"] == "Eagle"
    # existing_keywords column still reports what was on the file before the (hypothetical) save.
    assert row["existing_keywords"] == "Beach"


def test_build_tree_groups_files_under_their_folder() -> None:
    """A single folder of files becomes one top node labelled with its absolute path."""
    forest = build_tree([Path("/photos/a.jpg"), Path("/photos/b.jpg")])
    assert len(forest) == 1
    assert forest[0].path == Path("/photos")
    assert forest[0].label == "/photos"
    assert forest[0].files == [Path("/photos/a.jpg"), Path("/photos/b.jpg")]
    assert forest[0].folders == []


def test_build_tree_nests_subfolders_under_parent() -> None:
    """Subfolders nest under the parent folder with a relative label."""
    forest = build_tree([Path("/p/a.jpg"), Path("/p/sub/b.jpg")])
    assert len(forest) == 1
    top = forest[0]
    assert top.label == "/p"
    assert top.files == [Path("/p/a.jpg")]
    assert [f.label for f in top.folders] == ["sub"]
    assert top.folders[0].files == [Path("/p/sub/b.jpg")]


def test_build_tree_collapses_single_child_chains() -> None:
    """A chain of single, file-less folders collapses into one labelled node."""
    forest = build_tree([Path("/p/x/y/b.jpg")])
    assert len(forest) == 1
    assert forest[0].label == "/p/x/y"
    assert forest[0].files == [Path("/p/x/y/b.jpg")]


def test_build_tree_keeps_a_branching_single_root_together() -> None:
    """A folder whose subfolders each hold files stays one top node with both children."""
    forest = build_tree([Path("/p/a/x.jpg"), Path("/p/b/y.jpg")])
    assert len(forest) == 1
    assert forest[0].label == "/p"
    assert sorted(f.label for f in forest[0].folders) == ["a", "b"]


def test_build_tree_splits_disjoint_roots() -> None:
    """Files under unrelated roots become separate top-level nodes (no '/' wrapper)."""
    forest = build_tree([Path("/r1/x.jpg"), Path("/r2/y.jpg")])
    assert sorted(node.label for node in forest) == ["/r1", "/r2"]


def test_build_tree_empty() -> None:
    """No paths yields no nodes."""
    assert build_tree([]) == []


def test_paths_under_filters_by_folder() -> None:
    """paths_under keeps only the paths beneath a folder, preserving order."""
    paths = [Path("/a/1.jpg"), Path("/b/2.jpg"), Path("/a/sub/3.jpg")]
    assert paths_under(paths, Path("/a")) == [Path("/a/1.jpg"), Path("/a/sub/3.jpg")]
    assert paths_under(paths, Path("/b")) == [Path("/b/2.jpg")]


def test_rank_vision_models_surfaces_likely_first_without_dropping_any() -> None:
    """Likely vision models sort first; every input model is still present."""
    models = ["llama3", "qwen3-vl-30b", "gpt-4o", "llava-7b"]
    ranked = rank_vision_models(models)
    assert ranked[:2] == ["qwen3-vl-30b", "llava-7b"]
    assert set(ranked) == set(models)


def test_provider_labels_cover_every_provider() -> None:
    """Every backend name has a display label and they are distinct."""
    assert set(PROVIDER_LABELS) == set(PROVIDER_NAMES)
    assert len(set(PROVIDER_LABELS.values())) == len(PROVIDER_NAMES)


def test_default_gui_extensions_includes_jpg_and_jpeg() -> None:
    """The broad default lists jpg and jpeg separately (matching is not variant-aware)."""
    exts = DEFAULT_GUI_EXTENSIONS.split(",")
    assert "jpg" in exts
    assert "jpeg" in exts


def test_format_existing_keywords_uses_leaf_first_chains() -> None:
    """Hierarchies render once, leaf-first with '<'; only truly flat keywords are listed apart."""
    kw = KeywordSet(
        subject=["Animal", "Bird", "Sky"],
        hierarchical=["Animal|Bird"],
    )
    text = format_existing_keywords(kw)
    assert "Bird<Animal" in text
    assert "Sky" in text.splitlines()
    # The flat copies of the chain segments are folded into the chain line, not repeated.
    assert "Animal" not in text.splitlines()
    assert format_existing_keywords(KeywordSet()) == ""


def test_format_existing_keywords_keeps_only_deepest_chain() -> None:
    """Cumulative Lightroom paths (A|B plus A|B|C) collapse into the single deepest chain."""
    kw = KeywordSet(
        subject=["Animal", "Bird", "Duck"],
        hierarchical=["Animal|Bird", "Animal|Bird|Duck"],
    )
    assert format_existing_keywords(kw) == "Duck<Bird<Animal"


def test_hierarchy_preview_renders_a_guided_tree() -> None:
    """The preview folds the cumulative paths a save writes into one tree with branch guides."""
    preview = hierarchy_preview(KeywordSet(), ["Duck<Bird<Animal"], overwrite=True)
    assert preview == "Animal\n└─ Bird\n   └─ Duck"


def test_hierarchy_tree_text_merges_shared_roots() -> None:
    """Two chains under one root share the root line, with tree-style connectors."""
    text = hierarchy_tree_text(["Animal|Bird", "Animal|Bird|Duck", "Animal|Cat"])
    assert text == "Animal\n├─ Bird\n│  └─ Duck\n└─ Cat"


def test_chain_to_display_reverses_to_leaf_first() -> None:
    """A root-first '|' path flips to the editable field's leaf-first '<' form."""
    assert chain_to_display("Animal|Bird|Duck") == "Duck<Bird<Animal"
    assert chain_to_display("Flat") == "Flat"


def test_reveal_command_per_platform(tmp_path: Path) -> None:
    """MacOS and Windows get a reveal argv; Linux falls back to opening the folder (None)."""
    photo = tmp_path / "a.jpg"
    assert reveal_command(photo, "darwin") == ["open", "-R", str(photo)]
    assert reveal_command(photo, "win32") == ["explorer", f"/select,{photo}"]
    assert reveal_command(photo, "linux") is None


def test_reveal_label_names_the_platform_browser() -> None:
    """The context-menu label matches each platform's file browser name."""
    assert reveal_label("darwin") == "Reveal in Finder"
    assert reveal_label("win32") == "Show in Explorer"
    assert reveal_label("linux") == "Show in File Manager"


def test_file_type_label_flags_sidecars(tmp_path: Path) -> None:
    """The Type label is the lowercased extension, with +xmp when a sidecar exists."""
    photo = tmp_path / "IMG_0001.CR3"
    photo.write_text("x", encoding="utf-8")
    assert file_type_label(photo) == "cr3"
    (tmp_path / "IMG_0001.xmp").write_text("<x/>", encoding="utf-8")
    assert file_type_label(photo) == "cr3+xmp"


def test_file_dialog_name_filters_puts_configured_types_first() -> None:
    """The default filter is the user's configured types, sorted, deduped, and lowercased."""
    filters = file_dialog_name_filters("jpg, cr3 ,JPG")
    assert filters[0] == "Your file types (*.cr3 *.jpg)"
    assert filters[1].startswith("All known image formats (*.arw ")
    assert "JPEG (*.jpg *.jpeg)" in filters
    assert "Camera Raw (*.arw *.cr2 *.cr3 *.dng *.nef *.orf *.raf *.rw2)" in filters
    assert filters[-1] == "All files (*)"


def test_file_dialog_name_filters_without_configured_types_still_offers_known_formats() -> None:
    """With no parseable extensions, the known-format hints and 'All files' remain."""
    filters = file_dialog_name_filters("  ,, ")
    assert filters[0].startswith("All known image formats (")
    assert "PNG (*.png)" in filters
    assert filters[-1] == "All files (*)"


def test_tagged_summary_letters_and_empty_marker() -> None:
    """Present fields compress to their letters in T/D/K order; none becomes a dash."""
    assert tagged_summary({FIELD_KEYWORDS, FIELD_TITLE}) == "TK"
    assert tagged_summary(set()) == "-"


def test_tagged_legend_maps_every_letter() -> None:
    """The header legend pairs each letter with its field name, in display order."""
    assert tagged_legend() == "T = title, D = description, K = keywords"


def test_tagged_tooltip_spells_out_the_present_letters() -> None:
    """The cell tooltip names only the fields on the file; an empty set says nothing."""
    assert tagged_tooltip({FIELD_KEYWORDS, FIELD_TITLE}) == (
        "Already on the file: T = title, K = keywords"
    )
    assert tagged_tooltip(set()) == "Already on the file: nothing"


def test_tagged_tooltip_is_fully_translated() -> None:
    """The pt_BR catalog covers the whole tooltip, field names included (regression test)."""
    activate("pt_BR")
    try:
        assert tagged_tooltip({FIELD_KEYWORDS, FIELD_TITLE}) == (
            "Já no arquivo: T = título, K = palavras-chave"
        )
    finally:
        activate("en")


def test_fields_written_reports_only_nonempty_values() -> None:
    """Empty values write nothing, so only the populated fields count as newly present."""
    written = fields_written("Duck", "", KeywordSet(subject=["Bird"]))
    assert written == {FIELD_TITLE, FIELD_KEYWORDS}
    assert fields_written(None, None, KeywordSet()) == set()


def _config_values(**overrides: object) -> GuiConfigValues:
    """Build GuiConfigValues with sensible test defaults, overridable per test."""
    base: dict[str, object] = {
        "provider_name": "lmstudio",
        "model_name": "qwen/qwen3-vl-30b",
        "api_base_url": "http://localhost:1234/v1",
        "extensions": "jpg,cr3",
        "recursive": True,
        "write_title": True,
        "write_description": True,
        "write_keywords": True,
        "preserve_keywords": True,
        "use_sidecar": True,
        "telemetry_enabled": True,
    }
    base.update(overrides)
    return GuiConfigValues(**base)  # type: ignore[arg-type]


def test_config_toml_text_round_trips_through_load_defaults() -> None:
    """The GUI-written TOML parses and lands on the right Defaults fields."""
    import tomllib  # noqa: PLC0415 - test-local parser.

    from photo_tagger.cli_options import load_defaults  # noqa: PLC0415

    text = config_toml_text(
        _config_values(
            model_name='qwen "vl" model',
            write_description=False,
            use_sidecar=False,
            telemetry_enabled=False,
        ),
    )
    defaults = load_defaults(tomllib.loads(text))

    assert defaults.provider.model_name == 'qwen "vl" model'  # quotes survive escaping
    assert defaults.provider.api_base_url == "http://localhost:1234/v1"
    assert defaults.extensions == "jpg,cr3"
    assert defaults.recursive is True
    assert defaults.output.write_description is False
    assert defaults.output.use_sidecar is False
    assert defaults.telemetry.enabled is False


def test_config_toml_text_omits_blank_url() -> None:
    """A blank base URL is left out so the provider default applies."""
    text = config_toml_text(_config_values(api_base_url=None))
    assert "api_base_url" not in text


def test_merged_config_text_preserves_comments_and_unknown_keys() -> None:
    """Merging updates only the GUI-managed keys; comments and other settings survive."""
    import tomllib  # noqa: PLC0415 - test-local parser.

    existing = (
        "# my hand-written config\n"
        'extensions = "cr3"\n'
        "\n"
        "[provider]\n"
        'provider_name = "ollama"\n'
        '# model_name = "qwen/qwen3-vl-30b"\n'
        "\n"
        "[filter]\n"
        "skip_tagged = true\n"
        "\n"
        "[artifacts]\n"
        'summary_file = "summary.txt"  # keep me\n'
    )
    merged = merged_config_text(existing, _config_values(model_name="llava"))

    # Comments and untouched tables survive verbatim.
    assert "# my hand-written config" in merged
    assert "# keep me" in merged
    data = tomllib.loads(merged)
    assert data["filter"]["skip_tagged"] is True
    assert data["artifacts"]["summary_file"] == "summary.txt"
    # GUI-managed keys are updated in place.
    assert data["extensions"] == "jpg,cr3"
    assert data["provider"]["provider_name"] == "lmstudio"
    assert data["provider"]["model_name"] == "llava"
    assert data["output"]["use_sidecar"] is True


def test_merged_config_text_drops_blank_url() -> None:
    """Clearing the URL removes the key so the provider default applies again."""
    existing = '[provider]\napi_base_url = "http://old:1234/v1"\n'
    merged = merged_config_text(existing, _config_values(api_base_url=None))
    assert "api_base_url" not in merged


def test_apply_proposal_carries_the_cache_flag() -> None:
    """A cached proposal marks the item so the tree can label it 'ready (cached)'."""
    item = PhotoItem(path=Path("/a.jpg"))
    proposal = Proposal(
        path=Path("/a.jpg"),
        existing_title=None,
        existing_description=None,
        existing_keywords=KeywordSet(),
        title="T",
        description="D",
        keywords=[],
        from_cache=True,
    )
    apply_proposal(item, proposal)
    assert item.from_cache is True


def test_thumb_badges_lifecycle_and_info_markers() -> None:
    """One lifecycle badge at most, plus metadata and sidecar markers when they apply."""
    failed = PhotoItem(path=Path("/a.jpg"), status=FAILED, known_fields={"title"})
    assert thumb_badges(failed, has_sidecar=True) == [BADGE_FAILED, BADGE_METADATA, BADGE_SIDECAR]

    saved = PhotoItem(path=Path("/a.jpg"), status=SAVED)
    assert thumb_badges(saved, has_sidecar=False) == [BADGE_SAVED]

    ready = PhotoItem(path=Path("/a.jpg"), status=READY, has_proposal=True)
    assert thumb_badges(ready, has_sidecar=False) == [BADGE_UNSAVED]

    pending = PhotoItem(path=Path("/a.jpg"), known_fields=set())
    assert thumb_badges(pending, has_sidecar=False) == []


def test_keyword_diff_merge_marks_added_and_unchanged() -> None:
    """Without overwrite, kept keywords are unchanged and new ones are added; none removed."""
    existing = KeywordSet(subject=["Beach", "Bird"])
    diff = keyword_diff(existing, ["Bird", "Eagle"], overwrite=False)
    assert ("Beach", UNCHANGED) in diff
    assert ("Bird", UNCHANGED) in diff
    assert ("Eagle", ADDED) in diff
    assert all(state != REMOVED for _, state in diff)


def test_keyword_diff_overwrite_marks_removed() -> None:
    """With overwrite, keywords not in the new set are marked removed."""
    existing = KeywordSet(subject=["Beach", "Bird"])
    diff = keyword_diff(existing, ["Bird", "Eagle"], overwrite=True)
    assert ("Eagle", ADDED) in diff
    assert ("Bird", UNCHANGED) in diff
    assert ("Beach", REMOVED) in diff


def test_folder_node_is_constructible() -> None:
    """FolderNode is a plain dataclass usable directly in tests and rendering."""
    node = FolderNode(path=Path("/x"), label="/x", folders=[], files=[Path("/x/a.jpg")])
    assert node.label == "/x"


def test_deselect_paths_unchecks_matches_and_counts_only_changes() -> None:
    """Only listed, still-selected items flip; the count is the newly-skipped tally."""
    a, b, c = Path("/a.jpg"), Path("/b.jpg"), Path("/c.jpg")
    items = {str(p): PhotoItem(path=p) for p in (a, b, c)}
    items[str(b)].selected = False  # already off: it must not count again

    changed = deselect_paths(items, [a, b])

    assert changed == 1
    assert items[str(a)].selected is False
    assert items[str(b)].selected is False
    assert items[str(c)].selected is True  # untouched


def test_deselect_paths_ignores_unknown_paths() -> None:
    """A path not in the list deselects nothing and counts zero."""
    a = Path("/a.jpg")
    items = {str(a): PhotoItem(path=a)}
    assert deselect_paths(items, [Path("/missing.jpg")]) == 0
    assert items[str(a)].selected is True


def test_paths_matching_fields_all_requires_every_field() -> None:
    """match_all selects only photos that carry every required field."""
    a, b, c = Path("/a.jpg"), Path("/b.jpg"), Path("/c.jpg")
    presence = {a: {"title", "description"}, b: {"keywords"}, c: {"title"}}
    # A title-and-description criterion skips a, but keeps the keyword-only b and title-only c.
    matched = paths_matching_fields(presence, {"title", "description"}, match_all=True)
    assert matched == {a}


def test_paths_matching_fields_any_requires_one_field() -> None:
    """Without match_all, having any one required field is enough (the 'any metadata' rule)."""
    a, b = Path("/a.jpg"), Path("/b.jpg")
    presence = {a: {"keywords"}, b: set()}
    matched = paths_matching_fields(
        presence,
        {"title", "description", "keywords"},
        match_all=False,
    )
    assert matched == {a}


def test_paths_matching_fields_empty_required_matches_nothing() -> None:
    """An empty required set never matches, so it cannot deselect every photo by accident."""
    presence = {Path("/a.jpg"): {"title"}}
    assert paths_matching_fields(presence, set(), match_all=True) == set()
    assert paths_matching_fields(presence, set(), match_all=False) == set()


def test_status_sort_rank_follows_lifecycle_order() -> None:
    """Ranks increase along the lifecycle, so a status sort is meaningful, not alphabetical."""
    ranks = [status_sort_rank(s) for s in (PENDING, WORKING, READY, SAVED, FAILED)]
    assert ranks == [0, 1, 2, 3, 4]


def test_status_sort_rank_unknown_status_sorts_last() -> None:
    """An unrecognized status ranks after every known one rather than raising."""
    assert status_sort_rank("bogus") == len(STATUS_SORT_ORDER)
    assert status_sort_rank("bogus") > status_sort_rank(FAILED)


def test_sort_photos_by_name_flips_with_direction() -> None:
    """Sorting by name is case-insensitive ascending, and descending reverses it end to end."""
    items = [PhotoItem(path=Path(f"/d/{n}")) for n in ("c.jpg", "A.jpg", "b.jpg")]
    assert [i.path.name for i in sort_photos(items, SORT_NAME, descending=False)] == [
        "A.jpg",
        "b.jpg",
        "c.jpg",
    ]
    assert [i.path.name for i in sort_photos(items, SORT_NAME, descending=True)] == [
        "c.jpg",
        "b.jpg",
        "A.jpg",
    ]


def test_sort_photos_by_status_uses_lifecycle_rank() -> None:
    """Status sorts by lifecycle rank (pending < ready < failed), not the alphabetical label."""
    a = PhotoItem(path=Path("/d/a.jpg"), status=FAILED)
    b = PhotoItem(path=Path("/d/b.jpg"), status=PENDING)
    c = PhotoItem(path=Path("/d/c.jpg"), status=READY)
    order = sort_photos([a, b, c], SORT_STATUS, descending=False)
    assert [i.path.name for i in order] == ["b.jpg", "c.jpg", "a.jpg"]


def test_sort_photos_by_type_then_name() -> None:
    """Type sorts by extension label, with the filename breaking ties within one type."""
    items = [
        PhotoItem(path=Path("/d/b.png")),
        PhotoItem(path=Path("/d/a.png")),
        PhotoItem(path=Path("/d/c.jpg")),
    ]
    order = sort_photos(items, SORT_TYPE, descending=False)
    assert [i.path.name for i in order] == ["c.jpg", "a.png", "b.png"]


def test_sort_photos_by_tagged_puts_untagged_first() -> None:
    """Tagged sorts by the compressed letters, so the '-' of an untagged photo sorts first."""
    tagged = PhotoItem(path=Path("/d/a.jpg"), known_fields={FIELD_TITLE, FIELD_KEYWORDS})
    untagged = PhotoItem(path=Path("/d/b.jpg"), known_fields=set())
    order = sort_photos([tagged, untagged], SORT_TAGGED, descending=False)
    assert [i.path.name for i in order] == ["b.jpg", "a.jpg"]


def test_photo_sort_key_is_uniformly_shaped_across_criteria() -> None:
    """Every criterion yields a same-shape string tuple, so one criterion sorts a whole list."""
    item = PhotoItem(path=Path("/d/a.jpg"), status=READY, known_fields={FIELD_TITLE})
    for criterion in (SORT_NAME, SORT_TYPE, SORT_STATUS, SORT_TAGGED):
        key = photo_sort_key(item, criterion)
        assert len(key) == 3  # noqa: PLR2004 - primary, name, full-path tiebreaks
        assert all(isinstance(part, str) for part in key)


def test_filter_photos_all_keeps_everything_in_order() -> None:
    """FILTER_ALL is a pass-through: every photo survives, order preserved."""
    items = [
        PhotoItem(path=Path("/d/a.jpg"), status=PENDING),
        PhotoItem(path=Path("/d/b.jpg"), status=SAVED),
    ]
    assert filter_photos(items, FILTER_ALL) == items


def test_filter_photos_by_status_and_selection() -> None:
    """The status and selection filters each keep only the photos in that state."""
    saved = PhotoItem(path=Path("/d/a.jpg"), status=SAVED, selected=False)
    failed = PhotoItem(path=Path("/d/b.jpg"), status=FAILED, selected=True)
    pending = PhotoItem(path=Path("/d/c.jpg"), status=PENDING, selected=True)
    items = [saved, failed, pending]
    assert filter_photos(items, FILTER_SAVED) == [saved]
    assert filter_photos(items, FILTER_FAILED) == [failed]
    assert filter_photos(items, FILTER_PENDING) == [pending]
    assert filter_photos(items, FILTER_SELECTED) == [failed, pending]


def test_filter_photos_generated_uses_has_proposal() -> None:
    """The Generated filter keys off has_proposal, not the lifecycle status."""
    generated = PhotoItem(path=Path("/d/a.jpg"), has_proposal=True)
    plain = PhotoItem(path=Path("/d/b.jpg"), has_proposal=False)
    assert filter_photos([generated, plain], FILTER_GENERATED) == [generated]


def test_filter_untagged_matches_only_scanned_empty() -> None:
    """Untagged keeps a scanned-but-empty photo, and never a tagged or still-unscanned one."""
    empty = PhotoItem(path=Path("/d/a.jpg"), known_fields=set())
    tagged = PhotoItem(path=Path("/d/b.jpg"), known_fields={FIELD_TITLE})
    unscanned = PhotoItem(path=Path("/d/c.jpg"), known_fields=None)
    assert filter_photos([empty, tagged, unscanned], FILTER_UNTAGGED) == [empty]


def test_photo_matches_filter_unknown_criterion_matches_all() -> None:
    """An unrecognized criterion behaves like FILTER_ALL rather than hiding every photo."""
    assert photo_matches_filter(PhotoItem(path=Path("/d/a.jpg")), "bogus") is True


def test_status_summary_counts_states() -> None:
    """The summary reports total, selected, generated, saved, and failed counts."""
    items = [
        PhotoItem(path=Path("/a.jpg"), has_proposal=True, status=SAVED),
        PhotoItem(path=Path("/b.jpg"), has_proposal=True, status=READY),
        PhotoItem(path=Path("/c.jpg"), selected=False, status="failed"),
    ]
    summary = status_summary(items)
    assert "3 files" in summary
    assert "2 selected" in summary
    assert "2 generated" in summary
    assert "1 saved" in summary
    assert "1 failed" in summary


def test_status_summary_uses_singular_for_one_file() -> None:
    """A single photo reads '1 file', not '1 files'."""
    assert "1 file ·" in status_summary([PhotoItem(path=Path("/a.jpg"))])


def test_config_text_with_language_sets_and_preserves() -> None:
    """Setting a language keeps every other line (comments included) intact."""
    existing = '# my config\nextensions = "jpg"\n\n[provider]\nmodel_name = "m"\n'
    result = config_text_with_language(existing, "pt_BR")
    assert 'language = "pt_BR"' in result
    assert "# my config" in result
    assert 'model_name = "m"' in result


def test_config_text_with_language_auto_removes_the_key() -> None:
    """Choosing the system default removes the pinned key instead of writing 'auto'."""
    existing = 'language = "pt_BR"\nextensions = "jpg"\n'
    result = config_text_with_language(existing, "auto")
    assert "language" not in result
    assert 'extensions = "jpg"' in result


def test_config_text_with_language_works_on_an_empty_file() -> None:
    """A missing config starts from empty text and still gets the key."""
    assert 'language = "en"' in config_text_with_language("", "en")
    assert config_text_with_language("", "auto") == ""


def test_config_text_with_output_language_sets_and_preserves() -> None:
    """Setting a metadata language keeps comments and unrelated [inference] keys intact."""
    existing = '# my config\nextensions = "jpg"\n\n[inference]\ntemperature = 0.3\n'
    result = config_text_with_output_language(existing, "German")
    assert 'output_language = "German"' in result
    assert "# my config" in result
    assert "temperature = 0.3" in result


def test_config_text_with_output_language_default_removes_the_key() -> None:
    """Choosing English again (any casing) unpins the key but keeps the table's other keys."""
    existing = '[inference]\noutput_language = "German"\ntemperature = 0.3\n'
    result = config_text_with_output_language(existing, "english")
    assert "output_language" not in result
    assert "temperature = 0.3" in result


def test_config_text_with_output_language_drops_an_emptied_table() -> None:
    """Removing the key also removes an [inference] table that held nothing else."""
    existing = 'extensions = "jpg"\n\n[inference]\noutput_language = "German"\n'
    result = config_text_with_output_language(existing, "English")
    assert "inference" not in result
    assert 'extensions = "jpg"' in result


def test_config_text_with_output_language_works_on_an_empty_file() -> None:
    """A missing config starts from empty text; the default writes nothing at all."""
    result = config_text_with_output_language("", "Spanish")
    assert "[inference]" in result
    assert 'output_language = "Spanish"' in result
    assert config_text_with_output_language("", "English") == ""


def test_output_language_suggestions_start_with_the_default() -> None:
    """The menu builder relies on the default language leading the suggestion list."""
    from photo_tagger.config import DEFAULT_OUTPUT_LANGUAGE  # noqa: PLC0415

    assert OUTPUT_LANGUAGE_SUGGESTIONS[0] == DEFAULT_OUTPUT_LANGUAGE
    # The values reach the prompt verbatim, so the list must stay in English regardless of the
    # UI language; the GUI translates the labels at display time only.
    assert "Brazilian Portuguese" in OUTPUT_LANGUAGE_SUGGESTIONS


def test_ensure_path_dirs_prepends_missing_and_dedups() -> None:
    """Absent dirs are prepended in order; ones already on PATH are not duplicated."""
    base = os.pathsep.join(["/usr/bin", "/bin"])
    out = ensure_path_dirs(base, ["/opt/homebrew/bin", "/usr/bin"])
    assert out == os.pathsep.join(["/opt/homebrew/bin", "/usr/bin", "/bin"])


def test_ensure_path_dirs_returns_input_unchanged_when_all_present() -> None:
    """When every dir is already present, the original string is returned as-is."""
    base = os.pathsep.join(["/opt/homebrew/bin", "/usr/bin"])
    assert ensure_path_dirs(base, ["/usr/bin"]) is base


def test_ensure_path_dirs_handles_empty_path() -> None:
    """An empty starting PATH yields just the added dirs (no leading separator)."""
    assert ensure_path_dirs("", ["/opt/homebrew/bin"]) == "/opt/homebrew/bin"


def test_login_shell_path_parses_shell_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """The login shell's reported PATH is split into entries, blanks dropped."""
    monkeypatch.setenv("SHELL", "/bin/zsh")
    out = os.pathsep.join(["/run/current-system/sw/bin", "/usr/bin", ""])
    monkeypatch.setattr(
        "photo_tagger.gui_state.subprocess.run",
        lambda *_a, **_k: SimpleNamespace(stdout=out),
    )
    assert login_shell_path() == ["/run/current-system/sw/bin", "/usr/bin"]


def test_login_shell_path_empty_without_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no $SHELL there is nothing to ask, so it returns []."""
    monkeypatch.delenv("SHELL", raising=False)
    assert login_shell_path() == []


def test_login_shell_path_empty_on_shell_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A shell that errors or times out degrades to [] rather than raising."""
    monkeypatch.setenv("SHELL", "/bin/zsh")

    def boom(*_a: object, **_k: object) -> object:
        raise OSError

    monkeypatch.setattr("photo_tagger.gui_state.subprocess.run", boom)
    assert login_shell_path() == []
