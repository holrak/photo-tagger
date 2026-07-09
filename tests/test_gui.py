# mypy: ignore-errors
"""
Headless tests for the PySide6 desktop GUI.

Skipped entirely when PySide6 is absent (CLI-only setups and CI, which do not install the optional
``[gui]`` extra). When present, Qt's ``offscreen`` platform builds real widgets without a display
server, so the tree, the detail pane, and the generation worker can be exercised for real.

The ``# mypy: ignore-errors`` header opts this file out of the zuban type check, for the same reason
gui.py does: PySide6 ships no stubs and is not installed in the lint job, so a strict run would only
see an unresolved-import error here.
"""

import os
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest


pytest.importorskip("PySide6")
# Must be set before the first QApplication is created.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QTreeWidgetItem

from photo_tagger import gui, telemetry
from photo_tagger.errors import ProviderError
from photo_tagger.gui_state import (
    FAILED,
    PENDING,
    READY,
    SAVED,
    WORKING,
    Proposal,
)
from photo_tagger.metadata import FIELD_DESCRIPTION, FIELD_KEYWORDS, FIELD_TITLE, ImageContext
from photo_tagger.models import InferenceResult, KeywordSet
from photo_tagger.providers import PROVIDER_LABELS, PROVIDER_NAMES


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """One QApplication for the module (Qt allows only a single instance)."""
    return cast("QApplication", QApplication.instance() or QApplication([]))


@pytest.fixture
def window(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> Iterator[gui.MainWindow]:
    """Yield a fresh main window, closing it (and its thread) after each test."""
    win = gui.MainWindow()
    # Adding photos kicks off the background exiftool scan for the Tagged column; keep tests
    # deterministic (and exiftool-free) by driving _on_scan_done directly where needed.
    monkeypatch.setattr(win, "_start_metadata_scan", lambda: None)
    yield win
    win.close()


def _jpeg(path: Path) -> Path:
    """Write a tiny real JPEG so previews and exiftool have something to act on."""
    # Imported lazily so this module only needs Pillow when the GUI tests actually run.
    from PIL import Image  # noqa: PLC0415

    buf = BytesIO()
    Image.new("RGB", (8, 8), color="red").save(buf, format="JPEG")
    path.write_bytes(buf.getvalue())
    return path


def _stub_reads(monkeypatch: pytest.MonkeyPatch, *, keywords: list[str]) -> None:
    """Patch the exiftool-backed reads so tests need no exiftool binary."""
    monkeypatch.setattr(gui, "read_caption", lambda _p: ("Old Title", "Old caption."))
    monkeypatch.setattr(
        gui,
        "read_image_context",
        lambda _p, **_kwargs: ImageContext(existing_keywords=KeywordSet(subject=keywords)),
    )


def _add_dir(window: gui.MainWindow, files: dict[str, Path]) -> None:
    """Set jpg extensions and add the directory the files live in."""
    folder = next(iter(files.values())).parent
    window._extensions.setText("jpg")  # noqa: SLF001
    window._add_inputs([folder])  # noqa: SLF001


def _select(window: gui.MainWindow, item: QTreeWidgetItem | None) -> None:
    """Select a tree item, asserting it was actually found first."""
    assert item is not None
    window._tree.setCurrentItem(item)  # noqa: SLF001


def _check_state(window: gui.MainWindow, path: Path) -> Qt.CheckState:
    """Return the rendered checkbox state of *path*'s tree leaf (asserting it exists)."""
    leaf = window._leaf_for(path)  # noqa: SLF001
    assert leaf is not None
    return leaf.checkState(0)


# ---------------------------------------------------------------------------
# Toolbar
# ---------------------------------------------------------------------------


def test_window_builds_with_capitalized_providers(window: gui.MainWindow) -> None:
    """The combo shows capitalized labels but carries the internal names as data."""
    assert "Photo Tagger" in window.windowTitle()
    combo = window._provider  # noqa: SLF001
    names = [combo.itemData(i) for i in range(combo.count())]
    labels = [combo.itemText(i) for i in range(combo.count())]
    assert names == list(PROVIDER_NAMES)
    assert labels == [PROVIDER_LABELS[n] for n in PROVIDER_NAMES]
    assert "LM Studio" in labels


def test_provider_name_returns_internal_value(window: gui.MainWindow) -> None:
    """Selecting a label resolves back to the internal provider name."""
    window._provider.setCurrentIndex(list(PROVIDER_NAMES).index("openai"))  # noqa: SLF001
    assert window._provider_name() == "openai"  # noqa: SLF001


def test_refresh_models_populates_combo_vision_first(
    window: gui.MainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refresh queries the backend and lists models, vision-likely ones first."""
    fake = SimpleNamespace(
        default_base_url="http://localhost/v1",
        resolve_api_key=lambda _k: None,
        list_models=lambda _base, _key: ["llama3", "qwen3-vl-30b"],
    )
    monkeypatch.setattr(gui, "get_backend", lambda _name: fake)
    window._refresh_models()  # noqa: SLF001
    items = [window._model.itemText(i) for i in range(window._model.count())]  # noqa: SLF001
    assert items[0] == "qwen3-vl-30b"
    assert "llama3" in items


# ---------------------------------------------------------------------------
# API key field
# ---------------------------------------------------------------------------


def test_api_key_value_strips_and_blank_is_none(window: gui.MainWindow) -> None:
    """A typed key is trimmed; a blank field means "use the env var" (None)."""
    window._api_key.setText("  sk-typed  ")  # noqa: SLF001
    assert window._api_key_value() == "sk-typed"  # noqa: SLF001
    window._api_key.setText("   ")  # noqa: SLF001
    assert window._api_key_value() is None  # noqa: SLF001


def test_api_key_field_is_masked(window: gui.MainWindow) -> None:
    """The key field hides its contents so a shoulder-surfer cannot read it."""
    from PySide6.QtWidgets import QLineEdit  # noqa: PLC0415

    assert window._api_key.echoMode() == QLineEdit.EchoMode.Password  # noqa: SLF001


def test_refresh_models_passes_typed_api_key(
    window: gui.MainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A key typed into the field is handed to the backend's key resolution."""
    window._api_key.setText("sk-refresh")  # noqa: SLF001
    seen: dict[str, str | None] = {}
    fake = SimpleNamespace(
        default_base_url="http://localhost/v1",
        resolve_api_key=lambda key: seen.setdefault("key", key),
        list_models=lambda _base, _key: ["m"],
    )
    monkeypatch.setattr(gui, "get_backend", lambda _name: fake)
    window._refresh_models()  # noqa: SLF001
    assert seen["key"] == "sk-refresh"


def test_worker_passes_typed_api_key_to_create_agent(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generate worker forwards its api_key to create_agent."""
    captured: dict[str, object] = {}

    def fake_create_agent(*_a: object, **kwargs: object) -> object:
        captured.update(kwargs)
        msg = "stop before per-photo work"
        raise ProviderError(msg)

    monkeypatch.setattr(gui, "create_agent", fake_create_agent)
    worker = gui.GenerateWorker("openai", "m", None, [Path("/a.jpg")], api_key="sk-worker")
    worker.file_failed.connect(lambda *_a: None)
    worker.run()
    assert captured["api_key"] == "sk-worker"


def test_worker_passes_output_language_to_create_agent(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generate worker forwards its metadata language to create_agent."""
    captured: dict[str, object] = {}

    def fake_create_agent(*_a: object, **kwargs: object) -> object:
        captured.update(kwargs)
        msg = "stop before per-photo work"
        raise ProviderError(msg)

    monkeypatch.setattr(gui, "create_agent", fake_create_agent)
    worker = gui.GenerateWorker(
        "lmstudio",
        "m",
        None,
        [Path("/a.jpg")],
        output_language="German",
    )
    worker.file_failed.connect(lambda *_a: None)
    worker.run()
    assert captured["output_language"] == "German"


def test_run_generation_builds_worker_with_typed_api_key(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Starting a run hands the typed key to the worker it spawns."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_generation(monkeypatch)  # so the background worker does no real I/O
    _add_dir(window, {"a": img})
    window._api_key.setText("sk-run")  # noqa: SLF001

    window._run_generation([window._items[str(img)]])  # noqa: SLF001

    worker = window._worker  # noqa: SLF001
    assert worker is not None
    assert worker._api_key == "sk-run"  # noqa: SLF001
    window._teardown_thread()  # noqa: SLF001 - join the worker thread the run started


def test_run_generation_hands_the_metadata_language_to_the_worker(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Starting a run passes the window's metadata language to the worker it spawns."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_generation(monkeypatch)  # so the background worker does no real I/O
    _add_dir(window, {"a": img})
    window._output_language = "German"  # noqa: SLF001

    window._run_generation([window._items[str(img)]])  # noqa: SLF001

    worker = window._worker  # noqa: SLF001
    assert worker is not None
    assert worker._output_language == "German"  # noqa: SLF001
    window._teardown_thread()  # noqa: SLF001 - join the worker thread the run started


# ---------------------------------------------------------------------------
# Tree: nesting, selection, removal
# ---------------------------------------------------------------------------


def test_add_inputs_builds_a_nested_tree(window: gui.MainWindow, tmp_path: Path) -> None:
    """A folder with a subfolder nests under one top node."""
    (tmp_path / "sub").mkdir()
    files = {"a": _jpeg(tmp_path / "a.jpg"), "b": _jpeg(tmp_path / "sub" / "b.jpg")}
    _add_dir(window, files)

    assert len(window._items) == 2  # noqa: SLF001, PLR2004 - two photos
    assert window._tree.topLevelItemCount() == 1  # noqa: SLF001
    top = window._tree.topLevelItem(0)  # noqa: SLF001
    assert top is not None
    assert bool(top.data(0, gui._IS_DIR_ROLE))  # noqa: SLF001 - the top node is a folder
    # One file leaf and one "sub" folder under the top.
    kinds = {bool(top.child(i).data(0, gui._IS_DIR_ROLE)) for i in range(top.childCount())}  # noqa: SLF001
    assert kinds == {True, False}


def test_unchecking_folder_deselects_descendants(window: gui.MainWindow, tmp_path: Path) -> None:
    """Unchecking the top folder deselects every photo beneath it."""
    _add_dir(window, {"a": _jpeg(tmp_path / "a.jpg"), "b": _jpeg(tmp_path / "b.jpg")})
    top = window._tree.topLevelItem(0)  # noqa: SLF001
    assert top is not None
    top.setCheckState(0, Qt.CheckState.Unchecked)
    assert all(not item.selected for item in window._items.values())  # noqa: SLF001


def test_remove_selected_folder_drops_its_files(window: gui.MainWindow, tmp_path: Path) -> None:
    """Removing a folder node removes all photos under it (not just deselects)."""
    _add_dir(window, {"a": _jpeg(tmp_path / "a.jpg"), "b": _jpeg(tmp_path / "b.jpg")})
    _select(window, window._tree.topLevelItem(0))  # noqa: SLF001
    window._remove_selected()  # noqa: SLF001
    assert window._items == {}  # noqa: SLF001
    assert window._tree.topLevelItemCount() == 0  # noqa: SLF001


def test_remove_selected_file_drops_one(window: gui.MainWindow, tmp_path: Path) -> None:
    """Removing a single file leaf leaves the rest in place."""
    a = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": a, "b": _jpeg(tmp_path / "b.jpg")})
    leaf = window._leaf_for(a)  # noqa: SLF001
    assert leaf is not None
    window._tree.setCurrentItem(leaf)  # noqa: SLF001
    window._remove_selected()  # noqa: SLF001
    assert str(a) not in window._items  # noqa: SLF001
    assert len(window._items) == 1  # noqa: SLF001


def test_clear_empties_everything(window: gui.MainWindow, tmp_path: Path) -> None:
    """Clear drops all items and the tree."""
    _add_dir(window, {"a": _jpeg(tmp_path / "a.jpg")})
    window._clear()  # noqa: SLF001
    assert window._items == {}  # noqa: SLF001
    assert window._tree.topLevelItemCount() == 0  # noqa: SLF001


# ---------------------------------------------------------------------------
# Tree sorting (click a header to sort by name or status)
# ---------------------------------------------------------------------------


def _leaf_order(window: gui.MainWindow) -> list[str]:
    """Return the file-leaf names under the single top folder, in display order."""
    top = window._tree.topLevelItem(0)  # noqa: SLF001
    assert top is not None
    return [top.child(i).text(0) for i in range(top.childCount())]


def test_tree_sorts_by_name(window: gui.MainWindow, tmp_path: Path) -> None:
    """Clicking the Photos header orders leaves by name, not creation order."""
    for name in ("c.jpg", "a.jpg", "b.jpg"):  # created out of order
        _jpeg(tmp_path / name)
    window._extensions.setText("jpg")  # noqa: SLF001
    window._add_inputs([tmp_path])  # noqa: SLF001

    window._tree.sortByColumn(0, Qt.SortOrder.AscendingOrder)  # noqa: SLF001

    assert _leaf_order(window) == ["a.jpg", "b.jpg", "c.jpg"]


def test_tree_sorts_by_status_using_lifecycle_rank(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """Sorting the Status column uses the lifecycle rank, not the alphabetical label."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    c = _jpeg(tmp_path / "c.jpg")
    window._extensions.setText("jpg")  # noqa: SLF001
    window._add_inputs([tmp_path])  # noqa: SLF001
    window._items[str(a)].status = FAILED  # noqa: SLF001
    window._items[str(b)].status = PENDING  # noqa: SLF001
    window._items[str(c)].status = READY  # noqa: SLF001
    for path in (a, b, c):
        window._refresh_status_cell(window._items[str(path)])  # noqa: SLF001

    window._tree.sortByColumn(2, Qt.SortOrder.AscendingOrder)  # noqa: SLF001 - Status column

    # Ascending lifecycle: pending(b) < ready(c) < failed(a). Alphabetical-by-label would differ.
    assert _leaf_order(window) == ["b.jpg", "c.jpg", "a.jpg"]


def test_tree_keeps_folders_above_files(window: gui.MainWindow, tmp_path: Path) -> None:
    """A subfolder stays grouped above sibling files even when its name sorts after them."""
    (tmp_path / "zzz").mkdir()
    _jpeg(tmp_path / "a.jpg")  # top-level file, name sorts before "zzz"
    _jpeg(tmp_path / "zzz" / "deep.jpg")
    window._extensions.setText("jpg")  # noqa: SLF001
    window._add_inputs([tmp_path])  # noqa: SLF001

    window._tree.sortByColumn(0, Qt.SortOrder.AscendingOrder)  # noqa: SLF001

    top = window._tree.topLevelItem(0)  # noqa: SLF001
    assert top is not None
    first = top.child(0)
    assert bool(first.data(0, gui._IS_DIR_ROLE))  # noqa: SLF001 - the folder, not "a.jpg"
    assert first.text(0) == "zzz"


# ---------------------------------------------------------------------------
# Deselect: skip already-tagged, skip from a list (CLI --skip-tagged/--skip-from)
# ---------------------------------------------------------------------------


def _stub_field_presence(monkeypatch: pytest.MonkeyPatch, by_name: dict[str, set[str]]) -> None:
    """Patch find_field_presence to report fields per filename, on the passed-in path objects."""
    monkeypatch.setattr(
        gui,
        "find_field_presence",
        lambda paths: {p: set(by_name.get(p.name, set())) for p in paths},
    )


def test_deselect_tagged_field_aware_keeps_keyword_only_photos(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """'Title and description' skips photos that have both, keeping keyword-only ones selected."""
    a = _jpeg(tmp_path / "a.jpg")  # already has a title and a description
    b = _jpeg(tmp_path / "b.jpg")  # only keywords -> must stay selected for description generation
    _add_dir(window, {"a": a, "b": b})
    _stub_field_presence(
        monkeypatch,
        {"a.jpg": {FIELD_TITLE, FIELD_DESCRIPTION}, "b.jpg": {FIELD_KEYWORDS}},
    )

    window._deselect_tagged(  # noqa: SLF001
        frozenset({FIELD_TITLE, FIELD_DESCRIPTION}),
        match_all=True,
        phrase="a title and a description",
    )

    assert window._items[str(a)].selected is False  # noqa: SLF001
    assert window._items[str(b)].selected is True  # noqa: SLF001
    # The rendered tree checkbox, not just the model flag, must reflect the deselection.
    assert _check_state(window, a) == Qt.CheckState.Unchecked
    assert _check_state(window, b) == Qt.CheckState.Checked
    assert "Deselected 1" in window._status.text()  # noqa: SLF001
    assert "a title and a description" in window._status.text()  # noqa: SLF001


def test_deselect_tagged_any_metadata_uses_or_semantics(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 'any metadata' criterion deselects a photo that has even one indicator field."""
    a = _jpeg(tmp_path / "a.jpg")  # keywords only
    b = _jpeg(tmp_path / "b.jpg")  # nothing
    _add_dir(window, {"a": a, "b": b})
    _stub_field_presence(monkeypatch, {"a.jpg": {FIELD_KEYWORDS}})

    window._deselect_tagged(  # noqa: SLF001
        frozenset({FIELD_TITLE, FIELD_DESCRIPTION, FIELD_KEYWORDS}),
        match_all=False,
        phrase="any metadata",
    )

    assert window._items[str(a)].selected is False  # noqa: SLF001
    assert window._items[str(b)].selected is True  # noqa: SLF001


def test_deselect_tagged_reports_when_none_match(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no checked photo has the field, the action says so instead of 'Deselected 0'."""
    a = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": a})
    _stub_field_presence(monkeypatch, {})  # no photo has any field

    window._deselect_tagged(  # noqa: SLF001
        frozenset({FIELD_DESCRIPTION}),
        match_all=True,
        phrase="a description",
    )

    assert window._items[str(a)].selected is True  # noqa: SLF001
    assert "No checked photos have a description" in window._status.text()  # noqa: SLF001


def test_tagged_presets_all_run_without_error(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every menu preset is wired to a real criterion the deselect handler accepts."""
    a = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": a})
    _stub_field_presence(monkeypatch, {})
    for _label, required, match_all, phrase in gui._TAGGED_PRESETS:  # noqa: SLF001
        window._deselect_tagged(required, match_all=match_all, phrase=phrase)  # noqa: SLF001
        assert phrase in window._status.text()  # noqa: SLF001


def test_deselect_tagged_with_no_photos_nags(window: gui.MainWindow) -> None:
    """With nothing added, the action reports it instead of calling exiftool."""
    window._deselect_tagged(  # noqa: SLF001
        frozenset({FIELD_TITLE}),
        match_all=True,
        phrase="a title",
    )
    assert "Add photos" in window._status.text()  # noqa: SLF001


def test_already_tagged_menu_actions_route_to_each_preset(
    window: gui.MainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each preset in the Select menu triggers _deselect_tagged with it (no late-binding bug)."""
    actions = window._tagged_menu.actions()  # noqa: SLF001
    assert len(actions) == len(gui._TAGGED_PRESETS)  # noqa: SLF001

    captured: list[tuple[frozenset[str], bool, str]] = []
    monkeypatch.setattr(
        window,
        "_deselect_tagged",
        lambda req, *, match_all, phrase: captured.append((req, match_all, phrase)),
    )
    for action in actions:
        action.trigger()

    assert captured == [(r, m, p) for _label, r, m, p in gui._TAGGED_PRESETS]  # noqa: SLF001


def test_select_menu_checks_and_unchecks_all(window: gui.MainWindow, tmp_path: Path) -> None:
    """Uncheck All clears every checkbox (model and rendered tree); Check All restores them."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    _add_dir(window, {"a": a, "b": b})

    window._set_all_checked(checked=False)  # noqa: SLF001
    assert all(not item.selected for item in window._items.values())  # noqa: SLF001
    assert _check_state(window, a) == Qt.CheckState.Unchecked

    window._set_all_checked(checked=True)  # noqa: SLF001
    assert all(item.selected for item in window._items.values())  # noqa: SLF001
    assert _check_state(window, b) == Qt.CheckState.Checked


def test_set_all_checked_with_no_photos_nags(window: gui.MainWindow) -> None:
    """With nothing added, the bulk-select actions report it instead of rebuilding the tree."""
    window._set_all_checked(checked=False)  # noqa: SLF001
    assert "Add photos" in window._status.text()  # noqa: SLF001


def test_apply_skip_file_unchecks_listed_photos(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """A skip-list file deselects every photo it names, by bare filename or full path."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    c = _jpeg(tmp_path / "c.jpg")
    _add_dir(window, {"a": a, "b": b, "c": c})
    skip = tmp_path / "skip.txt"
    skip.write_text(f"a.jpg\n{c}\n", encoding="utf-8")

    window._apply_skip_file(skip)  # noqa: SLF001

    assert window._items[str(a)].selected is False  # noqa: SLF001
    assert window._items[str(b)].selected is True  # noqa: SLF001
    assert window._items[str(c)].selected is False  # noqa: SLF001
    # The rendered tree must repaint: a and c unchecked, b still checked.
    assert _check_state(window, a) == Qt.CheckState.Unchecked
    assert _check_state(window, b) == Qt.CheckState.Checked
    assert _check_state(window, c) == Qt.CheckState.Unchecked
    assert "Deselected 2" in window._status.text()  # noqa: SLF001


def test_apply_skip_file_reports_empty_list(window: gui.MainWindow, tmp_path: Path) -> None:
    """A comment-only (no usable entries) file says so rather than 'Deselected 0'."""
    a = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": a})
    skip = tmp_path / "empty.txt"
    skip.write_text("# only a comment\n\n", encoding="utf-8")

    window._apply_skip_file(skip)  # noqa: SLF001

    assert window._items[str(a)].selected is True  # noqa: SLF001
    assert "no usable entries" in window._status.text()  # noqa: SLF001


def test_apply_skip_file_reports_no_matches(window: gui.MainWindow, tmp_path: Path) -> None:
    """A non-empty list that matches nothing reports it, leaving every photo checked."""
    a = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": a})
    skip = tmp_path / "skip.txt"
    skip.write_text("nonexistent.jpg\n", encoding="utf-8")

    window._apply_skip_file(skip)  # noqa: SLF001

    assert window._items[str(a)].selected is True  # noqa: SLF001
    assert "matched" in window._status.text().lower()  # noqa: SLF001


def test_apply_skip_file_warns_on_unreadable_file(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreadable skip list pops a warning and changes no selection."""
    a = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": a})
    warned: list[str] = []
    monkeypatch.setattr(
        gui,
        "QMessageBox",
        SimpleNamespace(warning=lambda *args, **_k: warned.append(str(args[-1]))),
    )

    window._apply_skip_file(tmp_path / "missing.txt")  # noqa: SLF001

    assert warned  # a warning dialog was shown
    assert window._items[str(a)].selected is True  # noqa: SLF001


# ---------------------------------------------------------------------------
# Detail pane
# ---------------------------------------------------------------------------


def test_selecting_a_photo_loads_and_populates(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selecting a file shows existing metadata and seeds the editable fields."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=["Beach"])
    _add_dir(window, {"a": img})
    leaf = window._leaf_for(img)  # noqa: SLF001
    assert leaf is not None
    window._tree.setCurrentItem(leaf)  # noqa: SLF001 - triggers _on_current_changed

    assert window._existing_title.text() == "Old Title"  # noqa: SLF001
    assert "Beach" in window._existing_keywords.toPlainText()  # noqa: SLF001
    assert window._title.text() == "Old Title"  # noqa: SLF001
    assert window._keywords.toPlainText() == "Beach"  # noqa: SLF001


def test_hierarchy_preview_updates_from_keywords(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Editing the keywords field refreshes the resulting-hierarchy preview."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": img})
    _select(window, window._leaf_for(img))  # noqa: SLF001
    window._overwrite.setChecked(True)  # noqa: SLF001
    window._keywords.setPlainText("Duck<Bird<Animal")  # noqa: SLF001 - triggers textChanged
    assert window._hierarchy.toPlainText() == "Animal\n└─ Bird\n   └─ Duck"  # noqa: SLF001


def test_save_current_writes_and_marks_saved(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Save merges keywords, calls write_metadata, and marks the item saved."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=["Beach"])
    captured: dict[str, object] = {}

    def fake_write(*_a: object, **kwargs: object) -> bool:
        captured.update(kwargs)
        return True

    monkeypatch.setattr(gui, "write_metadata", fake_write)
    _add_dir(window, {"a": img})
    _select(window, window._leaf_for(img))  # noqa: SLF001
    window._title.setText("New Title")  # noqa: SLF001
    window._keywords.setPlainText("Eagle\nSky")  # noqa: SLF001
    window._save_current()  # noqa: SLF001

    assert window._items[str(img)].status == SAVED  # noqa: SLF001
    assert captured["title"] == "New Title"


def test_selecting_shows_metadata_source(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Source field reports where the existing metadata was read from."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=["Beach"])
    monkeypatch.setattr(gui, "read_metadata_sources", lambda _p: ["XMP sidecar"])
    _add_dir(window, {"a": img})
    _select(window, window._leaf_for(img))  # noqa: SLF001
    assert window._existing_source.text() == "(from XMP sidecar)"  # noqa: SLF001


# ---------------------------------------------------------------------------
# Empty / placeholder state (the idle right pane)
# ---------------------------------------------------------------------------


def test_empty_state_is_shown_at_startup(window: gui.MainWindow) -> None:
    """A fresh window shows the placeholder page, not an empty detail form."""
    assert window._right.currentIndex() == gui._PAGE_EMPTY  # noqa: SLF001
    assert "Add photos" in window._empty_message.text()  # noqa: SLF001


def test_empty_state_nudges_to_pick_a_photo_once_loaded(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """With photos loaded but none open, the placeholder asks the user to pick one."""
    _add_dir(window, {"a": _jpeg(tmp_path / "a.jpg")})
    window._on_current_changed(None, None)  # noqa: SLF001 - selection cleared, no photo open

    assert window._right.currentIndex() == gui._PAGE_EMPTY  # noqa: SLF001
    assert "Select a photo" in window._empty_message.text()  # noqa: SLF001


def test_clear_returns_to_the_empty_state(window: gui.MainWindow, tmp_path: Path) -> None:
    """Clearing the list returns the right pane to the getting-started placeholder."""
    _add_dir(window, {"a": _jpeg(tmp_path / "a.jpg")})
    window._clear()  # noqa: SLF001

    assert window._right.currentIndex() == gui._PAGE_EMPTY  # noqa: SLF001
    assert "Add photos" in window._empty_message.text()  # noqa: SLF001


def test_removing_the_open_photo_returns_to_the_empty_state(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing the photo being inspected drops back to the placeholder, not a stale form."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": img})
    _select(window, window._leaf_for(img))  # noqa: SLF001 - open it -> detail page
    assert window._right.currentIndex() == gui._PAGE_DETAIL  # noqa: SLF001

    window._tree.setCurrentItem(window._leaf_for(img))  # noqa: SLF001
    window._remove_selected()  # noqa: SLF001

    assert window._current is None  # noqa: SLF001
    assert window._right.currentIndex() == gui._PAGE_EMPTY  # noqa: SLF001


# ---------------------------------------------------------------------------
# Export CSV
# ---------------------------------------------------------------------------


def _patch_save_dialog(monkeypatch: pytest.MonkeyPatch, path: Path | str) -> None:
    """Replace the save-file dialog so it returns *path* without opening a real dialog."""
    monkeypatch.setattr(
        gui,
        "QFileDialog",
        SimpleNamespace(getSaveFileName=lambda *_a, **_k: (str(path), "CSV files (*.csv)")),
    )


def test_export_csv_without_items_shows_hint(window: gui.MainWindow) -> None:
    """Exporting with an empty list nudges the user instead of writing a file."""
    window._export_csv()  # noqa: SLF001
    assert "Add photos" in window._status.text()  # noqa: SLF001


def test_export_csv_writes_report(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Export writes one CSV row per photo, folding in the edited working fields."""
    import csv as csv_module  # noqa: PLC0415 - test-local parser.

    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=["Beach"])
    _add_dir(window, {"a": img})
    _select(window, window._leaf_for(img))  # noqa: SLF001
    window._title.setText("New Title")  # noqa: SLF001
    window._keywords.setPlainText("Eagle\nSky")  # noqa: SLF001

    target = tmp_path / "report.csv"
    _patch_save_dialog(monkeypatch, target)
    window._export_csv()  # noqa: SLF001

    with target.open(encoding="utf-8", newline="") as fh:
        rows = list(csv_module.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["filename"] == "a.jpg"
    assert rows[0]["title"] == "New Title"
    assert "Eagle" in rows[0]["keywords"]
    assert "Beach" in rows[0]["keywords"]  # merged with the existing keyword
    assert rows[0]["existing_keywords"] == "Beach"
    assert "Exported 1" in window._status.text()  # noqa: SLF001


def test_export_csv_appends_missing_suffix(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chosen path without a .csv suffix gets one so the file is unambiguous."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": img})
    _patch_save_dialog(monkeypatch, tmp_path / "report")  # no extension
    window._export_csv()  # noqa: SLF001
    assert (tmp_path / "report.csv").exists()


def test_export_csv_cancel_writes_nothing(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling the dialog (empty path) leaves the filesystem untouched."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": img})
    _patch_save_dialog(monkeypatch, "")
    window._export_csv()  # noqa: SLF001
    assert list(tmp_path.glob("*.csv")) == []


def test_export_csv_reports_write_error(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A write failure surfaces a warning dialog rather than crashing the GUI."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": img})
    _patch_save_dialog(monkeypatch, tmp_path / "report.csv")

    def boom(*_a: object, **_k: object) -> None:
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr(gui, "write_report", boom)
    messages: list[str] = []
    monkeypatch.setattr(
        gui,
        "QMessageBox",
        SimpleNamespace(warning=lambda _parent, _title, text, *_a, **_k: messages.append(text)),
    )
    window._export_csv()  # noqa: SLF001
    assert messages
    assert "disk full" in messages[0]


def test_save_selected_writes_only_checked_generated_items(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Save selected writes checked photos with a proposal; an unchecked one is skipped."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    _stub_reads(monkeypatch, keywords=[])
    written: list[str] = []
    monkeypatch.setattr(
        gui,
        "write_metadata",
        lambda path, *_a, **_k: written.append(path.name) or True,
    )
    _add_dir(window, {"a": a, "b": b})
    # Both generated, but only "a" is checked; "b" is generated yet unchecked -> skipped.
    for name, selected in ((a, True), (b, False)):
        item = window._items[str(name)]  # noqa: SLF001
        item.has_proposal = True
        item.selected = selected
        item.title = "T"
        item.keywords = ["Eagle"]

    window._save_selected()  # noqa: SLF001
    assert written == ["a.jpg"]
    assert window._items[str(a)].status == SAVED  # noqa: SLF001


def test_save_marks_failed_when_write_fails(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed write_metadata sets the item status to failed."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    monkeypatch.setattr(gui, "write_metadata", lambda *_a, **_k: False)
    _add_dir(window, {"a": img})
    _select(window, window._leaf_for(img))  # noqa: SLF001
    window._save_current()  # noqa: SLF001
    assert window._items[str(img)].status == FAILED  # noqa: SLF001


# ---------------------------------------------------------------------------
# Per-field write toggles (Title / Description / Keywords)
# ---------------------------------------------------------------------------


def _capture_write(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Patch write_metadata to record its positional keywords and keyword arguments."""
    captured: dict[str, object] = {}

    def fake_write(path: Path, keywords: object, **kwargs: object) -> bool:
        captured["path"] = path
        captured["keywords"] = keywords
        captured.update(kwargs)
        return True

    monkeypatch.setattr(gui, "write_metadata", fake_write)
    return captured


def test_save_writes_only_title_and_description_when_keywords_unchecked(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unchecking 'Keywords' writes title + description and hands write_metadata no keywords."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=["Beach"])
    captured = _capture_write(monkeypatch)
    _add_dir(window, {"a": img})
    _select(window, window._leaf_for(img))  # noqa: SLF001
    window._title.setText("New Title")  # noqa: SLF001
    window._description.setPlainText("New description.")  # noqa: SLF001
    window._keywords.setPlainText("Eagle")  # noqa: SLF001
    window._write_keywords.setChecked(False)  # noqa: SLF001

    window._save_current()  # noqa: SLF001

    assert captured["title"] == "New Title"
    assert captured["description"] == "New description."
    # An empty KeywordSet means write_metadata emits no keyword tags, so existing ones survive.
    assert captured["keywords"].subject == []  # type: ignore[attr-defined]


def test_save_skips_title_when_title_unchecked(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unchecking 'Title' nulls the title kwarg while keywords and description still write."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    captured = _capture_write(monkeypatch)
    _add_dir(window, {"a": img})
    _select(window, window._leaf_for(img))  # noqa: SLF001
    window._title.setText("New Title")  # noqa: SLF001
    window._write_title.setChecked(False)  # noqa: SLF001

    window._save_current()  # noqa: SLF001

    assert captured["title"] is None


def test_save_with_no_write_fields_nags_and_writes_nothing(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With every write toggle off, Save reports it and never touches the file."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    wrote: list[int] = []
    monkeypatch.setattr(gui, "write_metadata", lambda *_a, **_k: wrote.append(1) or True)
    _add_dir(window, {"a": img})
    _select(window, window._leaf_for(img))  # noqa: SLF001
    for checkbox in (window._write_title, window._write_description, window._write_keywords):  # noqa: SLF001
        checkbox.setChecked(False)

    window._save_current()  # noqa: SLF001

    assert wrote == []
    assert "at least one field" in window._status.text()  # noqa: SLF001


def test_unchecking_write_keywords_disables_overwrite_and_blanks_diff(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Turning keywords off grays out Overwrite and shows the diff is moot."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=["Beach"])
    _add_dir(window, {"a": img})
    _select(window, window._leaf_for(img))  # noqa: SLF001

    window._write_keywords.setChecked(False)  # noqa: SLF001 - fires _on_write_keywords_toggled

    assert not window._overwrite.isEnabled()  # noqa: SLF001
    assert "keywords will not be written" in window._diff.toPlainText().lower()  # noqa: SLF001


# ---------------------------------------------------------------------------
# Logs and failure reporting
# ---------------------------------------------------------------------------


def test_file_failure_surfaces_reason_on_the_open_photo(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed photo records its reason, shows the error banner, and tooltips the tree cell."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": img})
    _select(window, window._leaf_for(img))  # noqa: SLF001 - open the photo

    window._on_file_failed(str(img), "model unreachable")  # noqa: SLF001

    item = window._items[str(img)]  # noqa: SLF001
    assert item.status == FAILED
    assert item.error == "model unreachable"
    assert not window._error_banner.isHidden()  # noqa: SLF001
    assert "model unreachable" in window._error_banner.text()  # noqa: SLF001
    leaf = window._leaf_for(img)  # noqa: SLF001
    assert leaf is not None
    assert leaf.toolTip(2) == "model unreachable"  # Status column


def test_opening_a_healthy_photo_hides_the_error_banner(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The banner only shows for the failed photo; opening a healthy one clears it."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": a, "b": b})
    _select(window, window._leaf_for(a))  # noqa: SLF001
    window._on_file_failed(str(a), "boom")  # noqa: SLF001
    assert not window._error_banner.isHidden()  # noqa: SLF001

    _select(window, window._leaf_for(b))  # noqa: SLF001 - healthy photo
    assert window._error_banner.isHidden()  # noqa: SLF001


def test_retry_failed_targets_only_failed_photos(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry failed re-runs every failed photo and leaves the rest alone."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    _add_dir(window, {"a": a, "b": b})
    window._items[str(a)].status = FAILED  # noqa: SLF001
    window._items[str(b)].status = READY  # noqa: SLF001

    captured: list[list[Path]] = []
    monkeypatch.setattr(
        window,
        "_run_generation",
        lambda items: captured.append([i.path for i in items]),
    )
    window._retry_failed()  # noqa: SLF001

    assert captured == [[a]]


def test_retry_failed_with_nothing_failed_nags(window: gui.MainWindow, tmp_path: Path) -> None:
    """With no failures, Retry failed reports it instead of starting a run."""
    _add_dir(window, {"a": _jpeg(tmp_path / "a.jpg")})
    window._retry_failed()  # noqa: SLF001
    assert window._thread is None  # noqa: SLF001
    assert "No failed photos" in window._status.text()  # noqa: SLF001


def test_retrying_the_open_photo_clears_its_banner(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-queuing the open failed photo hides the stale banner and marks it working."""
    img = _jpeg(tmp_path / "a.jpg")
    # Stub the whole generation path so the background worker does no real I/O.
    _stub_generation(monkeypatch)
    _add_dir(window, {"a": img})
    _select(window, window._leaf_for(img))  # noqa: SLF001
    window._on_file_failed(str(img), "boom")  # noqa: SLF001
    assert not window._error_banner.isHidden()  # noqa: SLF001

    window._run_generation([window._items[str(img)]])  # noqa: SLF001

    # Checked synchronously, before the worker thread's queued signals are processed: the
    # pre-run bookkeeping marks the photo working and clears its stale failure banner.
    assert window._items[str(img)].status == WORKING  # noqa: SLF001
    assert window._error_banner.isHidden()  # noqa: SLF001
    window._teardown_thread()  # noqa: SLF001 - join the worker thread the run started


def test_open_logs_creates_the_folder_and_reveals_it(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Open logs makes sure the folder exists and hands it to the OS file browser."""
    folder = tmp_path / "logs"
    monkeypatch.setattr(gui, "_LOG_FOLDER", folder)
    opened: list[str] = []
    monkeypatch.setattr(
        gui,
        "QDesktopServices",
        SimpleNamespace(openUrl=lambda url: opened.append(url.toLocalFile()) or True),
    )
    window._open_logs()  # noqa: SLF001
    assert folder.is_dir()
    assert opened == [str(folder)]


# ---------------------------------------------------------------------------
# Folder thumbnail grid
# ---------------------------------------------------------------------------


def test_selecting_a_folder_shows_the_thumbnail_grid(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selecting a folder switches the right pane to a grid of its photos."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    monkeypatch.setattr(window, "_start_thumbs", lambda _paths: None)  # no background thread
    _add_dir(window, {"a": a, "b": b})
    _select(window, window._tree.topLevelItem(0))  # noqa: SLF001 - the folder node

    assert window._right.currentIndex() == gui._PAGE_GRID  # noqa: SLF001
    assert window._grid.count() == 2  # noqa: SLF001, PLR2004 - two photos in the folder


def test_clicking_a_thumbnail_opens_the_detail(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Activating a grid item selects that photo and shows its detail page."""
    a = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=["Beach"])
    monkeypatch.setattr(window, "_start_thumbs", lambda _paths: None)
    _add_dir(window, {"a": a})
    _select(window, window._tree.topLevelItem(0))  # noqa: SLF001 - folder -> grid
    grid_item = window._grid.item(0)  # noqa: SLF001

    window._on_thumb_activated(grid_item)  # noqa: SLF001

    assert window._right.currentIndex() == gui._PAGE_DETAIL  # noqa: SLF001
    assert window._current is window._items[str(a)]  # noqa: SLF001


def test_thumbnail_worker_emits_bytes(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    """The worker decodes each path to bytes and stops when asked."""
    monkeypatch.setattr(
        gui,
        "prepare_image_for_agent",
        lambda *_a, **_k: SimpleNamespace(data=b"jpeg-bytes"),
    )
    emitted: list[tuple[str, bytes]] = []
    worker = gui.ThumbnailWorker([Path("/a.jpg"), Path("/b.jpg")])
    worker.ready.connect(lambda path, data: emitted.append((path, data)))
    worker.run()
    assert emitted == [("/a.jpg", b"jpeg-bytes"), ("/b.jpg", b"jpeg-bytes")]


def test_on_thumb_ready_sets_icon_and_caches(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A finished thumbnail updates the grid item and is cached."""
    a = _jpeg(tmp_path / "a.jpg")
    monkeypatch.setattr(window, "_start_thumbs", lambda _paths: None)
    _add_dir(window, {"a": a})
    _select(window, window._tree.topLevelItem(0))  # noqa: SLF001
    window._on_thumb_ready(str(a), a.read_bytes())  # noqa: SLF001
    assert str(a) in window._thumb_cache  # noqa: SLF001


# ---------------------------------------------------------------------------
# Generation worker
# ---------------------------------------------------------------------------


def _stub_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gui, "create_agent", lambda *_a, **_k: object())
    monkeypatch.setattr(
        gui,
        "read_image_context",
        lambda _p, **_kwargs: ImageContext(existing_keywords=KeywordSet(subject=["Beach"])),
    )
    monkeypatch.setattr(gui, "read_caption", lambda _p: ("Old", "Old caption."))
    monkeypatch.setattr(
        gui,
        "prepare_image_for_agent",
        lambda *_a, **_k: SimpleNamespace(data=b"x"),
    )
    monkeypatch.setattr(
        gui,
        "analyze_image_with_ai",
        lambda **_k: InferenceResult(title="T", description="D", keywords=["Eagle"]),
    )


def test_worker_emits_a_proposal_per_photo(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The worker reads context, runs the model, and emits one proposal per photo."""
    _stub_generation(monkeypatch)
    proposals: list[Proposal] = []
    finished: list[bool] = []
    worker = gui.GenerateWorker("lmstudio", "m", None, [Path("/a.jpg")])
    worker.file_done.connect(proposals.append)
    worker.finished.connect(lambda: finished.append(True))

    worker.run()

    assert len(proposals) == 1
    assert proposals[0].title == "T"
    assert proposals[0].keywords == ["Eagle"]
    assert finished == [True]


def test_worker_marks_all_failed_when_agent_cannot_build(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the agent cannot be created, every photo is reported as failed."""

    def boom(*_a: object, **_k: object) -> object:
        msg = "unreachable"
        raise ProviderError(msg)

    monkeypatch.setattr(gui, "create_agent", boom)
    failed: list[str] = []
    worker = gui.GenerateWorker("openai", "m", None, [Path("/a.jpg"), Path("/b.jpg")])
    worker.file_failed.connect(lambda path, _msg: failed.append(path))

    worker.run()

    assert failed == ["/a.jpg", "/b.jpg"]


def test_worker_reports_a_single_file_failure(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A per-photo error is reported without aborting the batch."""
    _stub_generation(monkeypatch)

    def boom(*_a: object, **_k: object) -> object:
        msg = "decode failed"
        raise ValueError(msg)

    monkeypatch.setattr(gui, "prepare_image_for_agent", boom)
    failed: list[str] = []
    worker = gui.GenerateWorker("lmstudio", "m", None, [Path("/a.jpg")])
    worker.file_failed.connect(lambda path, _msg: failed.append(path))

    worker.run()

    assert failed == ["/a.jpg"]


def test_generate_current_targets_only_the_open_photo(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate this photo runs for the open photo, ignoring which photos are checked."""
    a = _jpeg(tmp_path / "a.jpg")
    _jpeg(tmp_path / "b.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": a})  # both files added; all checked by default
    _select(window, window._leaf_for(a))  # noqa: SLF001 - open "a"

    captured: list[list[Path]] = []
    monkeypatch.setattr(
        window,
        "_run_generation",
        lambda items, **_kwargs: captured.append([i.path for i in items]),
    )
    window._generate_current()  # noqa: SLF001

    assert captured == [[a]]


def test_generate_targets_checked_photos(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate selected runs for the checked photos only."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    _add_dir(window, {"a": a, "b": b})
    window._items[str(b)].selected = False  # noqa: SLF001 - uncheck b

    captured: list[list[Path]] = []
    monkeypatch.setattr(
        window,
        "_run_generation",
        lambda items, **_kwargs: captured.append([i.path for i in items]),
    )
    window._generate()  # noqa: SLF001

    assert captured == [[a]]


def test_generate_current_needs_an_open_photo(window: gui.MainWindow) -> None:
    """With no photo open, Generate this photo nags instead of starting a run."""
    window._generate_current()  # noqa: SLF001
    assert window._thread is None  # noqa: SLF001 - no generation started
    assert "Open a photo" in window._status.text()  # noqa: SLF001


def test_worker_stops_after_current_photo_when_cancelled(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling stop() halts the batch at the next photo boundary, not mid-photo."""
    _stub_generation(monkeypatch)
    proposals: list[Proposal] = []
    worker = gui.GenerateWorker("lmstudio", "m", None, [Path("/a.jpg"), Path("/b.jpg")])

    def record_then_cancel(proposal: Proposal) -> None:
        # Cancel as soon as the first photo's proposal lands; the second photo must not run.
        proposals.append(proposal)
        worker.stop()

    worker.file_done.connect(record_then_cancel)

    worker.run()

    assert len(proposals) == 1


def test_cancel_generation_stops_worker_and_disables_button(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancel asks the worker to stop, flags the run as cancelling, and grays the button."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_generation(monkeypatch)
    _add_dir(window, {"a": img})
    window._run_generation([window._items[str(img)]])  # noqa: SLF001
    assert window._cancel_button.isEnabled()  # noqa: SLF001 - enabled while running

    window._cancel_generation()  # noqa: SLF001

    worker = window._worker  # noqa: SLF001
    assert worker is not None
    assert worker._stop is True  # noqa: SLF001
    assert window._cancelling is True  # noqa: SLF001
    assert not window._cancel_button.isEnabled()  # noqa: SLF001
    assert "Cancelling" in window._status.text()  # noqa: SLF001
    window._teardown_thread()  # noqa: SLF001 - join the worker thread the run started


def test_finish_after_cancel_resets_working_photos_to_pending(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """A cancelled run frees the un-started (still-WORKING) photos and reports the tally."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    _add_dir(window, {"a": a, "b": b})
    window._items[str(a)].status = WORKING  # noqa: SLF001 - never reached before cancel
    window._items[str(b)].status = READY  # noqa: SLF001 - already generated, must survive
    window._cancelling = True  # noqa: SLF001

    window._on_generate_finished()  # noqa: SLF001

    assert window._items[str(a)].status == PENDING  # noqa: SLF001
    assert window._items[str(b)].status == READY  # noqa: SLF001
    assert "Cancelled" in window._status.text()  # noqa: SLF001
    assert "1 photo not generated" in window._status.text()  # noqa: SLF001


def test_on_file_done_applies_proposal_to_item(window: gui.MainWindow, tmp_path: Path) -> None:
    """A finished proposal updates the matching item and its tree status."""
    img = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": img})
    proposal = Proposal(
        path=img,
        existing_title=None,
        existing_description=None,
        existing_keywords=KeywordSet(),
        title="Generated",
        description="Desc.",
        keywords=["Eagle"],
    )
    window._on_file_done(proposal)  # noqa: SLF001
    item = window._items[str(img)]  # noqa: SLF001
    assert item.status == READY
    assert item.title == "Generated"


# ---------------------------------------------------------------------------
# menu bar + telemetry toggle
# ---------------------------------------------------------------------------


def test_menu_bar_has_file_settings_help(window: gui.MainWindow) -> None:
    """The window carries a real menu bar with File, Settings, and Help menus."""
    titles = [action.text() for action in window.menuBar().actions()]
    assert "File" in titles
    assert "Settings" in titles
    assert "Help" in titles


def test_telemetry_toggle_defaults_on_and_persists(window: gui.MainWindow) -> None:
    """The Settings toggle reflects the default (on) and persists the choice when changed."""
    action = window._telemetry_action  # noqa: SLF001
    assert action.isCheckable()
    assert action.isChecked() is True  # default: telemetry on, no saved preference yet
    assert window._telemetry_enabled is True  # noqa: SLF001

    action.setChecked(False)  # user turns it off from the menu
    assert window._telemetry_enabled is False  # noqa: SLF001
    assert telemetry.read_gui_pref() is False  # the choice is persisted for next launch


def test_language_menu_lists_system_default_and_catalogs(window: gui.MainWindow) -> None:
    """The Settings > Language menu offers System Default plus every shipped language."""
    labels = [action.text() for action in window._language_menu.actions()]  # noqa: SLF001
    assert labels[0] == "System Default"
    assert "English" in labels
    assert "Português (Brasil)" in labels
    # No explicit choice saved: the system-default entry starts checked.
    assert window._language_menu.actions()[0].isChecked()  # noqa: SLF001


def test_language_choice_persists_to_the_config_file(
    window: gui.MainWindow,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Picking a language writes the config key; picking System Default removes it."""
    target = tmp_path / "config.toml"
    target.write_text('# keep me\nextensions = "jpg"\n', encoding="utf-8")
    monkeypatch.setattr(gui, "find_config_file", lambda: target)

    window._set_language("pt_BR")  # noqa: SLF001
    text = target.read_text(encoding="utf-8")
    assert 'language = "pt_BR"' in text
    assert "# keep me" in text
    assert "Restart" in window._status.text()  # noqa: SLF001

    window._set_language("auto")  # noqa: SLF001
    assert "language" not in target.read_text(encoding="utf-8")


def test_output_language_defaults_to_english_and_persists(
    window: gui.MainWindow,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The metadata language starts at English; a choice persists and English unpins it."""
    assert window._output_language == "English"  # noqa: SLF001
    assert window._output_language_combo.currentText() == "English"  # noqa: SLF001
    target = tmp_path / "config.toml"
    target.write_text('# keep me\nextensions = "jpg"\n', encoding="utf-8")
    monkeypatch.setattr(gui, "find_config_file", lambda: target)

    window._set_output_language("Brazilian Portuguese")  # noqa: SLF001
    text = target.read_text(encoding="utf-8")
    assert 'output_language = "Brazilian Portuguese"' in text
    assert "# keep me" in text
    assert window._output_language == "Brazilian Portuguese"  # noqa: SLF001
    assert "Brazilian Portuguese" in window._status.text()  # noqa: SLF001

    window._set_output_language("English")  # noqa: SLF001
    assert "output_language" not in target.read_text(encoding="utf-8")


def test_output_language_blank_means_the_default(
    window: gui.MainWindow,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Clearing the combo falls back to English instead of sending an empty language."""
    target = tmp_path / "config.toml"
    monkeypatch.setattr(gui, "find_config_file", lambda: target)
    window._set_output_language("German")  # noqa: SLF001
    window._set_output_language("   ")  # noqa: SLF001
    assert window._output_language == "English"  # noqa: SLF001
    assert "output_language" not in target.read_text(encoding="utf-8")


def test_telemetry_toggle_reflects_saved_off_preference(qapp: QApplication) -> None:
    """A window built after the user disabled telemetry comes up unchecked."""
    telemetry.write_gui_pref(enabled=False)
    win = gui.MainWindow()
    try:
        assert win._telemetry_enabled is False  # noqa: SLF001
        assert win._telemetry_action.isChecked() is False  # noqa: SLF001
    finally:
        win.close()


def test_close_flushes_telemetry_beacon(
    window: gui.MainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Closing the window must emit the beacon with block=True.

    The process exits right after closeEvent; a non-blocking send would ride a daemon thread that
    dies with the process, so the GUI beacon would be silently lost on every run.
    """
    calls: list[dict[str, object]] = []

    def fake_emit(
        run: telemetry.RunInfo,
        *,
        enabled: bool,
        block: bool = False,
    ) -> None:
        calls.append({"run": run, "enabled": enabled, "block": block})

    monkeypatch.setattr(gui.telemetry, "emit", fake_emit)
    window.close()

    assert len(calls) == 1
    assert calls[0]["block"] is True
    assert calls[0]["run"].interface == "gui"


# ---------------------------------------------------------------------------
# Redesigned chrome: progress bar, details disclosure, save options, menus
# ---------------------------------------------------------------------------


def test_progress_bar_tracks_the_run(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The progress bar appears for a run, ticks once per finished photo, and hides after."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_generation(monkeypatch)
    _add_dir(window, {"a": img})
    assert window._progress.isHidden()  # noqa: SLF001 - idle: no bar

    window._run_generation([window._items[str(img)]])  # noqa: SLF001
    assert not window._progress.isHidden()  # noqa: SLF001
    assert window._progress.maximum() == 1  # noqa: SLF001
    assert window._progress.value() == 0  # noqa: SLF001

    window._on_file_done(  # noqa: SLF001
        Proposal(
            path=img,
            existing_title=None,
            existing_description=None,
            existing_keywords=KeywordSet(),
            title="T",
            description="D",
            keywords=[],
        ),
    )
    assert window._progress.value() == 1  # noqa: SLF001

    window._teardown_thread()  # noqa: SLF001 - join the worker thread the run started
    assert window._progress.isHidden()  # noqa: SLF001


def test_details_disclosure_expands_and_collapses(window: gui.MainWindow) -> None:
    """The keyword-change details start collapsed and follow the disclosure toggle."""
    assert window._details_panel.isHidden()  # noqa: SLF001
    window._details_toggle.setChecked(True)  # noqa: SLF001
    assert not window._details_panel.isHidden()  # noqa: SLF001
    assert window._details_toggle.arrowType() == Qt.ArrowType.DownArrow  # noqa: SLF001
    window._details_toggle.setChecked(False)  # noqa: SLF001
    assert window._details_panel.isHidden()  # noqa: SLF001
    assert window._details_toggle.arrowType() == Qt.ArrowType.RightArrow  # noqa: SLF001


def test_details_toggle_summarizes_keyword_changes(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The collapsed disclosure still shows how many keywords a save would add or remove."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=["Beach"])
    _add_dir(window, {"a": img})
    _select(window, window._leaf_for(img))  # noqa: SLF001

    window._keywords.setPlainText("Beach\nEagle")  # noqa: SLF001 - one new keyword
    assert "+1" in window._details_toggle.text()  # noqa: SLF001

    window._keywords.setPlainText("Beach")  # noqa: SLF001 - back to the existing set
    assert "no change" in window._details_toggle.text()  # noqa: SLF001


def test_save_options_default_to_writing_all_fields(window: gui.MainWindow) -> None:
    """The Save options menu defaults match the CLI: write every field, merge, sidecar."""
    for action in (window._write_title, window._write_description, window._write_keywords):  # noqa: SLF001
        assert action.isCheckable()
        assert action.isChecked()
    assert not window._overwrite.isChecked()  # noqa: SLF001 - merge, not overwrite
    assert not window._embed.isChecked()  # noqa: SLF001 - sidecar, not embed


def test_file_menu_offers_csv_export(window: gui.MainWindow) -> None:
    """The CSV export moved off the toolbar and lives in the File menu."""
    texts = [action.text() for action in window._file_menu.actions()]  # noqa: SLF001
    assert "Export CSV Report..." in texts


def test_scan_options_hold_extensions_and_recursion(window: gui.MainWindow) -> None:
    """The Add menu's scan options carry the folder-scan settings with the GUI defaults."""
    assert window._extensions.text() == gui.DEFAULT_GUI_EXTENSIONS  # noqa: SLF001
    assert window._recursive.isChecked()  # noqa: SLF001


def test_retry_button_enabled_only_with_failures(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """Retry failed starts disabled, lights up on a failure, and resets with the list."""
    assert not window._retry_button.isEnabled()  # noqa: SLF001 - nothing failed yet
    img = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": img})
    assert not window._retry_button.isEnabled()  # noqa: SLF001

    window._on_file_failed(str(img), "boom")  # noqa: SLF001
    assert window._retry_button.isEnabled()  # noqa: SLF001

    window._clear()  # noqa: SLF001
    assert not window._retry_button.isEnabled()  # noqa: SLF001


def test_generate_menus_offer_one_time_skip_cache(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Generate split buttons' arrow menus run once without the cache."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": img})
    _select(window, window._leaf_for(img))  # noqa: SLF001 - so "this photo" has a target

    calls: list[tuple[int, bool]] = []
    monkeypatch.setattr(
        window,
        "_run_generation",
        lambda items, **kwargs: calls.append((len(items), kwargs.get("use_cache", True))),
    )
    window._generate_menu.actions()[0].trigger()  # noqa: SLF001
    window._generate_one_menu.actions()[0].trigger()  # noqa: SLF001

    assert calls == [(1, False), (1, False)]


def test_grid_thumbnails_share_the_context_menu(window: gui.MainWindow) -> None:
    """The folder grid asks for the same custom context menu as the tree."""
    policy = window._grid.contextMenuPolicy()  # noqa: SLF001
    assert policy == Qt.ContextMenuPolicy.CustomContextMenu


def test_help_menu_links_the_documentation(window: gui.MainWindow) -> None:
    """Help offers a Documentation entry pointing at the hosted docs."""
    texts = [action.text() for action in window._help_menu.actions()]  # noqa: SLF001
    assert "Documentation" in texts


def test_save_buttons_share_the_options_menu(window: gui.MainWindow) -> None:
    """Both Save buttons carry the same arrow menu holding the write toggles."""
    actions = window._save_options_menu.actions()  # noqa: SLF001
    assert window._write_title in actions  # noqa: SLF001
    assert window._overwrite in actions  # noqa: SLF001
    assert window._embed in actions  # noqa: SLF001


def test_save_tooltips_follow_the_chosen_options(window: gui.MainWindow) -> None:
    """Toggling a save option rewrites the Save buttons' current-options summary."""
    assert "title, description, keywords" in window._save_selected_button.toolTip()  # noqa: SLF001
    assert "XMP sidecar" in window._save_button.toolTip()  # noqa: SLF001

    window._write_description.setChecked(False)  # noqa: SLF001
    window._embed.setChecked(True)  # noqa: SLF001

    tip = window._save_selected_button.toolTip()  # noqa: SLF001
    assert "title, keywords" in tip
    assert "into the image file" in tip


def test_grid_checkbox_unchecks_photo_and_tree(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unchecking a thumbnail deselects the photo and repaints its tree row."""
    a = _jpeg(tmp_path / "a.jpg")
    _jpeg(tmp_path / "b.jpg")
    monkeypatch.setattr(window, "_start_thumbs", lambda _paths: None)
    _add_dir(window, {"a": a})
    _select(window, window._tree.topLevelItem(0))  # noqa: SLF001 - folder -> grid

    grid_item = window._grid_items[str(a)]  # noqa: SLF001
    assert grid_item.checkState() == Qt.CheckState.Checked
    grid_item.setCheckState(Qt.CheckState.Unchecked)  # fires itemChanged

    assert window._items[str(a)].selected is False  # noqa: SLF001
    assert _check_state(window, a) == Qt.CheckState.Unchecked
    # The parent folder is now partially checked.
    top = window._tree.topLevelItem(0)  # noqa: SLF001
    assert top is not None
    assert top.checkState(0) == Qt.CheckState.PartiallyChecked


def test_tree_uncheck_reflects_in_grid_checkboxes(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unchecking the folder in the tree unchecks every visible thumbnail."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    monkeypatch.setattr(window, "_start_thumbs", lambda _paths: None)
    _add_dir(window, {"a": a, "b": b})
    _select(window, window._tree.topLevelItem(0))  # noqa: SLF001 - folder -> grid

    top = window._tree.topLevelItem(0)  # noqa: SLF001
    assert top is not None
    top.setCheckState(0, Qt.CheckState.Unchecked)

    states = {item.checkState() for item in window._grid_items.values()}  # noqa: SLF001
    assert states == {Qt.CheckState.Unchecked}


def test_grid_checkbox_click_does_not_open_the_photo(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Toggling a thumbnail checkbox swallows the click; a later plain click still opens."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    monkeypatch.setattr(window, "_start_thumbs", lambda _paths: None)
    _add_dir(window, {"a": img})
    _select(window, window._tree.topLevelItem(0))  # noqa: SLF001 - folder -> grid
    grid_item = window._grid_items[str(img)]  # noqa: SLF001

    grid_item.setCheckState(Qt.CheckState.Unchecked)  # the toggle half of the click
    window._on_thumb_activated(grid_item)  # noqa: SLF001 - the click half

    assert window._right.currentIndex() == gui._PAGE_GRID  # noqa: SLF001 - stayed in the grid

    window._on_thumb_activated(grid_item)  # noqa: SLF001 - a later plain click
    assert window._right.currentIndex() == gui._PAGE_DETAIL  # noqa: SLF001


def test_modifier_click_on_thumbnail_keeps_the_grid(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shift/cmd click is building a selection, so it must not navigate away."""
    img = _jpeg(tmp_path / "a.jpg")
    monkeypatch.setattr(window, "_start_thumbs", lambda _paths: None)
    monkeypatch.setattr(gui, "_selection_modifiers_active", lambda: True)
    _add_dir(window, {"a": img})
    _select(window, window._tree.topLevelItem(0))  # noqa: SLF001 - folder -> grid

    window._on_thumb_activated(window._grid_items[str(img)])  # noqa: SLF001

    assert window._right.currentIndex() == gui._PAGE_GRID  # noqa: SLF001


def test_bulk_context_menu_checks_unchecks_and_removes(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """Right-clicking inside a multi-selection offers bulk check/uncheck/remove."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    c = _jpeg(tmp_path / "c.jpg")
    _add_dir(window, {"a": a, "b": b, "c": c})
    for path in (a, b):
        leaf = window._leaf_for(path)  # noqa: SLF001
        assert leaf is not None
        leaf.setSelected(True)

    leaf_a = window._leaf_for(a)  # noqa: SLF001
    menu = window._build_tree_context_menu(leaf_a)  # noqa: SLF001
    assert menu is not None
    actions = {action.text(): action for action in menu.actions() if action.text()}
    assert "Uncheck 2 Photos" in actions
    assert "Check 2 Photos" in actions

    actions["Uncheck 2 Photos"].trigger()
    assert window._items[str(a)].selected is False  # noqa: SLF001
    assert window._items[str(b)].selected is False  # noqa: SLF001
    assert window._items[str(c)].selected is True  # noqa: SLF001 - not in the selection
    top = window._tree.topLevelItem(0)  # noqa: SLF001
    assert top is not None
    assert top.checkState(0) == Qt.CheckState.PartiallyChecked

    actions["Remove From List"].trigger()
    assert str(a) not in window._items  # noqa: SLF001
    assert str(b) not in window._items  # noqa: SLF001
    assert str(c) in window._items  # noqa: SLF001


def test_bulk_menu_check_only_and_skip_cache(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check Only keeps just the selection checked; the bulk generate can skip the cache."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    c = _jpeg(tmp_path / "c.jpg")
    _add_dir(window, {"a": a, "b": b, "c": c})
    for path in (a, b):
        leaf = window._leaf_for(path)  # noqa: SLF001
        assert leaf is not None
        leaf.setSelected(True)

    menu = window._build_tree_context_menu(window._leaf_for(a))  # noqa: SLF001
    assert menu is not None
    actions = {action.text(): action for action in menu.actions() if action.text()}

    actions["Check Only 2 Photos"].trigger()
    assert window._items[str(a)].selected is True  # noqa: SLF001
    assert window._items[str(b)].selected is True  # noqa: SLF001
    assert window._items[str(c)].selected is False  # noqa: SLF001 - not in the selection

    calls: list[tuple[int, bool]] = []
    monkeypatch.setattr(
        window,
        "_run_generation",
        lambda items, **kwargs: calls.append((len(items), kwargs.get("use_cache", True))),
    )
    actions["Generate 2 Photos (Skip Cache)"].trigger()
    assert calls == [(2, False)]


def test_select_menu_inverts_the_checkboxes(window: gui.MainWindow, tmp_path: Path) -> None:
    """Invert Checked flips every checkbox, model and rendered tree alike."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    _add_dir(window, {"a": a, "b": b})
    window._items[str(a)].selected = False  # noqa: SLF001
    texts = [action.text() for action in window._select_menu.actions()]  # noqa: SLF001
    assert "Invert Checked" in texts

    window._invert_checked()  # noqa: SLF001

    assert window._items[str(a)].selected is True  # noqa: SLF001
    assert window._items[str(b)].selected is False  # noqa: SLF001
    assert _check_state(window, a) == Qt.CheckState.Checked
    assert _check_state(window, b) == Qt.CheckState.Unchecked


def test_single_row_right_click_keeps_the_per_photo_menu(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """With one row selected, the menu is the per-photo one (Generate, reveal, ...)."""
    a = _jpeg(tmp_path / "a.jpg")
    _jpeg(tmp_path / "b.jpg")
    _add_dir(window, {"a": a})
    leaf = window._leaf_for(a)  # noqa: SLF001
    assert leaf is not None
    window._tree.setCurrentItem(leaf)  # noqa: SLF001 - selects just this row

    menu = window._build_tree_context_menu(leaf)  # noqa: SLF001
    assert menu is not None
    texts = [action.text() for action in menu.actions() if action.text()]
    assert texts[0] == "Generate"


def test_remove_selected_drops_every_selected_row(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """Delete acts on the whole selection, not only the current row."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    c = _jpeg(tmp_path / "c.jpg")
    _add_dir(window, {"a": a, "b": b, "c": c})
    for path in (a, c):
        leaf = window._leaf_for(path)  # noqa: SLF001
        assert leaf is not None
        leaf.setSelected(True)

    window._remove_selected()  # noqa: SLF001

    assert sorted(window._items) == [str(b)]  # noqa: SLF001


def test_generating_status_pluralizes(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One photo reads "1 photo", several read "N photos" (no "(s)" anywhere)."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    _stub_generation(monkeypatch)
    _add_dir(window, {"a": a, "b": b})

    window._run_generation([window._items[str(a)]])  # noqa: SLF001
    assert window._status.text() == "Generating 1 photo..."  # noqa: SLF001
    window._teardown_thread()  # noqa: SLF001

    window._run_generation(  # noqa: SLF001
        [window._items[str(a)], window._items[str(b)]],  # noqa: SLF001
    )
    assert window._status.text() == "Generating 2 photos..."  # noqa: SLF001
    window._teardown_thread()  # noqa: SLF001


# ---------------------------------------------------------------------------
# Tree columns, colors, and context menu
# ---------------------------------------------------------------------------


def test_type_column_shows_extension_and_sidecar(window: gui.MainWindow, tmp_path: Path) -> None:
    """The Type column shows the extension, plus +xmp when a sidecar sits next to the file."""
    a = _jpeg(tmp_path / "a.jpg")
    b = _jpeg(tmp_path / "b.jpg")
    (tmp_path / "a.xmp").write_text("<x/>", encoding="utf-8")
    _add_dir(window, {"a": a, "b": b})
    leaf_a = window._leaf_for(a)  # noqa: SLF001
    leaf_b = window._leaf_for(b)  # noqa: SLF001
    assert leaf_a is not None
    assert leaf_b is not None
    assert leaf_a.text(1) == "jpg+xmp"
    assert leaf_b.text(1) == "jpg"


def test_scan_results_fill_the_tagged_column(window: gui.MainWindow, tmp_path: Path) -> None:
    """A finished metadata scan paints the Tagged column and records the fields."""
    img = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": img})
    leaf = window._leaf_for(img)  # noqa: SLF001
    assert leaf is not None
    assert leaf.text(3) == ""  # unknown until the scan reports

    window._on_scan_done({str(img): {FIELD_TITLE, FIELD_KEYWORDS}})  # noqa: SLF001

    assert leaf.text(3) == "TK"
    assert window._items[str(img)].known_fields == {FIELD_TITLE, FIELD_KEYWORDS}  # noqa: SLF001


def test_metadata_scan_worker_emits_string_keyed_presence(
    qapp: QApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scan worker batches one presence read and emits it keyed by path string."""
    monkeypatch.setattr(
        gui,
        "find_field_presence",
        lambda paths: {p: {FIELD_TITLE} for p in paths},
    )
    results: list[dict] = []
    worker = gui.MetadataScanWorker([Path("/a.jpg")])
    worker.done.connect(results.append)
    worker.run()
    assert results == [{"/a.jpg": {FIELD_TITLE}}]


def test_failed_status_is_painted_red(window: gui.MainWindow, tmp_path: Path) -> None:
    """A failed photo's Status cell turns red so it stands out in a long list."""
    img = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": img})
    window._on_file_failed(str(img), "boom")  # noqa: SLF001
    leaf = window._leaf_for(img)  # noqa: SLF001
    assert leaf is not None
    brush = leaf.data(2, Qt.ItemDataRole.ForegroundRole)
    assert brush is not None
    assert brush.color().name() == "#f85149"


def test_context_menu_on_failed_photo_offers_retry(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """Right-clicking a failed photo leads with Retry, plus reveal and remove actions."""
    import sys as sys_module  # noqa: PLC0415 - platform-dependent expected label.

    from photo_tagger.gui_state import reveal_label  # noqa: PLC0415

    img = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": img})
    window._items[str(img)].status = FAILED  # noqa: SLF001

    menu = window._build_tree_context_menu(window._leaf_for(img))  # noqa: SLF001
    assert menu is not None
    texts = [action.text() for action in menu.actions() if action.text()]

    assert texts[0] == "Retry Generation"
    assert "Generate (Skip Cache)" in texts
    assert reveal_label(sys_module.platform) in texts
    assert "Remove From List" in texts


def test_context_menu_on_folder_skips_generate_actions(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """A folder's context menu has no per-photo generate entries."""
    img = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": img})
    menu = window._build_tree_context_menu(window._tree.topLevelItem(0))  # noqa: SLF001
    assert menu is not None
    texts = [action.text() for action in menu.actions() if action.text()]
    assert all("Generate" not in text and "Retry" not in text for text in texts)
    assert "Remove From List" in texts


# ---------------------------------------------------------------------------
# Result cache
# ---------------------------------------------------------------------------


def test_worker_reuses_cached_results(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second run over the same photo hits the cache instead of calling the model."""
    _stub_generation(monkeypatch)
    img = _jpeg(tmp_path / "a.jpg")
    cache_file = tmp_path / "cache.sqlite"

    first = gui.GenerateWorker("lmstudio", "m", None, [img], cache_file=cache_file)
    done_first: list[Proposal] = []
    first.file_done.connect(done_first.append)
    first.run()
    assert len(done_first) == 1
    assert cache_file.exists()

    def boom(**_k: object) -> object:
        msg = "the model was called despite a cache hit"
        raise AssertionError(msg)

    monkeypatch.setattr(gui, "analyze_image_with_ai", boom)
    second = gui.GenerateWorker("lmstudio", "m", None, [img], cache_file=cache_file)
    done_second: list[Proposal] = []
    second.file_done.connect(done_second.append)
    second.run()

    assert len(done_second) == 1
    assert done_second[0].title == "T"


def test_worker_cache_survives_metadata_rewrites(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedding metadata rewrites the file, but the content-hash key still hits the cache."""
    _stub_generation(monkeypatch)
    monkeypatch.setattr(
        gui,
        "read_image_context",
        lambda _p, **_kwargs: ImageContext(
            existing_keywords=KeywordSet(),
            content_hash="pixels-unchanged",
        ),
    )
    img = _jpeg(tmp_path / "a.jpg")
    cache_file = tmp_path / "cache.sqlite"

    first = gui.GenerateWorker("lmstudio", "m", None, [img], cache_file=cache_file)
    first.file_done.connect(lambda _p: None)
    first.run()

    # Simulate "Embed in Photo": the file bytes change, the image data does not.
    img.write_bytes(img.read_bytes() + b"embedded-metadata")

    def boom(**_k: object) -> object:
        msg = "the model was called although the pixels did not change"
        raise AssertionError(msg)

    monkeypatch.setattr(gui, "analyze_image_with_ai", boom)
    second = gui.GenerateWorker("lmstudio", "m", None, [img], cache_file=cache_file)
    done: list[Proposal] = []
    second.file_done.connect(done.append)
    second.run()

    assert len(done) == 1
    assert done[0].from_cache is True


def test_cache_toggle_controls_the_active_cache_file(window: gui.MainWindow) -> None:
    """Caching defaults on; unchecking the Settings toggle disables it for new runs."""
    assert window._cache_action.isChecked()  # noqa: SLF001
    assert window._active_cache_file() is not None  # noqa: SLF001
    window._cache_action.setChecked(False)  # noqa: SLF001
    assert window._active_cache_file() is None  # noqa: SLF001


# ---------------------------------------------------------------------------
# Saving settings as the config file
# ---------------------------------------------------------------------------


def test_save_config_writes_the_gui_choices(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Save Settings as Defaults writes a TOML file load_defaults understands, sans API key."""
    import tomllib  # noqa: PLC0415 - test-local parser.

    target = tmp_path / "config.toml"
    # No config in effect, so the save creates the user config from the template.
    monkeypatch.setattr(gui, "find_config_file", lambda: None)
    monkeypatch.setattr(gui, "user_config_path", lambda: target)
    window._model.setCurrentText("qwen/qwen3-vl-30b")  # noqa: SLF001
    window._api_key.setText("sk-secret")  # noqa: SLF001 - must NOT be written
    window._extensions.setText("jpg,cr3")  # noqa: SLF001
    window._embed.setChecked(True)  # noqa: SLF001

    window._save_config()  # noqa: SLF001

    text = target.read_text(encoding="utf-8")
    assert "sk-secret" not in text
    data = tomllib.loads(text)
    assert data["provider"]["model_name"] == "qwen/qwen3-vl-30b"
    assert data["extensions"] == "jpg,cr3"
    assert data["output"]["use_sidecar"] is False
    assert data["telemetry"]["enabled"] is True


def test_save_config_merges_into_the_config_in_effect(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Saving over an existing config updates GUI keys but keeps comments and other tables."""
    import tomllib  # noqa: PLC0415 - test-local parser.

    target = tmp_path / "config.toml"
    target.write_text(
        '# hands off\nextensions = "cr3"\n\n[filter]\nskip_tagged = true\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(gui, "find_config_file", lambda: target)
    window._extensions.setText("jpg")  # noqa: SLF001

    window._save_config()  # noqa: SLF001

    text = target.read_text(encoding="utf-8")
    assert "# hands off" in text
    data = tomllib.loads(text)
    assert data["extensions"] == "jpg"
    assert data["filter"]["skip_tagged"] is True  # untouched table survives
    assert "preserved" in window._status.text()  # noqa: SLF001


def test_cached_proposal_shows_in_the_status_column(
    window: gui.MainWindow,
    tmp_path: Path,
) -> None:
    """A proposal replayed from the cache labels its row 'ready (cached)'."""
    img = _jpeg(tmp_path / "a.jpg")
    _add_dir(window, {"a": img})
    window._on_file_done(  # noqa: SLF001
        Proposal(
            path=img,
            existing_title=None,
            existing_description=None,
            existing_keywords=KeywordSet(),
            title="T",
            description="D",
            keywords=[],
            from_cache=True,
        ),
    )
    leaf = window._leaf_for(img)  # noqa: SLF001
    assert leaf is not None
    assert leaf.text(2) == "ready (cached)"


def test_description_boxes_grow_only_with_content(
    window: gui.MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A short description keeps its box short; a long one grows it up to the cap."""
    img = _jpeg(tmp_path / "a.jpg")
    _stub_reads(monkeypatch, keywords=[])
    _add_dir(window, {"a": img})
    _select(window, window._leaf_for(img))  # noqa: SLF001

    short = window._description.minimumHeight()  # noqa: SLF001 - setFixedHeight sets min=max
    window._description.setPlainText("line\n" * 40)  # noqa: SLF001

    grown = window._description.minimumHeight()  # noqa: SLF001
    assert grown > short
    assert grown <= 140  # noqa: PLR2004 - the documented cap
