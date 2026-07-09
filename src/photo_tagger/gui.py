# mypy: ignore-errors
"""
PySide6 desktop frontend for photo-tagger.

A review-before-write workflow over the same pipeline the CLI uses:

1. Drag photos or folders onto the window (or use *Add files/folder*). They appear in a
   checkable, nested tree on the left; uncheck or remove anything you do not want.
2. *Generate* runs the vision model on the checked photos on a background thread,
   streaming each proposal back through Qt signals (no pipeline code is Qt-aware).
3. Click a photo to see its preview, its existing title/description/keywords, and the
   proposed values in editable fields, then *Save* writes them with ExifTool.

Requires the optional ``[gui]`` extra (``pip install 'photo-tagger[gui]'``). The ``photo-tagger
gui`` command imports this module lazily, so the base CLI never depends on Qt. The Qt-free logic
lives in :mod:`photo_tagger.gui_state`; this file is the widget and event-loop shell and is excluded
from coverage and the static analyzers.
"""

import html
import os
import subprocess  # nosec B404 - only used to reveal a photo in the OS file browser
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

from loguru import logger
from PySide6.QtCore import QObject, QRect, QSize, Qt, QThread, QUrl, Signal
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QDesktopServices,
    QIcon,
    QKeySequence,
    QPainter,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QTreeWidgetItemIterator,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from photo_tagger import __version__, telemetry
from photo_tagger.ai import analyze_image_with_ai, create_agent
from photo_tagger.cache import InferenceCache, build_cache_namespace, hash_image_file
from photo_tagger.cli_options import load_defaults
from photo_tagger.config import (
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_JPEG_QUALITY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_USER_PROMPT,
)
from photo_tagger.config_file import find_config_file, load_config, user_config_path
from photo_tagger.csv_report import write_report
from photo_tagger.diagnostics import CheckResult, run_checks
from photo_tagger.discovery import load_skip_list, skip_list_matches
from photo_tagger.errors import DiscoveryError, PhotoTaggerError, ProviderError
from photo_tagger.gui_state import (
    ADDED,
    BADGE_FAILED,
    BADGE_METADATA,
    BADGE_SAVED,
    BADGE_SIDECAR,
    BADGE_UNSAVED,
    DEFAULT_GUI_EXTENSIONS,
    FAILED,
    PENDING,
    PROVIDER_LABELS,
    READY,
    REMOVED,
    SAVED,
    WORKING,
    FolderNode,
    GuiConfigValues,
    PhotoItem,
    Proposal,
    apply_proposal,
    build_tree,
    config_toml_text,
    count_generated,
    deselect_paths,
    ensure_path_dirs,
    expand_inputs,
    file_type_label,
    format_existing_keywords,
    hierarchy_preview,
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
    rank_vision_models,
    reveal_command,
    reveal_label,
    status_sort_rank,
    status_summary,
    tagged_summary,
    thumb_badges,
)
from photo_tagger.image_io import prepare_image_for_agent
from photo_tagger.logging_setup import setup_logging
from photo_tagger.metadata import (
    FIELD_DESCRIPTION,
    FIELD_KEYWORDS,
    FIELD_TITLE,
    build_contextual_prompt,
    find_field_presence,
    read_caption,
    read_image_context,
    read_metadata_sources,
    write_metadata,
)
from photo_tagger.models import KeywordSet
from photo_tagger.providers import PROVIDER_NAMES, ProviderName, get_backend


if TYPE_CHECKING:
    # Annotation-only on Python 3.14 (lazy), so no runtime import is needed.
    from PySide6.QtGui import QCloseEvent, QDragEnterEvent, QDropEvent


_RESOURCES = Path(__file__).parent / "resources"
# A stable, cwd-independent place for the GUI's logs. The CLI defaults to ./logs, but a windowed
# app has no meaningful working directory (it may be launched from Finder with cwd "/"), so the
# logs live under the user's home where the "Open logs" button can always find them.
_LOG_FOLDER = Path.home() / ".photo-tagger" / "logs"
_PREVIEW_MAX = 640
_THUMB_MAX = 200  # pixels for the grid thumbnails the model never sees
_THUMB_SIZE = 160  # icon box in the grid
_GENERATE_RETRIES = 2
_PATH_ROLE = Qt.ItemDataRole.UserRole
_IS_DIR_ROLE = Qt.ItemDataRole.UserRole + 1
_STATUS_RANK_ROLE = Qt.ItemDataRole.UserRole + 2  # lifecycle rank for sorting the Status column

# Tree columns, in display order.
_COL_NAME = 0
_COL_TYPE = 1
_COL_STATUS = 2
_COL_TAGGED = 3

# Status colors: failures pop red, saved photos settle green; other states use the palette.
_STATUS_COLOR = {FAILED: QColor("#f85149"), SAVED: QColor("#3fb950")}

# The GUI cache lives next to the GUI logs unless the config names a cache_file. Sharing the
# CLI's default would be wrong: the CLI has no default cache, it only caches when asked.
_DEFAULT_CACHE_FILE = Path.home() / ".photo-tagger" / "cache.sqlite"

_DOCS_URL = "https://jbsilva.github.io/photo-tagger/"
_PAGE_EMPTY = 0  # right-pane stack index for the idle "add or pick a photo" placeholder
_PAGE_DETAIL = 1  # right-pane stack index for one photo's detail
_PAGE_GRID = 2  # right-pane stack index for a folder's thumbnail grid
_DIR_MARK = "dir"  # truthy sentinel stored on folder tree items; files leave the role unset
_NONE = "(none)"  # placeholder shown when a photo has no existing title/description/keywords

# Right-pane placeholder copy. It adapts to the list: a getting-started nudge while empty, and a
# "pick a photo" nudge once photos are loaded but none is open. This is what fills the right pane
# when there is nothing to inspect, instead of an empty (and confusing) detail form.
_EMPTY_START = (
    "Add photos to get started.\n\n"
    "Drag photos or folders onto the window, or use Add files and Add folder."
)
_EMPTY_PICK = (
    "Select a photo to review it.\n\n"
    "Generate proposes a title, description, and keywords you can edit before saving."
)

# Short status word shown in the tree's second column.
_STATUS_LABEL = {
    PENDING: "",
    WORKING: "working...",
    READY: "ready",
    SAVED: "saved ✓",
    FAILED: "failed ✗",
}

# The field-aware "deselect already-tagged" menu, mirroring the CLI's --skip-tagged but letting
# the user pick which fields count as "done". Each entry is (menu label, required fields, whether
# ALL must be present, status-bar phrase). "Any metadata" is the broad OR criterion (the original
# skip-tagged); the rest require all of their fields, so a keyword-only photo survives "a title and
# a description" and stays selected for title/description generation.
_TAGGED_PRESETS: tuple[tuple[str, frozenset[str], bool, str], ...] = (
    (
        "Has any metadata",
        frozenset({FIELD_TITLE, FIELD_DESCRIPTION, FIELD_KEYWORDS}),
        False,
        "any metadata",
    ),
    ("Has a title", frozenset({FIELD_TITLE}), True, "a title"),
    ("Has a description", frozenset({FIELD_DESCRIPTION}), True, "a description"),
    (
        "Has a title and a description",
        frozenset({FIELD_TITLE, FIELD_DESCRIPTION}),
        True,
        "a title and a description",
    ),
    ("Has keywords", frozenset({FIELD_KEYWORDS}), True, "keywords"),
)

# Theme-agnostic polish: only spacing/rounding plus the brand accent on primary actions and
# the preview area. Colors for text and input backgrounds are left to the OS palette, so the
# window stays readable in both light and dark mode (hardcoding a light background would put
# the palette's light text on white in dark mode).
_STYLESHEET = """
QWidget { font-size: 13px; }
QPushButton {
    padding: 6px 12px; border-radius: 6px;
    border: 1px solid rgba(130, 130, 140, 60%);
    background: rgba(130, 130, 140, 14%);
}
QPushButton:hover { background: rgba(130, 130, 140, 26%); }
QPushButton:pressed { background: rgba(130, 130, 140, 36%); }
QPushButton:disabled { color: rgba(130, 130, 140, 70%); border-color: rgba(130, 130, 140, 25%); }
QPushButton#primary {
    background: #6366f1; color: white; border: 1px solid #6366f1; font-weight: 600;
}
QPushButton#primary:hover { background: #4f46e5; border-color: #4f46e5; }
QPushButton#primary:disabled {
    background: #9aa0e8; border-color: #9aa0e8; color: #eaeaff;
}
QLineEdit, QPlainTextEdit, QComboBox { padding: 4px 6px; border-radius: 5px; }
/* Scroll-area edits keep their native square frame unless given an explicit border, which is
   why only the QLineEdit fields looked rounded. */
QPlainTextEdit, QTextEdit {
    border: 1px solid rgba(130, 130, 140, 35%); border-radius: 5px; padding: 4px 6px;
}
QPlainTextEdit#tree { font-family: "SF Mono", Menlo, Consolas, monospace; }
QComboBox::drop-down {
    subcontrol-origin: padding; subcontrol-position: center right;
    width: 22px; border: none; background: transparent;
}
QComboBox::down-arrow { image: url("@CHEVRON@"); width: 10px; height: 6px; }
QPushButton::menu-indicator {
    image: url("@CHEVRON@"); width: 10px; height: 6px;
    subcontrol-origin: padding; subcontrol-position: center right; right: 10px;
}
QPushButton#menubutton { padding-right: 28px; }
QTreeWidget::item { padding: 2px; }
QToolButton { border: none; background: transparent; padding: 4px; font-weight: 600; }
QToolButton:hover { color: #6366f1; }
QToolButton#add, QToolButton#split {
    padding: 6px 26px 6px 12px; border-radius: 6px; font-weight: 400;
    border: 1px solid rgba(130, 130, 140, 60%);
    background: rgba(130, 130, 140, 14%);
}
QToolButton#add:hover, QToolButton#split:hover { background: rgba(130, 130, 140, 26%); }
QToolButton#split:disabled { color: rgba(130, 130, 140, 70%); }
QToolButton#add::menu-button, QToolButton#split::menu-button {
    border: none; width: 22px;
    border-left: 1px solid rgba(130, 130, 140, 45%);
    margin-top: 6px; margin-bottom: 6px;
}
QToolButton#add::menu-arrow, QToolButton#split::menu-arrow {
    image: url("@CHEVRON@"); width: 10px; height: 6px;
}
QToolButton#primarysplit {
    padding: 6px 26px 6px 12px; border-radius: 6px; font-weight: 600;
    background: #6366f1; color: white; border: 1px solid #6366f1;
}
QToolButton#primarysplit:hover { background: #4f46e5; border-color: #4f46e5; }
QToolButton#primarysplit:disabled {
    background: #9aa0e8; border-color: #9aa0e8; color: #eaeaff;
}
QToolButton#primarysplit::menu-button {
    border: none; width: 22px;
    border-left: 1px solid rgba(255, 255, 255, 40%);
    margin-top: 6px; margin-bottom: 6px;
}
QToolButton#primarysplit::menu-arrow {
    image: url("@CHEVRONLIGHT@"); width: 10px; height: 6px;
}
QProgressBar {
    border: 1px solid rgba(130, 130, 140, 60%); border-radius: 5px; text-align: center;
}
QProgressBar::chunk { background: #6366f1; border-radius: 4px; }
QLabel#preview { background: #1f1f24; border-radius: 8px; color: #9a9aa5; }
QLabel#hint, QLabel#status { color: #8a8a8a; }
QLabel#empty { color: #8a8a8a; font-size: 15px; }
QLabel#section { font-weight: 600; }
QLabel#error {
    background: rgba(248, 81, 73, 18%); color: #f85149;
    border: 1px solid rgba(248, 81, 73, 45%); border-radius: 6px; padding: 8px;
}
"""


def _stylesheet() -> str:
    """Resolve the stylesheet's image placeholders to the bundled resource files."""
    chevron = (_RESOURCES / "chevron-down.svg").as_posix()
    chevron_light = (_RESOURCES / "chevron-down-light.svg").as_posix()
    return _STYLESHEET.replace("@CHEVRONLIGHT@", chevron_light).replace("@CHEVRON@", chevron)


def _app_icon() -> QIcon:
    """Load the bundled app icon, or an empty icon if it is not present."""
    icon_path = _RESOURCES / "icon.svg"
    return QIcon(str(icon_path)) if icon_path.exists() else QIcon()


def _readonly_box(min_height: int) -> QPlainTextEdit:
    """Return a read-only, scrollable text box for displaying existing metadata."""
    box = QPlainTextEdit()
    box.setReadOnly(True)
    box.setMinimumHeight(min_height)
    return box


def _fit_text_height(box: QPlainTextEdit, *, min_h: int = 44, max_h: int = 140) -> None:
    """
    Size *box* to its content, within bounds.

    Short text keeps the box short so the pane's space goes to fields that need it; long text grows
    the box up to *max_h* and scrolls past that. QPlainTextEdit reports its document height in line
    counts, hence the line-spacing multiplication.
    """
    lines = max(1, int(box.document().size().height()))
    height = lines * box.fontMetrics().lineSpacing() + 14
    box.setFixedHeight(max(min_h, min(max_h, height)))


class GenerateWorker(QObject):
    """
    Generates AI proposals for a list of photos off the UI thread.

    For each photo it reads the existing metadata, builds the same contextual prompt as the CLI,
    runs the model, and emits a :class:`~photo_tagger.gui_state.Proposal`. All communication with
    the window is via Qt signals; widgets are never touched here.
    """

    started = Signal(int)
    file_done = Signal(object)  # Proposal
    file_failed = Signal(str, str)  # path, error message
    finished = Signal()

    def __init__(  # noqa: PLR0913  # one parameter per independent run input
        self,
        provider: ProviderName,
        model: str,
        api_base_url: str | None,
        paths: list[Path],
        api_key: str | None = None,
        cache_file: Path | None = None,
    ) -> None:
        """Store the run parameters; nothing happens until :meth:`run`."""
        super().__init__()
        self._provider = provider
        self._model = model
        self._api_base_url = api_base_url
        self._paths = paths
        self._api_key = api_key
        self._cache_file = cache_file
        self._stop = False

    def stop(self) -> None:
        """
        Ask the loop to stop before starting the next photo.

        The model call for the photo already in flight runs to completion (there is no way to
        interrupt a blocking request), so cancellation takes effect at the next photo boundary.
        """
        self._stop = True

    def run(self) -> None:
        """Build the agent once, then generate a proposal per photo."""
        try:
            agent = create_agent(
                self._provider,
                self._model,
                api_base_url=self._api_base_url,
                api_key=self._api_key,
                retries=_GENERATE_RETRIES,
            )
        except PhotoTaggerError as exc:
            for path in self._paths:
                self.file_failed.emit(str(path), str(exc))
            self.finished.emit()
            return

        cache = self._open_cache()
        self.started.emit(len(self._paths))
        try:
            for path in self._paths:
                if self._stop:
                    break
                try:
                    proposal = self._generate_one(agent, path, cache)
                except Exception as exc:  # noqa: BLE001
                    # One photo's failure must not stop the rest of the batch.
                    logger.exception("gui_generate_failed", file=path.name, error=str(exc))
                    self.file_failed.emit(str(path), str(exc))
                else:
                    self.file_done.emit(proposal)
        finally:
            if cache is not None:
                cache.close()
        self.finished.emit()

    def _open_cache(self) -> InferenceCache | None:
        """Open the result cache for this run, degrading to no cache on any failure."""
        if self._cache_file is None:
            return None
        try:
            return InferenceCache(self._cache_file, model_name=_gui_cache_namespace(self._model))
        except Exception as exc:  # noqa: BLE001
            # A broken cache (unwritable dir, corrupt file) must not block generation.
            logger.warning("gui_cache_open_failed", file=str(self._cache_file), error=str(exc))
            return None

    def _generate_one(self, agent: object, path: Path, cache: InferenceCache | None) -> Proposal:
        """Read existing metadata, run the model (or hit the cache), and assemble a proposal."""
        context = read_image_context(path)
        existing_title, existing_description = read_caption(path)
        gps_info = {"position": context.gps_position} if context.gps_position else {}
        prompt = build_contextual_prompt(
            DEFAULT_USER_PROMPT,
            context.existing_keywords.subject,
            context.location_tags,
            gps_info,
            camera_info=context.camera_info,
        )
        image_hash = hash_image_file(path) if cache is not None else ""
        cached = cache.get(image_hash) if cache is not None else None
        inference = cached
        if inference is None:
            jpeg = prepare_image_for_agent(path, max_size=_PREVIEW_MAX)
            inference = analyze_image_with_ai(image_bytes=jpeg, agent=agent, user_prompt=prompt)
            if cache is not None:
                cache.put(image_hash, inference)
        else:
            logger.info("gui_cache_hit", file=path.name)
        return Proposal(
            path=path,
            existing_title=existing_title,
            existing_description=existing_description,
            existing_keywords=context.existing_keywords,
            title=inference.title,
            description=inference.description,
            keywords=list(inference.keywords),
            camera_info=dict(context.camera_info),
            location_tags=dict(context.location_tags),
            gps_position=context.gps_position,
            from_cache=cached is not None,
            input_tokens=inference.input_tokens,
            output_tokens=inference.output_tokens,
            total_tokens=inference.total_tokens,
            seconds=inference.seconds,
        )


class ThumbnailWorker(QObject):
    """
    Decodes grid thumbnails off the UI thread.

    Each thumbnail is a small JPEG (the same RAW-aware loader the model uses, at a tiny size). The
    bytes are emitted back to the main thread, which builds the QPixmap there (QPixmap must not be
    created off the GUI thread). :meth:`stop` lets the window abandon a folder's load when the user
    navigates away.
    """

    ready = Signal(str, bytes)  # path, JPEG bytes
    finished = Signal()

    def __init__(self, paths: list[Path]) -> None:
        """Store the paths to decode; nothing runs until :meth:`run`."""
        super().__init__()
        self._paths = paths
        self._stop = False

    def stop(self) -> None:
        """Ask the loop to stop before the next thumbnail."""
        self._stop = True

    def run(self) -> None:
        """Decode each thumbnail and emit its bytes, until done or stopped."""
        for path in self._paths:
            if self._stop:
                break
            try:
                content = prepare_image_for_agent(path, max_size=_THUMB_MAX)
            except Exception as exc:  # noqa: BLE001
                logger.warning("gui_thumbnail_failed", file=path.name, error=str(exc))
                continue
            self.ready.emit(str(path), bytes(content.data))
        self.finished.emit()


class MetadataScanWorker(QObject):
    """
    Reads which metadata fields each photo already carries, off the UI thread.

    One batched exiftool call covers the whole list, the same read the "Uncheck Already Tagged"
    action does synchronously. The result feeds the tree's Tagged column so the user can see at a
    glance which photos are already done. Failures degrade to an empty report.
    """

    done = Signal(object)  # dict[str, set[str]]: path -> present indicator fields
    finished = Signal()

    def __init__(self, paths: list[Path]) -> None:
        """Store the paths to scan; nothing runs until :meth:`run`."""
        super().__init__()
        self._paths = paths

    def run(self) -> None:
        """Scan all paths in one batched read and emit the per-path field sets."""
        try:
            presence = find_field_presence(self._paths)
        except Exception as exc:  # noqa: BLE001
            # The scan is a convenience; a broken exiftool must not take the window down.
            logger.warning("gui_metadata_scan_failed", error=str(exc))
            presence = {}
        self.done.emit({str(path): set(fields) for path, fields in presence.items()})
        self.finished.emit()


def _gui_cache_namespace(model: str) -> str:
    """
    Build the cache namespace for GUI runs: the model plus the GUI's fixed inference settings.

    The GUI runs the agent with the library defaults and sends images at ``_PREVIEW_MAX``, so those
    values (not the CLI flags) are what key its cache entries.
    """
    return build_cache_namespace(
        model,
        user_prompt=DEFAULT_USER_PROMPT,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
        frequency_penalty=DEFAULT_FREQUENCY_PENALTY,
        jpeg_dimensions=_PREVIEW_MAX,
        jpeg_quality=DEFAULT_JPEG_QUALITY,
    )


def _status_sort_key(item: QTreeWidgetItem) -> tuple[int, str]:
    """Sort key for the Status column: lifecycle rank first, then name as a stable tiebreak."""
    rank = item.data(_COL_STATUS, _STATUS_RANK_ROLE)
    return (int(rank) if rank is not None else 0, item.text(_COL_NAME).casefold())


class _SortableTreeItem(QTreeWidgetItem):  # NOSONAR S8500 - Qt sorts items via __lt__ only
    """
    A tree row that sorts sensibly when the user clicks a column header.

    Folders stay grouped above files whichever way the sort runs; the Photos column sorts by name
    (case-insensitive) and the Status column by lifecycle rank rather than the raw label.

    Only ``__lt__`` is overridden: Qt drives item sorting entirely through it, and the comparison is
    context-dependent (it follows the active sort column and direction), so a real total ordering or
    ``functools.total_ordering`` would be wrong here. Hence the S8500 suppression on the class.
    """

    def __lt__(self, other: QTreeWidgetItem) -> bool:
        tree = self.treeWidget()
        column = tree.sortColumn() if tree is not None else 0
        self_dir = bool(self.data(0, _IS_DIR_ROLE))
        if self_dir != bool(other.data(0, _IS_DIR_ROLE)):
            # Keep folders above files in both directions: Qt reverses the result for a
            # descending sort, so invert there to cancel that out.
            ascending = (
                tree is None or tree.header().sortIndicatorOrder() == Qt.SortOrder.AscendingOrder
            )
            return self_dir if ascending else not self_dir
        if column == _COL_STATUS:
            return _status_sort_key(self) < _status_sort_key(other)
        if column in (_COL_TYPE, _COL_TAGGED):
            return self.text(column).casefold() < other.text(column).casefold()
        return self.text(_COL_NAME).casefold() < other.text(_COL_NAME).casefold()


class MainWindow(QMainWindow):
    """
    The main window, laid out along the add -> generate -> review -> save workflow.

    A one-row header picks the provider and model (URL/key live in a Connection dialog), the left
    panel holds the checkable file tree with its add/select controls, the right pane reviews one
    photo (or a folder grid), and the bottom bar carries the batch actions with a progress bar.
    """

    def __init__(self) -> None:
        """Build the widgets, enable drag-and-drop, and wire the actions."""
        super().__init__()
        self._defaults = load_defaults()
        self._items: dict[str, PhotoItem] = {}
        self._preview_cache: dict[str, QPixmap] = {}
        self._thumb_cache: dict[str, QPixmap] = {}
        self._grid_items: dict[str, QListWidgetItem] = {}
        self._current: PhotoItem | None = None
        self._thread: QThread | None = None
        self._worker: GenerateWorker | None = None
        self._cancelling = False
        self._thumb_thread: QThread | None = None
        self._thumb_worker: ThumbnailWorker | None = None
        self._scan_thread: QThread | None = None
        self._scan_worker: MetadataScanWorker | None = None
        self._syncing = False
        # Raw config, for keys the Defaults dataclass fills with CLI-oriented values: the GUI
        # wants its own broad extension default unless the user actually saved one.
        self._raw_config = load_config()
        self._cache_file = self._defaults.artifacts.cache_file or _DEFAULT_CACHE_FILE
        # Wall-clock start of this GUI session, reported as the run duration on close.
        self._session_start = time.monotonic()
        # Telemetry on/off: a persisted Settings-menu choice wins over the config-file default.
        _pref = telemetry.read_gui_pref()
        self._telemetry_enabled = self._defaults.telemetry.enabled if _pref is None else _pref

        self.setWindowTitle(f"Photo Tagger {__version__}")
        self.setWindowIcon(_app_icon())
        self.resize(1180, 760)
        self.setAcceptDrops(True)
        self._placeholder_pixmap = _make_placeholder()

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.addLayout(self._build_header())
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_tree_panel())
        splitter.addWidget(self._build_right_pane())
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        # Start with the left pane wide enough for all four tree columns.
        splitter.setSizes([500, 680])
        layout.addWidget(splitter, stretch=1)
        layout.addLayout(self._build_bottom_bar())
        self._build_menus()
        self._show_empty()

    # --- construction ----------------------------------------------------------------------

    def _build_menus(self) -> None:
        """Build the menu bar: File actions, a Settings telemetry toggle, and Help."""
        menubar = self.menuBar()

        # Kept on self: QAction.menu() hands out a transient wrapper that shiboken may delete,
        # so tests (and future code) need a stable reference to the menu itself.
        file_menu = self._file_menu = menubar.addMenu("File")
        file_menu.addAction("Add Photos...", self._choose_files)
        file_menu.addAction("Add Folder...", self._choose_folder)
        file_menu.addSeparator()
        export_action = file_menu.addAction("Export CSV Report...", self._export_csv)
        export_action.setToolTip(
            "Save a CSV report of every photo: generated and existing metadata, EXIF, and "
            "token usage.",
        )
        file_menu.addSeparator()
        file_menu.addAction("Clear List", self._clear)
        file_menu.addSeparator()
        quit_action = file_menu.addAction("Quit", self.close)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.setMenuRole(QAction.MenuRole.QuitRole)

        settings_menu = menubar.addMenu("Settings")
        settings_menu.setToolTipsVisible(True)
        self._cache_action = QAction("Cache AI Results", self)
        self._cache_action.setCheckable(True)
        self._cache_action.setChecked(True)
        self._cache_action.setToolTip(
            f"Reuse earlier results for unchanged photos ({self._cache_file}). Uncheck to call "
            "the model again for everything; a single photo can skip the cache from its "
            "right-click menu.",
        )
        settings_menu.addAction(self._cache_action)
        self._telemetry_action = QAction("Send Anonymous Telemetry", self)
        self._telemetry_action.setCheckable(True)
        self._telemetry_action.setChecked(self._telemetry_enabled)
        self._telemetry_action.setToolTip(
            "Anonymous usage stats (model, batch size, OS, CPU arch, timing). No photos or "
            "personal data.",
        )
        self._telemetry_action.toggled.connect(self._on_telemetry_toggled)
        settings_menu.addAction(self._telemetry_action)
        settings_menu.addSeparator()
        save_defaults = settings_menu.addAction(
            "Save Settings as Defaults...",
            self._save_config,
        )
        save_defaults.setToolTip(
            "Update the config file with the current provider, model, URL, file types, and save "
            "options. Other settings and comments in the file are preserved; the API key is "
            "never written.",
        )
        edit_config = settings_menu.addAction("Edit Config File...", self._edit_config)
        edit_config.setToolTip(
            "Open the config file in your default editor for the settings the GUI does not "
            "surface (prompt file, sampling, workers, filters, ...). Created if missing.",
        )

        help_menu = self._help_menu = menubar.addMenu("Help")
        help_menu.addAction(
            "Documentation",
            lambda: QDesktopServices.openUrl(QUrl(_DOCS_URL)),
        )
        help_menu.addSeparator()
        help_menu.addAction("Test Connection", self._test_connection)
        help_menu.addAction("Open Logs", self._open_logs)
        help_menu.addSeparator()
        about_action = help_menu.addAction("About Photo Tagger", self._show_about)
        about_action.setMenuRole(QAction.MenuRole.AboutRole)

    def _on_telemetry_toggled(self, enabled: bool) -> None:  # noqa: FBT001 - Qt toggled(bool) slot.
        """Persist the telemetry choice and apply it to this session right away."""
        self._telemetry_enabled = enabled
        telemetry.write_gui_pref(enabled=enabled)
        self._status.setText("Anonymous telemetry on." if enabled else "Anonymous telemetry off.")

    def _active_cache_file(self) -> Path | None:
        """Return the cache file generation should use, or None when caching is toggled off."""
        return self._cache_file if self._cache_action.isChecked() else None

    def _current_config_values(self) -> GuiConfigValues:
        """Collect the GUI's current choices that persist to the config file (never the API key)."""
        return GuiConfigValues(
            provider_name=self._provider_name(),
            model_name=self._model.currentText().strip(),
            api_base_url=self._url.text().strip() or None,
            extensions=self._extensions.text().strip(),
            recursive=self._recursive.isChecked(),
            write_title=self._write_title.isChecked(),
            write_description=self._write_description.isChecked(),
            write_keywords=self._write_keywords.isChecked(),
            preserve_keywords=not self._overwrite.isChecked(),
            use_sidecar=not self._embed.isChecked(),
            telemetry_enabled=self._telemetry_enabled,
        )

    def _config_target(self) -> Path:
        """Return the config file to save into: the one in effect, or the user default path."""
        return find_config_file() or user_config_path()

    def _save_config(self) -> None:
        """
        Persist the GUI's current choices into the config file.

        An existing file is merged, not replaced: only the GUI-managed keys change, and comments,
        ordering, and every other setting survive. A missing file is created from a template.
        """
        target = self._config_target()
        values = self._current_config_values()
        try:
            if target.exists():
                text = merged_config_text(target.read_text(encoding="utf-8"), values)
                note = f"Updated {target} (other settings and comments preserved)."
            else:
                text = config_toml_text(values)
                note = f"Saved defaults to {target}."
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "Could not save the config file", str(exc))
            return
        self._status.setText(note)

    def _edit_config(self) -> None:
        """Open the config file in the user's editor, creating it first if it does not exist."""
        target = self._config_target()
        if not target.exists():
            self._save_config()
        if target.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def _show_about(self) -> None:
        """Show a small About dialog with the version and project link."""
        QMessageBox.about(
            self,
            "About Photo Tagger",
            f"<b>Photo Tagger {__version__}</b><br><br>"
            "Describe photos and add keywords with a vision-language model.<br><br>"
            '<a href="https://github.com/jbsilva/photo-tagger">github.com/jbsilva/photo-tagger</a>',
        )

    def _build_header(self) -> QHBoxLayout:
        """One compact row: the model choice the user changes often, the rest behind Connection."""
        provider = self._defaults.provider

        self._provider = QComboBox()
        for name in PROVIDER_NAMES:
            self._provider.addItem(PROVIDER_LABELS.get(name, name), name)
        self._provider.setCurrentIndex(max(0, list(PROVIDER_NAMES).index(provider.provider_name)))
        # Size to the widest label so "LM Studio" is not clipped.
        self._provider.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._provider.setMinimumContentsLength(10)
        self._provider.setToolTip("Backend that serves the vision-language model.")

        self._model = QComboBox()
        self._model.setEditable(True)
        self._model.setMinimumWidth(260)
        self._model.setCurrentText(provider.model_name)
        self._model.setToolTip(
            "Model identifier. Type it, or press Refresh to list what the provider serves.",
        )
        refresh = QPushButton("Refresh")
        refresh.setToolTip("Query the provider for the models it currently serves.")
        refresh.clicked.connect(self._refresh_models)

        self._connection_dialog = self._build_connection_dialog()
        connection = QPushButton("Connection...")
        connection.setToolTip("Server URL, API key, and a connection test.")
        connection.clicked.connect(self._connection_dialog.exec)

        row = QHBoxLayout()
        row.addWidget(QLabel("Provider"))
        row.addWidget(self._provider)
        row.addWidget(QLabel("Model"))
        row.addWidget(self._model, stretch=1)
        row.addWidget(refresh)
        row.addWidget(connection)
        return row

    def _build_connection_dialog(self) -> QDialog:
        """URL, API key, and the connection test: set-once settings, out of the main window."""
        provider = self._defaults.provider
        dialog = QDialog(self)
        dialog.setWindowTitle("Connection settings")
        dialog.setMinimumWidth(520)
        form = QFormLayout(dialog)
        # macOS style defaults to fixed-size fields; let them fill the dialog width instead.
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._url = QLineEdit(provider.api_base_url or "")
        self._url.setMinimumWidth(380)
        self._url.setPlaceholderText("(provider default URL)")
        self._url.setToolTip("Provider API base URL. Leave blank to use the provider's default.")
        form.addRow("Base URL", self._url)

        # Pre-filled from a config-file key if one is set, never from an environment variable: an
        # env key stays in the environment and is resolved at call time, so it never lands in the
        # widget. A typed key is masked, used only for this session, and never written to disk.
        self._api_key = QLineEdit(provider.api_key or "")
        self._api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key.setClearButtonEnabled(True)
        self._api_key.setMinimumWidth(380)
        self._api_key.setPlaceholderText("(uses provider env var)")
        self._api_key.setToolTip(
            "API key for the provider. Leave blank to use the provider's environment variable "
            "(OPENAI_API_KEY, LM_STUDIO_API_KEY, LLAMA_CPP_API_KEY, or OLLAMA_API_KEY). Required "
            "for OpenAI. A typed key is used for this session only and is never written to disk.",
        )
        form.addRow("API key", self._api_key)

        self._test_button = QPushButton("Test connection")
        self._test_button.setToolTip("Check ExifTool and that the provider serves the model.")
        self._test_button.clicked.connect(self._test_connection)
        close = QPushButton("Close")
        close.setDefault(True)
        close.clicked.connect(dialog.accept)
        buttons = QHBoxLayout()
        buttons.addWidget(self._test_button)
        buttons.addStretch(1)
        buttons.addWidget(close)
        form.addRow(buttons)
        return dialog

    def _build_tree_panel(self) -> QWidget:
        panel = QWidget()
        box = QVBoxLayout(panel)
        box.addLayout(self._build_tree_controls())

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Photos", "Type", "Status", "Tagged"])
        self._tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        header = self._tree.header()
        header.setStretchLastSection(False)
        for column, width in (
            (_COL_NAME, 250),
            (_COL_TYPE, 64),
            (_COL_STATUS, 80),
            (_COL_TAGGED, 56),
        ):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Interactive)
            self._tree.setColumnWidth(column, width)
        # Click a header to sort by any column; folders stay grouped above files.
        self._tree.setSortingEnabled(True)
        self._tree.sortByColumn(_COL_NAME, Qt.SortOrder.AscendingOrder)
        header.setToolTip(
            "Click a column header to sort. Type: file extension, +xmp when a sidecar exists.\n"
            "Tagged: metadata already on the file (T title, D description, K keywords).",
        )
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._on_tree_context_menu)
        self._tree.itemChanged.connect(self._on_item_changed)
        self._tree.currentItemChanged.connect(self._on_current_changed)
        for key in (QKeySequence.StandardKey.Delete, QKeySequence(Qt.Key.Key_Backspace)):
            shortcut = QShortcut(key, self._tree)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(self._remove_selected)
        box.addWidget(self._tree, stretch=1)

        hint = QLabel("Drag photos or folders here. Select one and press Delete to remove it.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        box.addWidget(hint)
        return panel

    def _build_tree_controls(self) -> QHBoxLayout:
        controls = QHBoxLayout()

        # One split button: a click adds files; the arrow offers the folder dialog and the scan
        # options. Native file dialogs cannot select files and folders at once, so the split is
        # the closest single-control equivalent (drag-and-drop takes both anyway).
        add = QToolButton()
        add.setObjectName("add")
        add.setText("Add photos...")
        add.setToolTip(
            "Add photos (click), or open the arrow for adding a whole folder and for the "
            "folder-scan options. Dragging files or folders onto the window also works.",
        )
        add.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        add.clicked.connect(self._choose_files)
        add.setMenu(self._build_add_menu())
        controls.addWidget(add)
        controls.addStretch(1)

        select = QPushButton("Select")
        select.setObjectName("menubutton")
        select.setToolTip("Check or uncheck photos in bulk.")
        select.setMenu(self._build_select_menu())
        controls.addWidget(select)

        remove = QPushButton("Remove")
        remove.setToolTip("Remove the selected folder or photo from the list (or press Delete).")
        remove.clicked.connect(self._remove_selected)
        controls.addWidget(remove)
        return controls

    def _build_add_menu(self) -> QMenu:
        """Build the Add button's arrow menu: the folder dialog plus the folder-scan settings."""
        menu = QMenu(self)
        menu.addAction("Add Folder...", self._choose_folder)
        menu.addSeparator()
        panel = QWidget()
        form = QFormLayout(panel)
        self._extensions = QLineEdit(self._raw_config.get("extensions", DEFAULT_GUI_EXTENSIONS))
        self._extensions.setMinimumWidth(280)
        self._extensions.setToolTip(
            "Extensions to scan for in folders (comma-separated).\n"
            "Case-insensitive: jpg matches .JPG. Note jpeg is separate from jpg.",
        )
        form.addRow("File types", self._extensions)
        self._recursive = QCheckBox("Include subfolders")
        self._recursive.setChecked(bool(self._raw_config.get("recursive", True)))
        self._recursive.setToolTip("Descend into subfolders when adding a folder.")
        form.addRow("", self._recursive)
        host = QWidgetAction(menu)
        host.setDefaultWidget(panel)
        menu.addAction(host)
        return menu

    def _build_select_menu(self) -> QMenu:
        """Bulk check/uncheck actions, including the CLI's --skip-tagged/--skip-from mirrors."""
        menu = QMenu(self)
        menu.addAction("Check All", lambda: self._set_all_checked(checked=True))
        menu.addAction("Uncheck All", lambda: self._set_all_checked(checked=False))
        menu.addSeparator()

        self._tagged_menu = menu.addMenu("Uncheck Already Tagged")
        self._tagged_menu.setToolTip(
            "Uncheck photos that already have the chosen metadata (in the image or its XMP "
            "sidecar), e.g. 'a title and a description' to skip those while keeping "
            "keyword-only photos. Mirrors the CLI's --skip-tagged.",
        )
        for text, required, match_all, phrase in _TAGGED_PRESETS:
            action = self._tagged_menu.addAction(text)
            action.triggered.connect(
                lambda _checked=False, req=required, all_=match_all, ph=phrase: (
                    self._deselect_tagged(
                        req,
                        match_all=all_,
                        phrase=ph,
                    )
                ),
            )

        from_file = menu.addAction("Uncheck From Skip List...", self._deselect_from_file)
        from_file.setToolTip(
            "Uncheck photos whose filename or full path is listed in a text file (one per "
            "line), like the CLI's --skip-from.",
        )
        return menu

    def _set_all_checked(self, *, checked: bool) -> None:
        """Check or uncheck every photo at once."""
        if not self._items:
            self._status.setText("Add photos before selecting.")
            return
        for item in self._items.values():
            item.selected = checked
        self._rebuild_tree()
        self._update_status()

    # --- tree context menu -------------------------------------------------------------------

    def _on_tree_context_menu(self, pos: object) -> None:
        """Show the per-row context menu for the tree item under the cursor."""
        tree_item = self._tree.itemAt(pos)
        menu = self._build_tree_context_menu(tree_item)
        if menu is not None:
            menu.exec(self._tree.viewport().mapToGlobal(pos))

    def _build_tree_context_menu(self, tree_item: QTreeWidgetItem | None) -> QMenu | None:
        """Build the context menu for *tree_item* (a file leaf or a folder), or None."""
        if tree_item is None:
            return None
        path = tree_item.data(0, _PATH_ROLE)
        if path is None:
            return None
        menu = QMenu(self._tree)
        item = self._items.get(path)
        if not bool(tree_item.data(0, _IS_DIR_ROLE)) and item is not None:
            label = "Retry Generation" if item.status == FAILED else "Generate"
            generate = menu.addAction(label, lambda: self._run_generation([item]))
            generate.setEnabled(self._thread is None)
            fresh = menu.addAction(
                "Generate (Skip Cache)",
                lambda: self._run_generation([item], use_cache=False),
            )
            fresh.setToolTip("Call the model even when a cached result exists for this photo.")
            fresh.setEnabled(self._thread is None)
            menu.addSeparator()
        menu.addAction(reveal_label(sys.platform), lambda: self._reveal(Path(path)))
        menu.addSeparator()
        remove = menu.addAction("Remove From List")
        remove.triggered.connect(
            lambda: (self._tree.setCurrentItem(tree_item), self._remove_selected()),
        )
        return menu

    def _on_grid_context_menu(self, pos: object) -> None:
        """Offer a thumbnail the same right-click actions as its row in the tree."""
        grid_item = self._grid.itemAt(pos)
        if grid_item is None:
            return
        menu = self._build_tree_context_menu(self._leaf_for(Path(grid_item.data(_PATH_ROLE))))
        if menu is not None:
            menu.exec(self._grid.viewport().mapToGlobal(pos))

    def _reveal(self, path: Path) -> None:
        """Show *path* selected in the OS file browser, or open its folder where unsupported."""
        argv = reveal_command(path, sys.platform)
        if argv is None:
            folder = path if path.is_dir() else path.parent
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
            return
        subprocess.Popen(argv)  # noqa: S603  # nosec B603 - fixed reveal argv, path from our list

    # --- background metadata scan (the Tagged column) ----------------------------------------

    def _start_metadata_scan(self) -> None:
        """Scan photos with an unknown Tagged state in the background, one batch at a time."""
        if self._scan_thread is not None:
            return  # a scan is running; _on_scan_finished re-checks for stragglers
        pending = [item.path for item in self._items.values() if item.known_fields is None]
        if not pending:
            return
        self._scan_thread = QThread(self)
        self._scan_worker = MetadataScanWorker(pending)
        self._scan_worker.moveToThread(self._scan_thread)
        self._scan_thread.started.connect(self._scan_worker.run)
        self._scan_worker.done.connect(self._on_scan_done)
        self._scan_worker.finished.connect(self._on_scan_finished)
        self._scan_thread.start()

    def _on_scan_done(self, presence: dict[str, set[str]]) -> None:
        """Record the scanned field sets and repaint the Tagged column."""
        for key, fields in presence.items():
            item = self._items.get(key)
            if item is not None:
                item.known_fields = set(fields)
                self._refresh_status_cell(item)

    def _on_scan_finished(self) -> None:
        """Tear down the scan thread and pick up photos added while it ran."""
        self._stop_scan()
        self._start_metadata_scan()

    def _stop_scan(self) -> None:
        if self._scan_thread is not None:
            self._scan_thread.quit()
            self._scan_thread.wait()
            self._scan_thread = None
        self._scan_worker = None

    def _build_right_pane(self) -> QWidget:
        """Build a stack showing the idle placeholder, one photo's detail, or a folder's grid."""
        self._right = QStackedWidget()
        self._right.addWidget(self._build_empty_state())  # _PAGE_EMPTY
        self._right.addWidget(self._build_detail_panel())  # _PAGE_DETAIL
        self._right.addWidget(self._build_grid())  # _PAGE_GRID
        return self._right

    def _build_empty_state(self) -> QWidget:
        """Build the idle placeholder shown when no photo is open, so the pane is never empty."""
        page = QWidget()
        box = QVBoxLayout(page)
        box.addStretch(1)
        icon = QLabel()
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = _app_icon().pixmap(96, 96)
        if not pixmap.isNull():
            icon.setPixmap(pixmap)
        box.addWidget(icon)
        self._empty_message = QLabel(_EMPTY_START)
        self._empty_message.setObjectName("empty")
        self._empty_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_message.setWordWrap(True)
        box.addWidget(self._empty_message)
        box.addStretch(1)
        return page

    def _build_grid(self) -> QListWidget:
        grid = QListWidget()
        grid.setViewMode(QListView.ViewMode.IconMode)
        grid.setResizeMode(QListView.ResizeMode.Adjust)
        grid.setMovement(QListView.Movement.Static)
        grid.setIconSize(QSize(_THUMB_SIZE, _THUMB_SIZE))
        grid.setGridSize(QSize(_THUMB_SIZE + 24, _THUMB_SIZE + 40))
        grid.setSpacing(8)
        grid.setUniformItemSizes(True)
        grid.setWordWrap(True)
        grid.itemClicked.connect(self._on_thumb_activated)
        # Thumbnails answer to the same right-click menu as their row in the tree.
        grid.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        grid.customContextMenuRequested.connect(self._on_grid_context_menu)
        self._grid = grid
        return grid

    def _build_detail_panel(self) -> QWidget:
        content = QWidget()
        box = QVBoxLayout(content)
        # Shown only when the selected photo failed to generate: carries the reason and a hint
        # that "Open logs" has the full traceback. Hidden for healthy photos.
        self._error_banner = QLabel()
        self._error_banner.setObjectName("error")
        self._error_banner.setWordWrap(True)
        self._error_banner.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._error_banner.hide()
        box.addWidget(self._error_banner)
        self._preview = QLabel("Select a photo to preview it.")
        self._preview.setObjectName("preview")
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumHeight(240)
        box.addWidget(self._preview)
        box.addLayout(self._build_compare_grid())
        box.addLayout(self._build_details_section())
        box.addLayout(self._build_save_row())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        return scroll

    def _section_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("section")
        return label

    def _build_compare_grid(self) -> QGridLayout:
        """Existing (read-only) and New (editable) columns side by side for easy comparison."""
        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        # The metadata source rides along in the Existing header (it has no "New" counterpart,
        # so a full grid row of its own broke the columns' symmetry).
        existing_header = QHBoxLayout()
        existing_header.addWidget(self._section_label("Existing"))
        self._existing_source = QLabel("")
        self._existing_source.setObjectName("hint")
        self._existing_source.setToolTip(
            "Where the existing metadata was read from: the image file, an XMP sidecar, or both.",
        )
        existing_header.addWidget(self._existing_source)
        existing_header.addStretch(1)
        grid.addLayout(existing_header, 0, 1)
        grid.addWidget(self._section_label("New (editable)"), 0, 2)

        self._existing_title = QLineEdit()
        self._existing_title.setReadOnly(True)
        self._title = QLineEdit()
        self._title.setToolTip("The title to write. Edit freely before saving.")
        grid.addWidget(QLabel("Title"), 1, 0)
        grid.addWidget(self._existing_title, 1, 1)
        grid.addWidget(self._title, 1, 2)

        top = Qt.AlignmentFlag.AlignTop
        self._existing_description = _readonly_box(44)
        self._description = QPlainTextEdit()
        self._description.setToolTip("The description to write.")
        # Descriptions are usually a sentence or two; grow the boxes with the text instead of
        # reserving a fixed block of the pane (textChanged also fires on programmatic fills).
        self._description.textChanged.connect(lambda: _fit_text_height(self._description))
        _fit_text_height(self._description)
        grid.addWidget(QLabel("Description"), 2, 0, top)
        grid.addWidget(self._existing_description, 2, 1)
        grid.addWidget(self._description, 2, 2)

        self._existing_keywords = _readonly_box(150)
        self._keywords = QPlainTextEdit()
        self._keywords.setMinimumHeight(150)
        self._keywords.setPlaceholderText("One per line. Use < for hierarchy (Duck<Bird<Animal)")
        self._keywords.setToolTip(
            "Keywords to write, one per line. Use '<' for a hierarchy "
            "(e.g. 'Duck<Bird<Animal'); the changes and resulting paths show below.",
        )
        self._keywords.textChanged.connect(self._refresh_derived)
        grid.addWidget(QLabel("Keywords"), 3, 0, top)
        grid.addWidget(self._existing_keywords, 3, 1)
        grid.addWidget(self._keywords, 3, 2)
        return grid

    def _build_details_section(self) -> QVBoxLayout:
        """Collapsible keyword-change details: the diff and the resulting hierarchy paths."""
        box = QVBoxLayout()
        self._details_toggle = QToolButton()
        self._details_toggle.setText("Keyword changes")
        self._details_toggle.setCheckable(True)
        self._details_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self._details_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._details_toggle.setToolTip(
            "Show exactly what saving will change: added and removed keywords, plus the "
            "resulting keyword tree.",
        )
        self._details_toggle.toggled.connect(self._on_details_toggled)
        box.addWidget(self._details_toggle)

        self._details_panel = QWidget()
        form = QFormLayout(self._details_panel)
        form.setContentsMargins(0, 0, 0, 0)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self._diff = QTextEdit()
        self._diff.setReadOnly(True)
        self._diff.setMinimumHeight(90)
        self._diff.setToolTip("Keyword changes a save will make: green added, red removed.")
        self._hierarchy = _readonly_box(60)
        self._hierarchy.setObjectName("tree")  # monospace, so the branch guides line up
        self._hierarchy.setToolTip(
            "The keyword tree that saving will write (stored as Lightroom hierarchy paths).",
        )
        form.addRow("Changes", self._diff)
        form.addRow("Tree", self._hierarchy)
        self._details_panel.hide()
        box.addWidget(self._details_panel)
        return box

    def _on_details_toggled(self, expanded: bool) -> None:  # noqa: FBT001 - Qt toggled(bool) slot.
        """Expand or collapse the keyword-change details under the disclosure arrow."""
        arrow = Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        self._details_toggle.setArrowType(arrow)
        self._details_panel.setVisible(expanded)

    def _build_save_options_menu(self) -> QMenu:
        """Build the menu deciding what a save writes: field toggles, merge mode, sidecar."""
        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        self._write_title = QAction("Write Title", self)
        self._write_title.setToolTip("Write the title. Uncheck to leave the existing title as is.")
        self._write_description = QAction("Write Description", self)
        self._write_description.setToolTip(
            "Write the description. Uncheck to leave the existing description as is.",
        )
        self._write_keywords = QAction("Write Keywords", self)
        self._write_keywords.setToolTip(
            "Write keywords. Uncheck to leave existing keywords untouched, e.g. to refresh only "
            "the title and description.",
        )
        # Defaults come from the config file, so choices saved via Settings > Save Settings as
        # Defaults come back on the next launch.
        output = self._defaults.output
        for action, checked in (
            (self._write_title, output.write_title),
            (self._write_description, output.write_description),
            (self._write_keywords, output.write_keywords),
        ):
            action.setCheckable(True)
            action.setChecked(checked)
            menu.addAction(action)
        # Connect only after setChecked above, so building the menu does not fire the handler
        # before _overwrite (which it toggles) has been created further down.
        self._write_keywords.toggled.connect(self._on_write_keywords_toggled)
        menu.addSeparator()

        self._overwrite = QAction("Overwrite Existing Keywords", self)
        self._overwrite.setToolTip(
            "Replace existing keywords instead of merging the new ones in.",
        )
        self._overwrite.toggled.connect(self._refresh_derived)
        self._embed = QAction("Embed in Photo", self)
        self._embed.setToolTip("Write into the image file instead of an XMP sidecar.")
        for action, checked in (
            (self._overwrite, not output.preserve_keywords),
            (self._embed, not output.use_sidecar),
        ):
            action.setCheckable(True)
            action.setChecked(checked)
            menu.addAction(action)
        # A config that starts with keywords off must also start with Overwrite grayed out.
        self._overwrite.setEnabled(self._write_keywords.isChecked())
        return menu

    def _build_save_row(self) -> QHBoxLayout:
        """Per-photo actions at the bottom of the detail pane; batch actions live below."""
        row = QHBoxLayout()
        row.addStretch(1)
        self._generate_one_button = QToolButton()
        self._generate_one_button.setObjectName("split")
        self._generate_one_button.setText("Generate this photo")
        self._generate_one_button.setToolTip(
            "Run the model on just this photo, regardless of which photos are checked.",
        )
        self._generate_one_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self._generate_one_button.clicked.connect(
            lambda: self._generate_current(),  # noqa: PLW0108  # drop Qt's clicked(checked) arg
        )
        self._generate_one_menu = QMenu(self)
        self._generate_one_menu.setToolTipsVisible(True)
        skip_one = self._generate_one_menu.addAction(
            "Generate This Photo (Skip Cache)",
            lambda: self._generate_current(use_cache=False),
        )
        skip_one.setToolTip("One-time: call the model even when a cached result exists.")
        self._generate_one_button.setMenu(self._generate_one_menu)
        self._save_button = QPushButton("Save this photo")
        self._save_button.setToolTip(
            "Write this photo's fields, honoring the Save options next to Save selected.",
        )
        self._save_button.clicked.connect(self._save_current)
        row.addWidget(self._generate_one_button)
        row.addWidget(self._save_button)
        return row

    def _build_bottom_bar(self) -> QHBoxLayout:
        """Status on the left; the batch workflow (generate, then save) on the right."""
        self._status = QLabel("Drag photos or folders here to begin.")
        self._status.setObjectName("status")
        self._progress = QProgressBar()
        self._progress.setMaximumWidth(220)
        self._progress.setFormat("%v / %m")
        self._progress.setVisible(False)

        self._retry_button = QPushButton("Retry failed")
        self._retry_button.setToolTip(
            "Re-run the model on every photo that failed to generate. Enabled once a photo "
            "has actually failed.",
        )
        self._retry_button.setEnabled(False)
        self._retry_button.clicked.connect(self._retry_failed)
        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.setToolTip(
            "Stop generating. The photo currently in flight finishes; the rest are left "
            "untouched so you can resume them later.",
        )
        self._cancel_button.setEnabled(False)
        self._cancel_button.clicked.connect(self._cancel_generation)

        # A split button: a click generates normally (cache included); the arrow offers the
        # one-time skip-cache run without changing the Settings toggle.
        self._generate_button = QToolButton()
        self._generate_button.setObjectName("primarysplit")
        self._generate_button.setText("Generate selected")
        self._generate_button.setToolTip("Run the model on the checked photos.")
        self._generate_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self._generate_button.clicked.connect(
            lambda: self._generate(),  # noqa: PLW0108  # drop Qt's clicked(checked) arg
        )
        # Kept on self: Qt's menu() accessor returns a transient wrapper shiboken may delete.
        self._generate_menu = QMenu(self)
        self._generate_menu.setToolTipsVisible(True)
        skip_all = self._generate_menu.addAction(
            "Generate Selected (Skip Cache)",
            lambda: self._generate(use_cache=False),
        )
        skip_all.setToolTip("One-time: call the model even for photos with cached results.")
        self._generate_button.setMenu(self._generate_menu)

        save_options = QPushButton("Save options")
        save_options.setObjectName("menubutton")
        save_options.setToolTip("Which fields a save writes, merge vs overwrite, and sidecar.")
        save_options.setMenu(self._build_save_options_menu())
        self._save_selected_button = QPushButton("Save selected")
        self._save_selected_button.setObjectName("primary")
        self._save_selected_button.setToolTip(
            "Write the checked photos that have a generated proposal, using the Save options.",
        )
        self._save_selected_button.clicked.connect(self._save_selected)

        row = QHBoxLayout()
        row.addWidget(self._status, stretch=1)
        row.addWidget(self._progress)
        row.addWidget(self._retry_button)
        row.addWidget(self._cancel_button)
        row.addWidget(self._generate_button)
        row.addWidget(save_options)
        row.addWidget(self._save_selected_button)
        return row

    # --- drag and drop ---------------------------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 - Qt override.
        """Accept a drag that carries file/folder URLs."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 - Qt override.
        """Expand dropped files/folders and add the resulting photos to the tree."""
        dropped = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.toLocalFile()]
        if dropped:
            self._add_inputs(dropped)

    # --- adding, removing, listing photos --------------------------------------------------

    def _choose_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Add photos")
        if files:
            self._add_inputs([Path(f) for f in files])

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Add a folder of photos")
        if folder:
            self._add_inputs([Path(folder)])

    def _add_inputs(self, paths: list[Path]) -> None:
        found = expand_inputs(
            paths,
            self._extensions.text().strip(),
            recursive=self._recursive.isChecked(),
        )
        fresh = new_paths([Path(p) for p in self._items], found)
        if not fresh:
            self._update_status()
            return
        for path in fresh:
            self._items[str(path)] = PhotoItem(path=path)
        self._rebuild_tree()
        self._update_status()
        self._start_metadata_scan()

    def _remove_selected(self) -> None:
        item = self._tree.currentItem()
        if item is None:
            return
        path = item.data(0, _PATH_ROLE)
        is_dir = bool(item.data(0, _IS_DIR_ROLE))
        if path is None:
            return
        if is_dir:
            prefix = Path(path)
            removed = [k for k in self._items if Path(k).is_relative_to(prefix)]
        else:
            removed = [path]
        for key in removed:
            self._items.pop(key, None)
            self._preview_cache.pop(key, None)
            if self._current is not None and str(self._current.path) == key:
                self._show_empty()
        self._rebuild_tree()
        self._update_status()

    def _clear(self) -> None:
        if self._thread is not None:
            return
        self._stop_thumbs()
        self._stop_scan()
        self._items.clear()
        self._preview_cache.clear()
        self._thumb_cache.clear()
        self._grid.clear()
        self._grid_items = {}
        self._current = None
        self._rebuild_tree()
        self._show_empty()
        self._status.setText("Drag photos or folders here to begin.")
        self._retry_button.setEnabled(False)

    def _deselect(self, paths: set[Path]) -> int:
        """Uncheck the matched photos and refresh the tree; return how many changed."""
        changed = deselect_paths(self._items, paths)
        if changed:
            self._rebuild_tree()
        return changed

    def _deselect_tagged(self, required: frozenset[str], *, match_all: bool, phrase: str) -> None:
        """Uncheck photos that already carry the chosen field(s); *phrase* names the criterion."""
        if not self._items:
            self._status.setText("Add photos before deselecting.")
            return
        # One batched exiftool read, like the CLI's --skip-tagged. A wait cursor covers the
        # brief pause, the same way opening a photo's metadata does.
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            presence = find_field_presence([item.path for item in self._items.values()])
        finally:
            QApplication.restoreOverrideCursor()
        # The read just told us every photo's fields; refresh the Tagged column for free.
        self._on_scan_done({str(path): set(fields) for path, fields in presence.items()})
        matched = paths_matching_fields(presence, set(required), match_all=match_all)
        changed = self._deselect(matched)
        if changed:
            self._status.setText(
                f"Deselected {changed} photo(s) with {phrase}; "
                f"{self._selected_count()} still selected.",
            )
        else:
            self._status.setText(f"No checked photos have {phrase}.")

    def _deselect_from_file(self) -> None:
        """Pick a skip-list file and uncheck the photos it names."""
        if not self._items:
            self._status.setText("Add photos before deselecting.")
            return
        chosen, _ = QFileDialog.getOpenFileName(self, "Choose a skip-list file")
        if chosen:
            self._apply_skip_file(Path(chosen))

    def _apply_skip_file(self, skip_file: Path) -> None:
        """Uncheck every photo whose name or path is listed in *skip_file*."""
        try:
            entries = load_skip_list(skip_file)
        except DiscoveryError as exc:
            QMessageBox.warning(self, "Could not read the skip list", str(exc))
            return
        if not entries:
            # The file read fine but had nothing usable (empty, blank lines, or only comments).
            # Say so, rather than the ambiguous "Deselected 0" a real no-match would also show.
            self._status.setText("That skip list had no usable entries (empty or only comments).")
            return
        matched = skip_list_matches([item.path for item in self._items.values()], entries)
        changed = self._deselect(matched)
        if changed:
            self._status.setText(
                f"Deselected {changed} photo(s) from the skip list; "
                f"{self._selected_count()} still selected.",
            )
        else:
            self._status.setText("No photos in the list matched the skip list.")

    def _export_csv(self) -> None:
        """Write a CSV report of every photo in the list to a chosen path."""
        if not self._items:
            self._status.setText("Add photos before exporting a CSV.")
            return
        chosen, _ = QFileDialog.getSaveFileName(
            self,
            "Export CSV report",
            "photo-tagger-report.csv",
            "CSV files (*.csv)",
        )
        if not chosen:
            return
        target = Path(chosen)
        if target.suffix.lower() != ".csv":
            target = target.with_suffix(".csv")
        # Fold any unsaved edits in the open photo into its row before exporting.
        self._commit_current()
        overwrite = self._overwrite.isChecked()
        rows = [
            photo_item_to_report_row(item, overwrite=overwrite) for item in self._items.values()
        ]
        try:
            write_report(target, rows)
        except OSError as exc:
            QMessageBox.warning(self, "Could not write the CSV", str(exc))
            return
        self._status.setText(f"Exported {len(rows)} photo(s) to {target.name}.")

    def _rebuild_tree(self) -> None:
        self._syncing = True
        # Build with sorting off so items do not shuffle on every insert; re-enabling at the
        # end re-applies whatever column/direction the header is currently set to.
        self._tree.setSortingEnabled(False)
        self._tree.clear()
        for node in build_tree([item.path for item in self._items.values()]):
            self._add_folder_node(self._tree, node)
        self._tree.setSortingEnabled(True)
        self._syncing = False

    def _add_folder_node(self, parent: object, node: FolderNode) -> None:
        folder_item = _SortableTreeItem(parent, [node.label, ""])
        folder_item.setData(0, _PATH_ROLE, str(node.path))
        folder_item.setData(0, _IS_DIR_ROLE, _DIR_MARK)
        folder_item.setFlags(folder_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        folder_item.setExpanded(True)
        for sub in node.folders:
            self._add_folder_node(folder_item, sub)
        for path in node.files:
            item = self._items[str(path)]
            leaf = _SortableTreeItem(
                folder_item,
                [path.name, file_type_label(path), "", ""],
            )
            leaf.setData(0, _PATH_ROLE, str(path))
            # Files leave _IS_DIR_ROLE unset (None), which reads as "not a folder".
            leaf.setFlags(leaf.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            leaf.setCheckState(0, _checked(item.selected))
            self._render_status_cells(leaf, item)
        self._sync_folder_check(folder_item)

    # --- tree interaction ------------------------------------------------------------------

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._syncing or column != 0:
            return
        self._syncing = True
        if bool(item.data(0, _IS_DIR_ROLE)):
            self._set_descendants_checked(item, item.checkState(0))
        else:
            path = item.data(0, _PATH_ROLE)
            if path is not None:
                self._items[path].selected = item.checkState(0) == Qt.CheckState.Checked
            parent = item.parent()
            while parent is not None:
                self._sync_folder_check(parent)
                parent = parent.parent()
        self._syncing = False
        self._update_status()

    def _set_descendants_checked(self, folder_item: QTreeWidgetItem, state: Qt.CheckState) -> None:
        for index in range(folder_item.childCount()):
            child = folder_item.child(index)
            child.setCheckState(0, state)
            if bool(child.data(0, _IS_DIR_ROLE)):
                self._set_descendants_checked(child, state)
            else:
                path = child.data(0, _PATH_ROLE)
                if path is not None:
                    self._items[path].selected = state == Qt.CheckState.Checked

    def _sync_folder_check(self, folder_item: QTreeWidgetItem) -> None:
        states = {folder_item.child(i).checkState(0) for i in range(folder_item.childCount())}
        if states == {Qt.CheckState.Checked}:
            folder_item.setCheckState(0, Qt.CheckState.Checked)
        elif states == {Qt.CheckState.Unchecked}:
            folder_item.setCheckState(0, Qt.CheckState.Unchecked)
        else:
            folder_item.setCheckState(0, Qt.CheckState.PartiallyChecked)

    def _on_current_changed(self, current: QTreeWidgetItem | None, _previous: object) -> None:
        path = current.data(0, _PATH_ROLE) if current is not None else None
        if current is not None and bool(current.data(0, _IS_DIR_ROLE)) and path is not None:
            # A folder: show its thumbnail grid instead of a single photo's detail.
            self._current = None
            self._show_detail(enabled=False)
            self._show_grid(Path(path))
            return
        self._stop_thumbs()
        if path is None:
            self._show_empty()
            return
        self._right.setCurrentIndex(_PAGE_DETAIL)
        self._current = self._items[path]
        self._show_item(self._items[path])

    # --- folder thumbnail grid -------------------------------------------------------------

    def _show_grid(self, folder: Path) -> None:
        self._stop_thumbs()
        self._grid.clear()
        self._grid_items = {}
        under = paths_under([item.path for item in self._items.values()], folder)
        pending: list[Path] = []
        for path in under:
            key = str(path)
            grid_item = QListWidgetItem(path.name)
            grid_item.setData(_PATH_ROLE, key)
            self._grid.addItem(grid_item)
            self._grid_items[key] = grid_item
            self._update_grid_item(self._items[key], grid_item)
            if key not in self._thumb_cache:
                pending.append(path)
        self._right.setCurrentIndex(_PAGE_GRID)
        self._status.setText(f"{len(under)} photo(s) in {folder.name or folder}.")
        if pending:
            self._start_thumbs(pending)

    def _update_grid_item(self, item: PhotoItem, grid_item: QListWidgetItem) -> None:
        """Refresh a grid thumbnail: the image (or placeholder) plus its state badges."""
        key = str(item.path)
        base = self._thumb_cache.get(key, self._placeholder_pixmap)
        badges = thumb_badges(item, has_sidecar=item.path.with_suffix(".xmp").exists())
        grid_item.setIcon(QIcon(_badged_pixmap(base, badges)))
        notes = [_BADGE_TEXT[name] for name in badges]
        if item.status == FAILED and item.error:
            notes.append(item.error)
        grid_item.setToolTip("\n".join([item.path.name, *notes]))

    def _on_thumb_activated(self, item: QListWidgetItem) -> None:
        key = item.data(_PATH_ROLE)
        leaf = self._leaf_for(Path(key))
        if leaf is not None:
            self._tree.setCurrentItem(leaf)  # routes to the detail page

    def _on_thumb_ready(self, path: str, data: bytes) -> None:
        pixmap = QPixmap()
        pixmap.loadFromData(data)
        if pixmap.isNull():
            return
        self._thumb_cache[path] = pixmap
        grid_item = self._grid_items.get(path)
        item = self._items.get(path)
        if grid_item is not None and item is not None:
            self._update_grid_item(item, grid_item)

    def _start_thumbs(self, paths: list[Path]) -> None:
        self._thumb_thread = QThread(self)
        self._thumb_worker = ThumbnailWorker(paths)
        self._thumb_worker.moveToThread(self._thumb_thread)
        self._thumb_thread.started.connect(self._thumb_worker.run)
        self._thumb_worker.ready.connect(self._on_thumb_ready)
        self._thumb_thread.start()

    def _stop_thumbs(self) -> None:
        if self._thumb_worker is not None:
            self._thumb_worker.stop()
        if self._thumb_thread is not None:
            self._thumb_thread.quit()
            self._thumb_thread.wait()
            self._thumb_thread = None
        self._thumb_worker = None

    # --- detail pane -----------------------------------------------------------------------

    def _show_item(self, item: PhotoItem) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._ensure_loaded(item)
            self._ensure_sources(item)
            self._render_preview(item)
        finally:
            QApplication.restoreOverrideCursor()
        sources = ", ".join(item.existing_sources)
        self._existing_source.setText(f"(from {sources})" if sources else "(no metadata found)")
        self._existing_title.setText(item.existing_title or _NONE)
        self._existing_description.setPlainText(item.existing_description or _NONE)
        _fit_text_height(self._existing_description)
        existing_kw = format_existing_keywords(item.existing_keywords)
        self._existing_keywords.setPlainText(existing_kw or _NONE)
        self._title.setText(item.title)
        self._description.setPlainText(item.description)
        self._keywords.setPlainText(keywords_to_text(item.keywords))
        self._show_detail(enabled=True)
        self._update_error_banner(item)
        self._refresh_derived()

    def _update_error_banner(self, item: PhotoItem) -> None:
        """Show the failure reason for a failed photo; hide the banner otherwise."""
        if item.status == FAILED and item.error:
            self._error_banner.setText(
                f"Generation failed: {item.error}\n"
                "Use 'Retry failed' to try again, or 'Open logs' for the full traceback.",
            )
            self._error_banner.show()
        else:
            self._error_banner.hide()

    def _ensure_loaded(self, item: PhotoItem) -> None:
        if item.loaded:
            return
        title, description = read_caption(item.path)
        context = read_image_context(item.path)
        item.existing_title = title
        item.existing_description = description
        item.existing_keywords = context.existing_keywords
        item.loaded = True
        if not item.has_proposal:
            # Seed the editable copy from the existing values so a file can be edited
            # and saved even without generating a proposal first.
            item.title = title or ""
            item.description = description or ""
            item.keywords = list(context.existing_keywords.subject)

    def _ensure_sources(self, item: PhotoItem) -> None:
        if item.sources_read:
            return
        item.existing_sources = read_metadata_sources(item.path)
        item.sources_read = True

    def _render_preview(self, item: PhotoItem) -> None:
        pixmap = self._preview_pixmap(item)
        if pixmap is None or pixmap.isNull():
            self._preview.setText("(no preview available)")
            return
        scaled = pixmap.scaled(
            self._preview.width(),
            self._preview.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._preview.setPixmap(scaled)

    def _preview_pixmap(self, item: PhotoItem) -> QPixmap | None:
        key = str(item.path)
        if key in self._preview_cache:
            return self._preview_cache[key]
        try:
            content = prepare_image_for_agent(item.path, max_size=_PREVIEW_MAX)
        except Exception as exc:  # noqa: BLE001
            # A preview must never crash the window; degrade to a placeholder.
            logger.warning("gui_preview_failed", file=item.path.name, error=str(exc))
            return None
        pixmap = QPixmap()
        pixmap.loadFromData(content.data)
        self._preview_cache[key] = pixmap
        return pixmap

    def _on_write_keywords_toggled(self) -> None:
        """Overwrite-vs-merge only matters when keywords are written; gray it out otherwise."""
        self._overwrite.setEnabled(self._write_keywords.isChecked())
        self._refresh_derived()

    def _refresh_derived(self) -> None:
        """Recompute the keyword-change diff and hierarchy preview from the edited fields."""
        if self._current is None:
            return
        if not self._write_keywords.isChecked():
            # Keywords are not being written, so the diff and hierarchy do not apply.
            self._diff.setHtml("(keywords will not be written)")
            self._hierarchy.setPlainText(_NONE)
            self._details_toggle.setText("Keyword changes (not written)")
            return
        edited = parse_keyword_lines(self._keywords.toPlainText())
        overwrite = self._overwrite.isChecked()
        existing = self._current.existing_keywords
        paths = hierarchy_preview(existing, edited, overwrite=overwrite)
        self._hierarchy.setPlainText(paths or _NONE)
        diff = keyword_diff(existing, edited, overwrite=overwrite)
        self._diff.setHtml(_diff_html(diff))
        # A collapsed section still tells the user whether saving changes anything.
        added = sum(1 for _kw, state in diff if state == ADDED)
        removed = sum(1 for _kw, state in diff if state == REMOVED)
        summary = f"+{added} / -{removed}" if added or removed else "no change"
        self._details_toggle.setText(f"Keyword changes ({summary})")

    def _show_detail(self, *, enabled: bool) -> None:
        for widget in (
            self._title,
            self._description,
            self._keywords,
            self._save_button,
            self._generate_one_button,
        ):
            widget.setEnabled(enabled)
        if not enabled:
            self._error_banner.hide()

    def _show_empty(self) -> None:
        """Show the idle placeholder page instead of an empty detail form, and clear selection."""
        self._current = None
        self._show_detail(enabled=False)
        self._empty_message.setText(_EMPTY_PICK if self._items else _EMPTY_START)
        self._right.setCurrentIndex(_PAGE_EMPTY)

    def _commit_current(self) -> None:
        """Copy the visible editable fields back onto the selected item."""
        item = self._current
        if item is None:
            return
        item.title = self._title.text().strip()
        item.description = self._description.toPlainText().strip()
        item.keywords = parse_keyword_lines(self._keywords.toPlainText())

    def _write_fields_chosen(self) -> bool:
        """Report whether at least one write toggle (Title/Description/Keywords) is on."""
        return (
            self._write_title.isChecked()
            or self._write_description.isChecked()
            or self._write_keywords.isChecked()
        )

    def _write_item(self, item: PhotoItem) -> bool:
        """
        Write the item's checked fields to disk; return success.

        Unchecked fields stay as is.
        """
        keywords = (
            keywords_to_save(
                item.existing_keywords,
                item.keywords,
                overwrite=self._overwrite.isChecked(),
            )
            if self._write_keywords.isChecked()
            else KeywordSet()
        )
        ok = write_metadata(
            item.path,
            keywords,
            description=(item.description or None) if self._write_description.isChecked() else None,
            title=(item.title or None) if self._write_title.isChecked() else None,
            use_sidecar=not self._embed.isChecked(),
        )
        item.status = SAVED if ok else FAILED
        self._refresh_status_cell(item)
        return ok

    def _save_current(self) -> None:
        if self._current is None:
            return
        if not self._write_fields_chosen():
            self._status.setText(
                "Pick at least one field to write (Title, Description, or Keywords).",
            )
            return
        self._commit_current()
        ok = self._write_item(self._current)
        name = self._current.path.name
        self._status.setText(f"Saved {name}." if ok else f"Failed to save {name}.")
        self._resort()
        self._update_status()

    def _save_selected(self) -> None:
        if not self._write_fields_chosen():
            self._status.setText(
                "Pick at least one field to write (Title, Description, or Keywords).",
            )
            return
        self._commit_current()
        targets = [item for item in self._items.values() if item.selected and item.has_proposal]
        if not targets:
            self._status.setText("No checked photos have a proposal to save.")
            return
        saved = sum(int(self._write_item(item)) for item in targets)
        self._status.setText(f"Saved {saved} of {len(targets)} checked photo(s).")
        self._resort()
        self._update_status()

    # --- generation ------------------------------------------------------------------------

    def _generate(self, *, use_cache: bool = True) -> None:
        selected = [item for item in self._items.values() if item.selected]
        if not selected:
            self._status.setText("Check at least one photo first.")
            return
        self._run_generation(selected, use_cache=use_cache)

    def _generate_current(self, *, use_cache: bool = True) -> None:
        if self._current is None:
            self._status.setText("Open a photo to generate it.")
            return
        self._run_generation([self._current], use_cache=use_cache)

    def _retry_failed(self) -> None:
        failed = [item for item in self._items.values() if item.status == FAILED]
        if not failed:
            self._status.setText("No failed photos to retry.")
            return
        self._run_generation(failed)

    def _run_generation(self, items: list[PhotoItem], *, use_cache: bool = True) -> None:
        if self._thread is not None or not items:
            return
        for item in items:
            item.status = WORKING
            self._refresh_status_cell(item)
        current = self._current
        if current is not None and current in items:
            # Clear a stale failure banner the moment its photo is re-queued.
            self._update_error_banner(current)
        self._cancelling = False
        self._set_running(running=True, total=len(items))
        self._status.setText(f"Generating {len(items)} photo(s)...")

        self._thread = QThread(self)
        self._worker = GenerateWorker(
            self._provider_name(),
            self._model.currentText().strip(),
            self._url.text().strip() or None,
            [item.path for item in items],
            api_key=self._api_key_value(),
            cache_file=self._active_cache_file() if use_cache else None,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.file_done.connect(self._on_file_done)
        self._worker.file_failed.connect(self._on_file_failed)
        self._worker.finished.connect(self._on_generate_finished)
        self._thread.start()

    def _on_file_done(self, proposal: Proposal) -> None:
        item = self._items.get(str(proposal.path))
        if item is None:
            return
        apply_proposal(item, proposal)
        self._refresh_status_cell(item)
        self._advance_progress()
        if self._current is item:
            self._show_item(item)
        self._update_status()

    def _on_file_failed(self, path: str, message: str) -> None:
        item = self._items.get(path)
        if item is None:
            return
        item.status = FAILED
        item.error = message
        self._refresh_status_cell(item)
        self._advance_progress()
        if self._current is item:
            self._update_error_banner(item)
        self._update_status()

    def _cancel_generation(self) -> None:
        """Ask the running worker to stop after the photo currently in flight."""
        if self._worker is None:
            return
        self._cancelling = True
        self._worker.stop()
        self._cancel_button.setEnabled(False)
        self._status.setText("Cancelling after the current photo finishes...")

    def _on_generate_finished(self) -> None:
        # A cancelled run leaves the un-started photos marked WORKING; reset them to PENDING so
        # they look queued-again rather than stuck, and report what actually got done.
        reset = self._reset_working()
        if self._cancelling:
            self._status.setText(f"Cancelled. {reset} photo(s) not generated.")
        else:
            self._status.setText("Generation finished.")
        self._cancelling = False
        self._resort()
        self._teardown_thread()

    def _reset_working(self) -> int:
        """Revert any still-WORKING photos to PENDING; return how many were reset."""
        reset = 0
        for item in self._items.values():
            if item.status == WORKING:
                item.status = PENDING
                self._refresh_status_cell(item)
                reset += 1
        return reset

    def _has_failures(self) -> bool:
        """Report whether any photo is currently in the failed state."""
        return any(item.status == FAILED for item in self._items.values())

    def _set_running(self, *, running: bool, total: int = 0) -> None:
        self._generate_button.setEnabled(not running)
        self._generate_one_button.setEnabled(not running)
        self._retry_button.setEnabled(not running and self._has_failures())
        self._test_button.setEnabled(not running)
        self._cancel_button.setEnabled(running)
        self._progress.setVisible(running)
        if running:
            self._progress.setRange(0, total)
            self._progress.setValue(0)

    def _advance_progress(self) -> None:
        """Tick the run progress bar for one finished (or failed) photo."""
        if self._thread is not None:
            self._progress.setValue(self._progress.value() + 1)

    def _teardown_thread(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
        self._worker = None
        self._set_running(running=False)

    # --- providers and diagnostics ---------------------------------------------------------

    def _refresh_models(self) -> None:
        backend = get_backend(self._provider_name())
        base_url = self._url.text().strip() or backend.default_base_url
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            models = backend.list_models(base_url, backend.resolve_api_key(self._api_key_value()))
        except ProviderError as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Could not list models", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        current = self._model.currentText()
        self._model.clear()
        self._model.addItems(rank_vision_models(models))
        self._model.setCurrentText(current)
        self._status.setText(f"Found {len(models)} model(s) on {self._provider_name()}.")

    def _test_connection(self) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            results = run_checks(
                self._provider_name(),
                self._model.currentText().strip(),
                api_base_url=self._url.text().strip() or None,
                api_key=self._api_key_value(),
            )
        finally:
            QApplication.restoreOverrideCursor()
        box = QMessageBox(self)
        box.setWindowTitle("Connection check")
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText(_check_results_html(results))
        all_ok = all(r.ok for r in results)
        box.setIcon(QMessageBox.Icon.Information if all_ok else QMessageBox.Icon.Warning)
        box.exec()

    def _open_logs(self) -> None:
        """Reveal the log folder in the OS file browser so the user can read the run logs."""
        _LOG_FOLDER.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(_LOG_FOLDER)))

    # --- helpers ---------------------------------------------------------------------------

    def _provider_name(self) -> ProviderName:
        """Return the selected provider; the combo carries the internal name as item data."""
        return cast("ProviderName", self._provider.currentData())

    def _api_key_value(self) -> str | None:
        """Return the typed API key, or None to fall back to the provider's env var/default."""
        return self._api_key.text().strip() or None

    def _refresh_status_cell(self, item: PhotoItem) -> None:
        leaf = self._leaf_for(item.path)
        if leaf is not None:
            self._render_status_cells(leaf, item)
        # Keep the folder grid's badge overlay in step with the tree.
        grid_item = self._grid_items.get(str(item.path))
        if grid_item is not None:
            self._update_grid_item(item, grid_item)

    def _render_status_cells(self, leaf: QTreeWidgetItem, item: PhotoItem) -> None:
        """Paint the Status and Tagged columns for *item*'s row."""
        label = _STATUS_LABEL[item.status]
        if item.status == READY and item.from_cache:
            label = "ready (cached)"
        leaf.setText(_COL_STATUS, label)
        leaf.setData(_COL_STATUS, _STATUS_RANK_ROLE, status_sort_rank(item.status))
        color = _STATUS_COLOR.get(item.status)
        leaf.setData(
            _COL_STATUS,
            Qt.ItemDataRole.ForegroundRole,
            QBrush(color) if color is not None else None,
        )
        # Surface the failure reason on hover so it is discoverable straight from the tree.
        leaf.setToolTip(_COL_STATUS, item.error if item.status == FAILED else "")
        if item.known_fields is not None:
            leaf.setText(_COL_TAGGED, tagged_summary(item.known_fields))
            leaf.setToolTip(
                _COL_TAGGED,
                "Already on the file: " + (", ".join(sorted(item.known_fields)) or "nothing"),
            )

    def _resort(self) -> None:
        """Re-apply the active sort so changed statuses settle when sorting by the Status column."""
        header = self._tree.header()
        self._tree.sortItems(header.sortIndicatorSection(), header.sortIndicatorOrder())

    def _leaf_for(self, path: Path) -> QTreeWidgetItem | None:
        target = str(path)
        iterator = QTreeWidgetItemIterator(self._tree)
        while iterator.value():
            item = iterator.value()
            if not bool(item.data(0, _IS_DIR_ROLE)) and item.data(0, _PATH_ROLE) == target:
                return item
            iterator += 1
        return None

    def _selected_count(self) -> int:
        """How many photos are currently checked (used in deselect feedback)."""
        return sum(1 for item in self._items.values() if item.selected)

    def _update_status(self) -> None:
        if self._items:
            self._status.setText(status_summary(self._items.values()))
        # Retry only makes sense when something actually failed (and no run is in flight).
        if self._thread is None:
            self._retry_button.setEnabled(self._has_failures())

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt override.
        """
        Stop any in-flight generation before closing.

        Asking the worker to stop first means closing mid-run only waits for the photo currently in
        flight, not the whole batch. Waiting on the thread keeps a running QThread from being
        destroyed under it.
        """
        if self._worker is not None:
            self._worker.stop()
        self._stop_thumbs()
        self._stop_scan()
        self._teardown_thread()
        self._emit_telemetry()
        super().closeEvent(event)

    def maybe_show_telemetry_notice(self) -> None:
        """
        Show the one-time telemetry disclosure on the first run telemetry is active.

        Offers to turn telemetry off right here; otherwise it stays on and can be toggled later from
        the Settings menu. Called from :func:`launch` after the window is shown, not from
        ``__init__``, so the headless test suite (which builds the window directly) never blocks on
        a modal dialog.
        """
        if not telemetry.should_send(config_enabled=self._telemetry_enabled):
            return
        if telemetry.first_run_notice() is None:  # already shown on an earlier run
            return
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Information)
        box.setWindowTitle("Anonymous usage telemetry")
        box.setText("Photo Tagger sends anonymous usage stats to guide development.")
        box.setInformativeText(
            "Collected: model name, batch size, OS, CPU architecture, timing.\n"
            "Never: photos, file paths, filenames, tags, or personal data.\n\n"
            "You can turn this off now, or anytime from Settings > Send Anonymous Telemetry.",
        )
        keep = box.addButton("Keep Enabled", QMessageBox.ButtonRole.AcceptRole)
        box.addButton("Turn It Off", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(keep)
        box.exec()
        if box.clickedButton() is not keep:
            # Unchecking fires _on_telemetry_toggled, which persists the choice.
            self._telemetry_action.setChecked(False)

    def _emit_telemetry(self) -> None:
        """Fire a best-effort GUI usage beacon on close; never blocks and never raises."""
        telemetry.emit(
            telemetry.RunInfo(
                interface="gui",
                provider=self._provider_name(),
                model=self._model.currentText().strip(),
                batch_size=count_generated(self._items.values()),
                duration_seconds=time.monotonic() - self._session_start,
            ),
            enabled=self._telemetry_enabled,
            block=False,
        )


def _make_placeholder() -> QPixmap:
    """Build a neutral grey tile shown in the grid until a thumbnail loads."""
    pixmap = QPixmap(_THUMB_SIZE, _THUMB_SIZE)
    pixmap.fill(QColor(50, 50, 56))
    return pixmap


# Thumbnail badge rendering: fill color and glyph per badge name. The lifecycle badge (first
# three) draws top-right; the informational ones stack top-left.
_BADGE_STYLE = {
    BADGE_FAILED: ("#f85149", "✗"),  # red cross
    BADGE_SAVED: ("#3fb950", "✓"),  # green check
    BADGE_UNSAVED: ("#6366f1", "•"),  # indigo dot: generated, not saved yet
    BADGE_METADATA: ("#8a8a8a", "M"),  # file already carries metadata
    BADGE_SIDECAR: ("#0e7490", "S"),  # an XMP sidecar exists
}
_BADGE_TEXT = {
    BADGE_FAILED: "generation failed",
    BADGE_SAVED: "saved",
    BADGE_UNSAVED: "generated, not saved yet",
    BADGE_METADATA: "already has metadata",
    BADGE_SIDECAR: "has an XMP sidecar",
}
_LIFECYCLE_BADGES = frozenset({BADGE_FAILED, BADGE_SAVED, BADGE_UNSAVED})


def _badged_pixmap(base: QPixmap, badges: list[str]) -> QPixmap:
    """Overlay the badge dots for *badges* onto a copy of *base*."""
    if not badges or base.isNull():
        return base
    pixmap = QPixmap(base)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    radius = max(12, pixmap.width() // 8)
    margin = max(3, radius // 4)
    left = margin
    font = painter.font()
    font.setPixelSize(int(radius * 0.7))
    font.setBold(True)
    painter.setFont(font)
    for name in badges:
        color, glyph = _BADGE_STYLE[name]
        x = pixmap.width() - radius - margin if name in _LIFECYCLE_BADGES else left
        if name not in _LIFECYCLE_BADGES:
            left += radius + margin
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(color))
        painter.drawEllipse(x, margin, radius, radius)
        painter.setPen(QColor("white"))
        painter.drawText(QRect(x, margin, radius, radius), Qt.AlignmentFlag.AlignCenter, glyph)
    painter.end()
    return pixmap


def _checked(selected: bool) -> Qt.CheckState:  # noqa: FBT001 - tiny private bool mapper.
    """Map a selected flag to a Qt check state."""
    return Qt.CheckState.Checked if selected else Qt.CheckState.Unchecked


_DIFF_STYLE = {
    ADDED: ("color:#3fb950", "+&nbsp;"),
    REMOVED: ("color:#f85149;text-decoration:line-through", "&minus;&nbsp;"),
}


def _check_results_html(results: list[CheckResult]) -> str:
    """
    Render connection-check results as rich text: a colored mark, the check, then its detail.

    Each check gets its own block with the detail on a second line, so long endpoint URLs and error
    messages stay readable instead of running together on one line.
    """
    blocks: list[str] = []
    for result in results:
        mark = (
            '<span style="color:#3fb950;">&#10004;</span>'
            if result.ok
            else '<span style="color:#f85149;">&#10008;</span>'
        )
        name = html.escape(result.name)
        detail = html.escape(result.detail)
        blocks.append(f'{mark} <b>{name}</b><br><span style="color:#8a8a8a;">{detail}</span>')
    return "<br><br>".join(blocks)


def _diff_html(diff: list[tuple[str, str]]) -> str:
    """Render the keyword diff as HTML: green added, red struck-through removed, grey kept."""
    rows: list[str] = []
    for keyword, state in diff:
        safe = html.escape(keyword)
        style, marker = _DIFF_STYLE.get(state, ("color:#8a8a8a", "&nbsp;&nbsp;&nbsp;"))
        rows.append(f'<span style="{style}">{marker}{safe}</span>')
    return "<br>".join(rows) or "(no change)"


def launch(argv: list[str] | None = None) -> int:
    """Create the application, show the main window, and run the event loop."""
    # A Finder/Dock launch inherits a minimal PATH, so graft on the PATH the user's login shell
    # would set. That recovers exiftool wherever their package manager put it (Homebrew, Nix,
    # MacPorts, ...). A no-op for a normal shell launch, where those dirs are already present.
    os.environ["PATH"] = ensure_path_dirs(os.environ.get("PATH", ""), login_shell_path())
    # And bridge a config-file exiftool path into the env var metadata reads (an exported var wins).
    if (exiftool_path := load_defaults().exiftool_path) is not None:
        os.environ.setdefault("PHOTO_TAGGER_EXIFTOOL", exiftool_path)
    # File-only logging: the window carries the live status, so the terminal stays quiet, but a
    # durable log (with full tracebacks for failed photos) is written for the "Open logs" button.
    setup_logging(file_log_level="DEBUG", console_log_level="OFF", log_folder=_LOG_FOLDER)
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("Photo Tagger")
    app.setApplicationDisplayName("Photo Tagger")
    app.setWindowIcon(_app_icon())
    app.setStyleSheet(_stylesheet())
    window = MainWindow()
    window.show()
    window.maybe_show_telemetry_notice()
    return app.exec()
