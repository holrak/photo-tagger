# mypy: ignore-errors
# The lint/pycroscope environment does not install the [gui] extra, so pycroscope cannot import
# this Qt shell (PySide6 is absent) and would also misread shiboken's runtime-generated attributes.
# Exclude the whole module from pycroscope, the same way it is excluded from coverage and zuban;
# the Qt-free logic that is worth analyzing lives in gui_state.py.
# static analysis: ignore
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
from PySide6.QtCore import (
    QLibraryInfo,
    QLocale,
    QObject,
    QRect,
    QSize,
    Qt,
    QThread,
    QTimer,
    QTranslator,
    QUrl,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QActionGroup,
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
    QInputDialog,
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

from photo_tagger import __version__, i18n, telemetry
from photo_tagger.ai import analyze_image_with_ai, create_agent
from photo_tagger.cache import (
    InferenceCache,
    build_cache_namespace,
    content_cache_key,
    open_cache,
    safe_cache_get,
    safe_cache_put,
)
from photo_tagger.cli_options import load_defaults
from photo_tagger.config import (
    DEFAULT_FREQUENCY_PENALTY,
    DEFAULT_JPEG_QUALITY,
    DEFAULT_MAX_TOKENS,
    DEFAULT_OUTPUT_LANGUAGE,
    DEFAULT_TEMPERATURE,
    DEFAULT_USER_PROMPT,
)
from photo_tagger.config_file import find_config_file, load_config, user_config_path
from photo_tagger.csv_report import write_report
from photo_tagger.diagnostics import CheckResult, run_checks
from photo_tagger.discovery import load_skip_list, skip_list_matches
from photo_tagger.errors import DiscoveryError, ProviderError
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
    WORKING,
    FolderNode,
    GuiConfigValues,
    PhotoItem,
    Proposal,
    SaveJob,
    SaveOptions,
    apply_proposal,
    build_save_job,
    build_tree,
    config_text_with_language,
    config_text_with_output_language,
    config_toml_text,
    deselect_paths,
    ensure_path_dirs,
    expand_inputs,
    file_dialog_name_filters,
    file_type_label,
    filter_photos,
    format_existing_keywords,
    hierarchy_preview,
    keyword_diff,
    keywords_to_text,
    login_shell_path,
    merged_config_text,
    new_paths,
    parse_keyword_lines,
    paths_matching_fields,
    paths_under,
    photo_item_to_report_row,
    progress_timing_text,
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
    tooltip,
    wrap_tooltip,
)
from photo_tagger.i18n import _, gettext_noop, ngettext
from photo_tagger.image_io import prepare_image_for_agent
from photo_tagger.logging_setup import setup_logging
from photo_tagger.metadata import (
    FIELD_DESCRIPTION,
    FIELD_KEYWORDS,
    FIELD_TITLE,
    build_contextual_prompt,
    find_field_presence,
    managed_helper,
    prompt_with_hint,
    read_caption,
    read_image_context,
    read_metadata_sources,
    write_metadata,
)
from photo_tagger.providers import PROVIDER_LABELS, PROVIDER_NAMES, ProviderName, get_backend


if TYPE_CHECKING:
    # Annotation-only on Python 3.14 (lazy), so no runtime import is needed.
    from exiftool import ExifToolHelper
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
# How often the elapsed/remaining readout is repainted during a run. Half a second reads as a live
# clock without being busywork.
_TIMING_TICK_MS = 500
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
# Placeholder shown when a photo has no existing title/description/keywords. Module-level strings
# use gettext_noop (extraction marker); their use sites translate with _() at display time.
_NONE = gettext_noop("(none)")

# Title of the warning box shown when persisting a setting to the config file fails.
_CONFIG_SAVE_ERROR_TITLE = gettext_noop("Could not save the config file")

# Right-pane placeholder copy. It adapts to the list: a getting-started nudge while empty, and a
# "pick a photo" nudge once photos are loaded but none is open. This is what fills the right pane
# when there is nothing to inspect, instead of an empty (and confusing) detail form.
_EMPTY_START = gettext_noop(
    "Add photos to get started.\n\n"
    "Drag photos or folders onto the window, or use the Add Photos button.",
)
_EMPTY_PICK = gettext_noop(
    "Select a photo to review it.\n\n"
    "Generate proposes a title, description, and keywords you can edit before saving.",
)

# Short status word shown in the tree's second column.
_STATUS_LABEL = {
    PENDING: "",
    WORKING: gettext_noop("working..."),
    READY: gettext_noop("ready"),
    SAVED: gettext_noop("saved ✓"),
    FAILED: gettext_noop("failed ✗"),
}

# The field-aware "deselect already-tagged" menu, mirroring the CLI's --skip-tagged but letting
# the user pick which fields count as "done". Each entry is (menu label, required fields, whether
# ALL must be present, status-bar phrase). "Any metadata" is the broad OR criterion (the original
# skip-tagged); the rest require all of their fields, so a keyword-only photo survives "a title and
# a description" and stays selected for title/description generation.
_TAGGED_PRESETS: tuple[tuple[str, frozenset[str], bool, str], ...] = (
    (
        gettext_noop("Has any metadata"),
        frozenset({FIELD_TITLE, FIELD_DESCRIPTION, FIELD_KEYWORDS}),
        False,
        gettext_noop("any metadata"),
    ),
    (gettext_noop("Has a title"), frozenset({FIELD_TITLE}), True, gettext_noop("a title")),
    (
        gettext_noop("Has a description"),
        frozenset({FIELD_DESCRIPTION}),
        True,
        gettext_noop("a description"),
    ),
    (
        gettext_noop("Has a title and a description"),
        frozenset({FIELD_TITLE, FIELD_DESCRIPTION}),
        True,
        gettext_noop("a title and a description"),
    ),
    (gettext_noop("Has keywords"), frozenset({FIELD_KEYWORDS}), True, gettext_noop("keywords")),
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
QToolButton#sortdir {
    border: 1px solid rgba(130, 130, 140, 60%); border-radius: 6px; padding: 4px 10px;
}
QToolButton#sortdir:hover { background: rgba(130, 130, 140, 26%); }
QToolButton#sortdir:checked { background: rgba(99, 102, 241, 22%); border-color: #6366f1; }
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
QLabel#hint, QLabel#status, QLabel#timing { color: #8a8a8a; }
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


def _fit_combo(combo: QComboBox) -> None:
    """
    Size a combo box to its widest entry so the drop-down arrow never clips the label.

    The grid toolbar's labels vary in length (and more so once translated), and the default policy
    sizes to the current item only, which clipped the wider entries behind the arrow. Sizing to the
    widest entry keeps every option readable in any language.
    """
    combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
    combo.setMinimumContentsLength(max(len(combo.itemText(i)) for i in range(combo.count())))


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
        *,
        api_key: str | None = None,
        cache_file: Path | None = None,
        output_language: str = DEFAULT_OUTPUT_LANGUAGE,
        hints: dict[str, str] | None = None,
    ) -> None:
        """
        Store the run parameters; nothing happens until :meth:`run`.

        *hints* maps a path (as ``str``) to the photographer's note for that photo; paths without
        one need no entry.
        """
        super().__init__()
        self._provider = provider
        self._model = model
        self._api_base_url = api_base_url
        self._paths = paths
        self._api_key = api_key
        self._cache_file = cache_file
        self._output_language = output_language
        self._hints = hints or {}
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
                output_language=self._output_language,
            )
        except Exception as exc:  # noqa: BLE001
            # Not just PhotoTaggerError: an exception escaping this slot on the worker thread
            # means `finished` never fires, so the window stays in the "running" state (buttons
            # disabled, photos stuck at "working...") for the rest of the session.
            logger.exception("gui_agent_construction_failed", error=str(exc))
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
        return open_cache(
            self._cache_file,
            namespace=_gui_cache_namespace(self._model, self._output_language),
        )

    def _generate_one(self, agent: object, path: Path, cache: InferenceCache | None) -> Proposal:
        """Read existing metadata, run the model (or hit the cache), and assemble a proposal."""
        # The content hash rides along in the context read (one exiftool call), so the cache
        # key covers the image stream only: embedding metadata does not invalidate the entry.
        context = read_image_context(path, include_content_hash=cache is not None)
        existing_title, existing_description = read_caption(path)
        gps_info = {"position": context.gps_position} if context.gps_position else {}
        hint = self._hints.get(str(path), "").strip()
        prompt = build_contextual_prompt(
            prompt_with_hint(DEFAULT_USER_PROMPT, hint),
            context.existing_keywords.subject,
            context.location_tags,
            gps_info,
            camera_info=context.camera_info,
        )
        # Keying and I/O go through the same swallow-and-degrade helpers as the CLI pipeline,
        # so a broken cache entry (or unhashable format) costs a model call, never the photo.
        content_key = content_cache_key(path, context.content_hash) if cache is not None else None
        # A hint means the cached answer was wrong for this photo, so skip the lookup and call
        # the model; the corrected result is stored under the same key below, and hint-less runs
        # replay it from then on.
        cached = (
            safe_cache_get(cache, content_key, file_name=path.name)
            if cache is not None and content_key is not None and not hint
            else None
        )
        inference = cached
        if inference is None:
            jpeg = prepare_image_for_agent(path, max_size=_PREVIEW_MAX)
            inference = analyze_image_with_ai(image_bytes=jpeg, agent=agent, user_prompt=prompt)
            if cache is not None and content_key is not None:
                safe_cache_put(cache, content_key, inference, file_name=path.name)
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


class SaveWorker(QObject):
    """
    Writes the resolved metadata for a list of photos off the UI thread.

    Saving a large batch is minutes of ExifTool work: doing it in the click handler froze the window
    (the OS "busy" cursor and nothing else), which reads as a crash. Here each write reports back by
    signal so the window can tick its progress bar and stay responsive.

    One ExifTool process serves the whole batch, the same way the CLI's batch reads do, instead of
    spawning one per photo.
    """

    file_done = Signal(str, bool)  # path, write succeeded
    finished = Signal()

    def __init__(self, jobs: list[SaveJob], *, backup: bool, use_sidecar: bool) -> None:
        """Store the resolved write jobs and the two file-level options; nothing runs until run."""
        super().__init__()
        self._jobs = jobs
        self._backup = backup
        self._use_sidecar = use_sidecar
        self._emitted = 0
        self._stop = False

    def stop(self) -> None:
        """Ask the loop to stop before the next photo; the write in flight finishes."""
        self._stop = True

    def run(self) -> None:
        """Write every job through one shared ExifTool, then report the batch as finished."""
        try:
            with managed_helper(None) as helper:
                self._write_all(helper)
        except Exception as exc:  # noqa: BLE001
            # Per-photo failures are handled in the loop, so this is ExifTool itself failing to
            # start or shut down. Report whatever never got a result as failed, or the window would
            # wait forever for photos that can no longer be written.
            logger.exception("gui_save_batch_failed", error=str(exc))
            for job in self._jobs[self._emitted :]:
                self.file_done.emit(str(job.path), False)  # noqa: FBT003 - Qt signal argument
        self.finished.emit()

    def _write_all(self, helper: ExifToolHelper) -> None:
        """Write each job in turn, one photo's failure never stopping the rest."""
        for job in self._jobs:
            if self._stop:
                return
            try:
                ok = write_metadata(
                    job.path,
                    job.keywords,
                    description=job.description,
                    title=job.title,
                    backup=self._backup,
                    use_sidecar=self._use_sidecar,
                    et=helper,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("gui_save_failed", file=job.path.name, error=str(exc))
                ok = False
            self._emitted += 1
            self.file_done.emit(str(job.path), ok)


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


def _gui_cache_namespace(model: str, output_language: str) -> str:
    """
    Build the cache namespace for GUI runs: the model plus the GUI's fixed inference settings.

    The GUI runs the agent with the library defaults and sends images at ``_PREVIEW_MAX``, so those
    values (not the CLI flags) are what key its cache entries. The metadata language rides along so
    switching it never replays results generated in the old language.
    """
    return build_cache_namespace(
        model,
        user_prompt=DEFAULT_USER_PROMPT,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
        frequency_penalty=DEFAULT_FREQUENCY_PENALTY,
        jpeg_dimensions=_PREVIEW_MAX,
        jpeg_quality=DEFAULT_JPEG_QUALITY,
        output_language=output_language,
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
        # One disk read serves both views of the config: the Defaults dataclasses and the raw
        # dict (for keys whose CLI-oriented defaults the GUI overrides, like extensions).
        self._raw_config = load_config()
        self._defaults = load_defaults(self._raw_config)
        self._items: dict[str, PhotoItem] = {}
        self._preview_cache: dict[str, QPixmap] = {}
        self._thumb_cache: dict[str, QPixmap] = {}
        self._grid_items: dict[str, QListWidgetItem] = {}
        self._init_grid_view_state()
        self._current: PhotoItem | None = None
        self._init_worker_state()
        self._syncing = False
        self._grid_check_toggled = False
        self._closing = False
        # The persisted UI language choice; "auto" means follow the OS locale. Changing it in the
        # Settings menu rewrites this key and takes effect on the next launch.
        self._language = str(self._raw_config.get("language", i18n.AUTO))
        # The language the model writes titles, descriptions, and keywords in. Distinct from the
        # UI language above; persisted under [inference] output_language, which the CLI shares.
        self._output_language = self._defaults.inference.output_language
        self._cache_file = self._defaults.artifacts.cache_file or _DEFAULT_CACHE_FILE
        # Wall-clock start of this GUI session, reported as the run duration on close.
        self._session_start = time.monotonic()
        # Keys of photos generated this session, for the telemetry batch size. Tracked cumulatively
        # (not counted off the list at close) so clearing or loading another folder never loses it.
        self._session_tagged: set[str] = set()
        # Telemetry on/off: a persisted Settings-menu choice wins over the config-file default.
        _pref = telemetry.read_gui_pref()
        self._telemetry_enabled = self._defaults.telemetry.enabled if _pref is None else _pref

        self.setWindowTitle(f"Photo Tagger {__version__}")
        self.setWindowIcon(_app_icon())
        self.resize(1180, 760)
        self.setAcceptDrops(True)
        self._placeholder_pixmap = _make_placeholder()
        # Built before the panes: both Save buttons (detail pane and bottom bar) attach it.
        self._save_options_menu = self._build_save_options_menu()

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
        self._refresh_save_tooltips()
        self._show_empty()

    def _init_worker_state(self) -> None:
        """
        Seed the handles for every background job the window runs.

        Four independent jobs, each with its own thread so one never waits on another: generation,
        saving, grid thumbnails, and the metadata scan.
        """
        self._thread: QThread | None = None
        self._worker: GenerateWorker | None = None
        self._cancelling = False
        self._save_thread: QThread | None = None
        self._save_worker: SaveWorker | None = None
        # The batch being written, keyed by path: the done handler needs each job's fields to
        # update the Tagged column without re-reading the file.
        self._save_jobs: dict[str, SaveJob] = {}
        self._saved_ok = 0
        # When the run in flight started, for the elapsed/remaining readout. None while idle.
        self._run_started: float | None = None
        self._thumb_thread: QThread | None = None
        self._thumb_worker: ThumbnailWorker | None = None
        self._scan_thread: QThread | None = None
        self._scan_worker: MetadataScanWorker | None = None

    def _init_grid_view_state(self) -> None:
        """
        Seed the folder grid's view state: which folder is shown and how it is filtered and sorted.

        Kept as session state (never written to disk), so the choices carry across folders but reset
        on relaunch. ``_grid_folder`` is what lets the filter/sort controls rebuild the grid.
        """
        self._grid_folder: Path | None = None
        self._grid_sort = SORT_NAME
        self._grid_sort_desc = False
        self._grid_filter = FILTER_ALL

    # --- construction ----------------------------------------------------------------------

    def _build_menus(self) -> None:
        """Build the menu bar: File actions, a Settings telemetry toggle, and Help."""
        menubar = self.menuBar()

        # Kept on self: QAction.menu() hands out a transient wrapper that shiboken may delete,
        # so tests (and future code) need a stable reference to the menu itself.
        file_menu = self._file_menu = menubar.addMenu(_("File"))
        file_menu.addAction(_("Add Photos..."), self._choose_files)
        file_menu.addAction(_("Add Folder..."), self._choose_folder)
        file_menu.addSeparator()
        export_action = file_menu.addAction(_("Export CSV Report..."), self._export_csv)
        export_action.setToolTip(
            tooltip(
                "Save a CSV report of every photo: generated and existing metadata, EXIF, and "
                "token usage.",
            ),
        )
        file_menu.addSeparator()
        file_menu.addAction(_("Clear List"), self._clear)
        file_menu.addSeparator()
        quit_action = file_menu.addAction(_("Quit"), self.close)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.setMenuRole(QAction.MenuRole.QuitRole)

        settings_menu = menubar.addMenu(_("Settings"))
        settings_menu.setToolTipsVisible(True)
        self._cache_action = QAction(_("Cache AI Results"), self)
        self._cache_action.setCheckable(True)
        self._cache_action.setChecked(True)
        self._cache_action.setToolTip(
            tooltip(
                "Reuse earlier results for unchanged photos ({cache_file}). Uncheck to call "
                "the model again for everything; a single photo can skip the cache from its "
                "right-click menu.",
                cache_file=self._cache_file,
            ),
        )
        settings_menu.addAction(self._cache_action)
        self._telemetry_action = QAction(_("Send Anonymous Telemetry"), self)
        self._telemetry_action.setCheckable(True)
        self._telemetry_action.setChecked(self._telemetry_enabled)
        self._telemetry_action.setToolTip(
            tooltip(
                "Anonymous usage stats (model, batch size, OS, CPU/GPU model, RAM, timing) and "
                "crash reports (error type and code location only). No photos, error messages, "
                "or personal data.",
            ),
        )
        self._telemetry_action.toggled.connect(self._on_telemetry_toggled)
        settings_menu.addAction(self._telemetry_action)
        settings_menu.addSeparator()
        self._build_language_menu(settings_menu)
        self._build_output_language_menu(settings_menu)
        settings_menu.addSeparator()
        save_defaults = settings_menu.addAction(
            _("Save Settings as Defaults..."),
            self._save_config,
        )
        save_defaults.setToolTip(
            tooltip(
                "Update the config file with the current provider, model, URL, file types, and "
                "save options. Other settings and comments in the file are preserved; the API "
                "key is never written.",
            ),
        )
        edit_config = settings_menu.addAction(_("Edit Config File..."), self._edit_config)
        edit_config.setToolTip(
            tooltip(
                "Open the config file in your default editor for the settings the GUI does not "
                "surface (prompt file, sampling, workers, filters, ...). Created if missing.",
            ),
        )

        help_menu = self._help_menu = menubar.addMenu(_("Help"))
        help_menu.addAction(
            _("Documentation"),
            lambda: QDesktopServices.openUrl(QUrl(_DOCS_URL)),
        )
        help_menu.addSeparator()
        help_menu.addAction(_("Test Connection"), self._test_connection)
        help_menu.addAction(_("Open Logs"), self._open_logs)
        help_menu.addSeparator()
        about_action = help_menu.addAction(_("About Photo Tagger"), self._show_about)
        about_action.setMenuRole(QAction.MenuRole.AboutRole)

    def _build_language_menu(self, settings_menu: QMenu) -> None:
        """Add the Language submenu: System Default plus every shipped catalog."""
        self._language_menu = settings_menu.addMenu(_("Language"))
        self._language_menu.menuAction().setToolTip(
            tooltip(
                "Language of the app itself (menus, buttons, messages). The language of the "
                "generated metadata is set under Metadata Language.",
            ),
        )
        self._language_group = QActionGroup(self)
        entries = [(i18n.AUTO, _("System Default")), *i18n.SUPPORTED_LANGUAGES.items()]
        for code, label in entries:
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(code == self._language)
            action.triggered.connect(
                lambda _checked=False, chosen=code: self._set_language(chosen),
            )
            self._language_group.addAction(action)
            self._language_menu.addAction(action)

    def _set_language(self, code: str) -> None:
        """Persist the UI language choice into the config file; applied on the next launch."""
        if code == self._language:
            return
        self._language = code
        target = self._config_target()
        try:
            existing = target.read_text(encoding="utf-8") if target.exists() else ""
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(config_text_with_language(existing, code), encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, _(_CONFIG_SAVE_ERROR_TITLE), str(exc))
            return
        self._status.setText(_("Language saved. Restart Photo Tagger to apply it."))

    def _build_output_language_menu(self, settings_menu: QMenu) -> None:
        """Add the Metadata Language submenu: one click sets the generated-metadata language."""
        menu = self._output_language_menu = settings_menu.addMenu(_("Metadata Language"))
        menu.setToolTipsVisible(True)
        menu.menuAction().setToolTip(
            tooltip(
                "Language of the generated titles, descriptions, and keywords. The language of "
                "the app itself is set under Language.",
            ),
        )
        self._output_language_group = QActionGroup(self)
        self._output_language_actions: dict[str, QAction] = {}
        # The trailing rows go in first so the language entries can insert above the separator
        # (a language chosen later via Other... joins the same list).
        self._output_language_separator = menu.addSeparator()
        other = menu.addAction(_("Other..."), self._choose_other_output_language)
        other.setToolTip(
            tooltip(
                "Any language name the model understands works; it is sent to the model as is. "
                "Saved to the config file, which the CLI's --output-language default also reads.",
            ),
        )
        # English (the default) first, then the suggestions sorted by their translated label.
        rest = sorted(
            (name for name in OUTPUT_LANGUAGE_SUGGESTIONS if name != DEFAULT_OUTPUT_LANGUAGE),
            key=lambda name: _(name).casefold(),
        )
        for name in (DEFAULT_OUTPUT_LANGUAGE, *rest):
            self._add_output_language_action(name)
        self._check_output_language_action(self._output_language)

    def _add_output_language_action(self, name: str) -> QAction:
        """
        Insert a checkable entry for *name* above the Other...

        row and return it.
        """
        # The label is translated for display; *name* (the English form) is what the prompt,
        # config file, and cache namespace carry. A custom name passes through _() unchanged.
        action = QAction(_(name), self)
        action.setCheckable(True)
        action.triggered.connect(
            lambda _checked=False, chosen=name: self._set_output_language(chosen),
        )
        self._output_language_group.addAction(action)
        self._output_language_menu.insertAction(self._output_language_separator, action)
        self._output_language_actions[name] = action
        return action

    def _check_output_language_action(self, name: str) -> None:
        """Check *name*'s menu entry, first creating one for a language not in the list."""
        action = self._output_language_actions.get(name)
        if action is None:
            action = self._add_output_language_action(name)
        action.setChecked(True)

    def _choose_other_output_language(self) -> None:
        """Ask for a free-form language name (Other...) and apply it."""
        language, ok = QInputDialog.getText(
            self,
            _("Metadata Language"),
            _("Language for the generated titles, descriptions, and keywords:"),
            text=self._output_language,
        )
        if ok:
            self._set_output_language(language)

    def _set_output_language(self, language: str) -> None:
        """Persist the metadata language into the config file; used from the next generation on."""
        normalized = language.strip() or DEFAULT_OUTPUT_LANGUAGE
        if normalized == self._output_language:
            # Still sync the menu: Other... may have re-entered the current language, and the
            # clicked action must not end up unchecked.
            self._check_output_language_action(normalized)
            return
        self._output_language = normalized
        self._check_output_language_action(normalized)
        target = self._config_target()
        try:
            existing = target.read_text(encoding="utf-8") if target.exists() else ""
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                config_text_with_output_language(existing, normalized),
                encoding="utf-8",
            )
        except OSError as exc:
            QMessageBox.warning(self, _(_CONFIG_SAVE_ERROR_TITLE), str(exc))
            return
        self._status.setText(
            _("Metadata language set to {language} for the next generation.").format(
                language=normalized,
            ),
        )

    def _on_telemetry_toggled(self, enabled: bool) -> None:  # noqa: FBT001 - Qt toggled(bool) slot.
        """Persist the telemetry choice and apply it to this session right away."""
        self._telemetry_enabled = enabled
        telemetry.write_gui_pref(enabled=enabled)
        self._status.setText(
            _("Anonymous telemetry on.") if enabled else _("Anonymous telemetry off."),
        )

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
            backup_xmp=self._backup.isChecked(),
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
                note = _("Updated {target} (other settings and comments preserved).")
            else:
                text = config_toml_text(values)
                note = _("Saved defaults to {target}.")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, _(_CONFIG_SAVE_ERROR_TITLE), str(exc))
            return
        self._status.setText(note.format(target=target))

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
            _("About Photo Tagger"),
            f"<b>Photo Tagger {__version__}</b><br><br>"
            + _("Describe photos and add keywords with a vision-language model.")
            + "<br><br>"
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
        self._provider.setToolTip(tooltip("Backend that serves the vision-language model."))

        self._model = QComboBox()
        self._model.setEditable(True)
        self._model.setMinimumWidth(260)
        self._model.setCurrentText(provider.model_name)
        self._model.setToolTip(
            tooltip(
                "Model identifier. Type it, or press Refresh to list what the provider serves.",
            ),
        )
        refresh = QPushButton(_("Refresh"))
        refresh.setToolTip(tooltip("Query the provider for the models it currently serves."))
        refresh.clicked.connect(self._refresh_models)

        self._connection_dialog = self._build_connection_dialog()
        connection = QPushButton(_("Connection..."))
        connection.setToolTip(tooltip("Server URL, API key, and a connection test."))
        connection.clicked.connect(self._connection_dialog.exec)

        row = QHBoxLayout()
        row.addWidget(QLabel(_("Provider")))
        row.addWidget(self._provider)
        row.addWidget(QLabel(_("Model")))
        row.addWidget(self._model, stretch=1)
        row.addWidget(refresh)
        row.addWidget(connection)
        return row

    def _build_connection_dialog(self) -> QDialog:
        """URL, API key, and the connection test: set-once settings, out of the main window."""
        provider = self._defaults.provider
        dialog = QDialog(self)
        dialog.setWindowTitle(_("Connection settings"))
        dialog.setMinimumWidth(520)
        form = QFormLayout(dialog)
        # macOS style defaults to fixed-size fields; let them fill the dialog width instead.
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self._url = QLineEdit(provider.api_base_url or "")
        self._url.setMinimumWidth(380)
        self._url.setPlaceholderText(_("(provider default URL)"))
        self._url.setToolTip(
            tooltip("Provider API base URL. Leave blank to use the provider's default."),
        )
        form.addRow(_("Base URL"), self._url)

        # Pre-filled from a config-file key if one is set, never from an environment variable: an
        # env key stays in the environment and is resolved at call time, so it never lands in the
        # widget. A typed key is masked, used only for this session, and never written to disk.
        self._api_key = QLineEdit(provider.api_key or "")
        self._api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key.setClearButtonEnabled(True)
        self._api_key.setMinimumWidth(380)
        self._api_key.setPlaceholderText(_("(uses provider env var)"))
        self._api_key.setToolTip(
            tooltip(
                "API key for the provider. Leave blank to use the provider's environment variable "
                "(OPENAI_API_KEY, LM_STUDIO_API_KEY, LLAMA_CPP_API_KEY, or OLLAMA_API_KEY). "
                "Required for OpenAI. A typed key is used for this session only and is never "
                "written to disk.",
            ),
        )
        form.addRow(_("API key"), self._api_key)

        self._test_button = QPushButton(_("Test Connection"))
        self._test_button.setToolTip(
            tooltip("Check ExifTool and that the provider serves the model."),
        )
        self._test_button.clicked.connect(self._test_connection)
        close = QPushButton(_("Close"))
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
        self._tree.setHeaderLabels([_("Photos"), _("Type"), _("Status"), _("Tagged")])
        # Extended: shift+click selects a range, Cmd/Ctrl+click adds single rows, and
        # shift+arrows grow the selection from the keyboard.
        self._tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
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
            tooltip(
                "Click a column header to sort. Type: file extension, +xmp when a sidecar "
                "exists.\nTagged: metadata already on the file ({legend}).",
                legend=tagged_legend(),
            ),
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

        hint = QLabel(_("Drag photos or folders here. Select one and press Delete to remove it."))
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
        add.setText(_("Add Photos..."))
        add.setToolTip(
            tooltip(
                "Add photos (click), or open the arrow for adding a whole folder and for the "
                "folder-scan options. Dragging files or folders onto the window also works.",
            ),
        )
        add.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        add.clicked.connect(self._choose_files)
        add.setMenu(self._build_add_menu())
        controls.addWidget(add)
        controls.addStretch(1)

        select = QPushButton(_("Select"))
        select.setObjectName("menubutton")
        select.setToolTip(tooltip("Check or uncheck photos in bulk."))
        select.setMenu(self._build_select_menu())
        controls.addWidget(select)

        remove = QPushButton(_("Remove"))
        remove.setToolTip(
            tooltip("Remove the selected folder or photo from the list (or press Delete)."),
        )
        remove.clicked.connect(self._remove_selected)
        controls.addWidget(remove)
        return controls

    def _build_add_menu(self) -> QMenu:
        """Build the Add button's arrow menu: the folder dialog plus the folder-scan settings."""
        menu = QMenu(self)
        menu.addAction(_("Add Folder..."), self._choose_folder)
        menu.addSeparator()
        panel = QWidget()
        form = QFormLayout(panel)
        self._extensions = QLineEdit(self._raw_config.get("extensions", DEFAULT_GUI_EXTENSIONS))
        self._extensions.setMinimumWidth(280)
        self._extensions.setToolTip(
            tooltip(
                "Extensions to scan for in folders (comma-separated).\n"
                "Case-insensitive: jpg matches .JPG. Note jpeg is separate from jpg.",
            ),
        )
        form.addRow(_("File types"), self._extensions)
        self._recursive = QCheckBox(_("Include subfolders"))
        self._recursive.setChecked(bool(self._raw_config.get("recursive", True)))
        self._recursive.setToolTip(tooltip("Descend into subfolders when adding a folder."))
        form.addRow("", self._recursive)
        host = QWidgetAction(menu)
        host.setDefaultWidget(panel)
        menu.addAction(host)
        return menu

    def _build_select_menu(self) -> QMenu:
        """Bulk check/uncheck actions, including the CLI's --skip-tagged/--skip-from mirrors."""
        menu = self._select_menu = QMenu(self)
        menu.addAction(_("Check All"), lambda: self._set_all_checked(checked=True))
        menu.addAction(_("Uncheck All"), lambda: self._set_all_checked(checked=False))
        menu.addAction(_("Invert Checked"), self._invert_checked)
        menu.addSeparator()

        self._tagged_menu = menu.addMenu(_("Uncheck Already Tagged"))
        self._tagged_menu.setToolTip(
            tooltip(
                "Uncheck photos that already have the chosen metadata (in the image or its XMP "
                "sidecar), e.g. 'a title and a description' to skip those while keeping "
                "keyword-only photos. Mirrors the CLI's --skip-tagged.",
            ),
        )
        for text, required, match_all, phrase in _TAGGED_PRESETS:
            action = self._tagged_menu.addAction(_(text))
            action.triggered.connect(
                lambda _checked=False, req=required, all_=match_all, ph=phrase: (
                    self._deselect_tagged(
                        req,
                        match_all=all_,
                        phrase=ph,
                    )
                ),
            )

        from_file = menu.addAction(_("Uncheck From Skip List..."), self._deselect_from_file)
        from_file.setToolTip(
            tooltip(
                "Uncheck photos whose filename or full path is listed in a text file (one per "
                "line), like the CLI's --skip-from.",
            ),
        )
        return menu

    def _set_all_checked(self, *, checked: bool) -> None:
        """Check or uncheck every photo at once."""
        if not self._items:
            self._status.setText(_("Add photos before selecting."))
            return
        for item in self._items.values():
            item.selected = checked
        self._rebuild_tree()
        self._update_status()

    def _invert_checked(self) -> None:
        """Flip every photo's checkbox: checked becomes unchecked and vice versa."""
        if not self._items:
            self._status.setText(_("Add photos before selecting."))
            return
        for item in self._items.values():
            item.selected = not item.selected
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
        # A right-click on one of several selected photos acts on the whole selection.
        selected = self._selected_photo_items()
        clicked_in_selection = any(str(item.path) == path for item in selected)
        if not bool(tree_item.data(0, _IS_DIR_ROLE)) and len(selected) > 1 and clicked_in_selection:
            return self._build_bulk_context_menu(selected)
        menu = QMenu(self._tree)
        item = self._items.get(path)
        if not bool(tree_item.data(0, _IS_DIR_ROLE)) and item is not None:
            label = _("Retry Generation") if item.status == FAILED else _("Generate")
            generate = menu.addAction(label, lambda: self._run_generation([item]))
            generate.setEnabled(self._thread is None)
            fresh = menu.addAction(
                _("Generate (Skip Cache)"),
                lambda: self._run_generation([item], use_cache=False),
            )
            fresh.setToolTip(
                tooltip("Call the model even when a cached result exists for this photo."),
            )
            fresh.setEnabled(self._thread is None)
            menu.addSeparator()
        menu.addAction(reveal_label(sys.platform), lambda: self._reveal(Path(path)))
        menu.addSeparator()
        remove = menu.addAction(_("Remove From List"))
        remove.triggered.connect(
            lambda: (self._tree.setCurrentItem(tree_item), self._remove_selected()),
        )
        return menu

    def _selected_photo_items(self) -> list[PhotoItem]:
        """Return the photos behind the tree's currently selected file rows."""
        out: list[PhotoItem] = []
        for tree_item in self._tree.selectedItems():
            if bool(tree_item.data(0, _IS_DIR_ROLE)):
                continue
            item = self._items.get(tree_item.data(0, _PATH_ROLE))
            if item is not None:
                out.append(item)
        return out

    def _build_bulk_context_menu(self, items: list[PhotoItem]) -> QMenu:
        """Build the context menu shown when several photos are selected at once."""
        menu = QMenu(self)
        n = len(items)
        menu.addAction(
            ngettext("Check {n} Photo", "Check {n} Photos", n).format(n=n),
            lambda: self._set_items_checked(items, checked=True),
        )
        menu.addAction(
            ngettext("Uncheck {n} Photo", "Uncheck {n} Photos", n).format(n=n),
            lambda: self._set_items_checked(items, checked=False),
        )
        only = menu.addAction(
            ngettext("Check Only {n} Photo", "Check Only {n} Photos", n).format(n=n),
            lambda: self._check_only_items(items),
        )
        only.setToolTip(
            tooltip("Check the selected photos and uncheck every other photo in the list."),
        )
        menu.addSeparator()
        generate = menu.addAction(
            ngettext("Generate {n} Photo", "Generate {n} Photos", n).format(n=n),
            lambda: self._run_generation(items),
        )
        generate.setEnabled(self._thread is None)
        fresh = menu.addAction(
            ngettext(
                "Generate {n} Photo (Skip Cache)",
                "Generate {n} Photos (Skip Cache)",
                n,
            ).format(n=n),
            lambda: self._run_generation(items, use_cache=False),
        )
        fresh.setToolTip(tooltip("One-time: call the model even for photos with cached results."))
        fresh.setEnabled(self._thread is None)
        menu.addSeparator()
        menu.addAction(
            _("Remove From List"),
            lambda: self._remove_items([str(item.path) for item in items]),
        )
        return menu

    def _check_only_items(self, items: list[PhotoItem]) -> None:
        """Check exactly *items*: everything else in the list is unchecked."""
        keys = {str(item.path) for item in items}
        others = [item for item in self._items.values() if str(item.path) not in keys]
        self._set_items_checked(others, checked=False)
        self._set_items_checked(items, checked=True)

    def _set_items_checked(self, items: list[PhotoItem], *, checked: bool) -> None:
        """Check or uncheck *items* in place, keeping tree, folders, and grid in step."""
        self._syncing = True
        leaves = []
        for item in items:
            item.selected = checked
            leaf = self._leaf_for(item.path)
            if leaf is not None:
                leaf.setCheckState(0, _checked(checked))
                leaves.append(leaf)
        # Folder tristates re-derive after every leaf is set, walking each chain upward.
        for leaf in leaves:
            parent = leaf.parent()
            while parent is not None:
                self._sync_folder_check(parent)
                parent = parent.parent()
        self._sync_grid_checks()
        self._syncing = False
        self._update_status()

    def _on_grid_item_changed(self, grid_item: QListWidgetItem) -> None:
        """Mirror a thumbnail checkbox change onto the photo and its tree row."""
        if self._syncing:
            return
        key = grid_item.data(_PATH_ROLE)
        item = self._items.get(key)
        if item is None:
            return
        checked = grid_item.checkState() == Qt.CheckState.Checked
        if checked == item.selected:
            # Icon and tooltip updates also emit itemChanged; only a real check-state change
            # counts as a toggle.
            return
        # Remember that this was a checkbox click so the itemClicked that follows the same
        # mouse release does not also open the photo. The itemClicked (when there is one)
        # arrives within the same event dispatch, so clear the flag once the loop settles;
        # otherwise a keyboard (space key) toggle would leave it stale and swallow the user's
        # next real click on a thumbnail.
        self._grid_check_toggled = True
        QTimer.singleShot(0, self._clear_grid_check_toggled)
        item.selected = checked
        self._syncing = True
        leaf = self._leaf_for(Path(key))
        if leaf is not None:
            leaf.setCheckState(0, _checked(item.selected))
            parent = leaf.parent()
            while parent is not None:
                self._sync_folder_check(parent)
                parent = parent.parent()
        self._syncing = False
        self._update_status()

    def _clear_grid_check_toggled(self) -> None:
        """Reset the checkbox-click marker once the current event dispatch has finished."""
        self._grid_check_toggled = False

    def _sync_grid_checks(self) -> None:
        """Repaint every visible thumbnail checkbox from the model (callers hold _syncing)."""
        for key, grid_item in self._grid_items.items():
            item = self._items.get(key)
            if item is not None:
                grid_item.setCheckState(_checked(item.selected))

    def _on_grid_context_menu(self, pos: object) -> None:
        """Offer a thumbnail the same right-click actions as its row in the tree."""
        grid_item = self._grid.itemAt(pos)
        if grid_item is None:
            return
        selected_keys = [gi.data(_PATH_ROLE) for gi in self._grid.selectedItems()]
        if len(selected_keys) > 1 and grid_item.data(_PATH_ROLE) in selected_keys:
            items = [self._items[key] for key in selected_keys if key in self._items]
            menu: QMenu | None = self._build_bulk_context_menu(items)
        else:
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
                # Merge rather than replace: a save may have added fields while the scan ran.
                item.known_fields = set(fields) | (item.known_fields or set())
                self._refresh_status_cell(item)
        # The Untagged filter depends on this scan, so a grid showing it must re-evaluate once the
        # results land; other filters and the Tagged sort read state the scan does not change.
        if self._grid_filter == FILTER_UNTAGGED:
            self._resort_grid()

    def _on_scan_finished(self) -> None:
        """Tear down the scan thread and pick up photos added while it ran."""
        self._stop_scan()
        self._start_metadata_scan()

    def _stop_scan(self) -> None:
        if self._scan_worker is not None:
            self._scan_worker.deleteLater()
        if self._scan_thread is not None:
            self._scan_thread.quit()
            self._scan_thread.wait()
            self._scan_thread.deleteLater()
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
        self._empty_message = QLabel(_(_EMPTY_START))
        self._empty_message.setObjectName("empty")
        self._empty_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_message.setWordWrap(True)
        box.addWidget(self._empty_message)
        box.addStretch(1)
        return page

    def _build_grid(self) -> QWidget:
        """Build the folder grid page: a filter/sort toolbar above the thumbnail list."""
        page = QWidget()
        box = QVBoxLayout(page)
        box.setContentsMargins(0, 0, 0, 0)
        box.addLayout(self._build_grid_toolbar())

        grid = QListWidget()
        grid.setViewMode(QListView.ViewMode.IconMode)
        grid.setResizeMode(QListView.ResizeMode.Adjust)
        grid.setMovement(QListView.Movement.Static)
        grid.setIconSize(QSize(_THUMB_SIZE, _THUMB_SIZE))
        grid.setGridSize(QSize(_THUMB_SIZE + 24, _THUMB_SIZE + 40))
        grid.setSpacing(8)
        grid.setUniformItemSizes(True)
        grid.setWordWrap(True)
        grid.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        grid.itemClicked.connect(self._on_thumb_activated)
        grid.itemChanged.connect(self._on_grid_item_changed)
        # Thumbnails answer to the same right-click menu as their row in the tree.
        grid.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        grid.customContextMenuRequested.connect(self._on_grid_context_menu)
        self._grid = grid
        box.addWidget(grid, stretch=1)
        return page

    def _build_grid_toolbar(self) -> QHBoxLayout:
        """
        Filter and sort controls for the folder grid, standing in for the tree's column headers.

        The grid has no headers to click, so a Show filter narrows which photos appear and a Sort by
        field picker plus an ascending/descending toggle order what remains. Each combo is sized to
        its widest entry so no label is clipped. Signals are wired only after the widgets are
        populated, so building them fires no spurious rebuild.
        """
        row = QHBoxLayout()
        row.addWidget(QLabel(_("Show")))
        self._grid_filter_combo = QComboBox()
        for label, criterion in (
            (_("All"), FILTER_ALL),
            (_("Selected"), FILTER_SELECTED),
            (_("Not generated"), FILTER_PENDING),
            (_("Generated"), FILTER_GENERATED),
            (_("Saved"), FILTER_SAVED),
            (_("Failed"), FILTER_FAILED),
            (_("Untagged"), FILTER_UNTAGGED),
        ):
            self._grid_filter_combo.addItem(label, criterion)
        self._grid_filter_combo.setToolTip(tooltip("Show only the photos in the chosen state."))
        _fit_combo(self._grid_filter_combo)
        row.addWidget(self._grid_filter_combo)
        row.addSpacing(16)

        row.addWidget(QLabel(_("Sort by")))
        self._grid_sort_combo = QComboBox()
        for label, criterion in (
            (_("Name"), SORT_NAME),
            (_("Type"), SORT_TYPE),
            (_("Status"), SORT_STATUS),
            (_("Tagged"), SORT_TAGGED),
        ):
            self._grid_sort_combo.addItem(label, criterion)
        self._grid_sort_combo.setToolTip(
            tooltip("Order the thumbnails by name, file type, status, or existing metadata."),
        )
        _fit_combo(self._grid_sort_combo)
        row.addWidget(self._grid_sort_combo)

        self._grid_sort_dir = QToolButton()
        self._grid_sort_dir.setObjectName("sortdir")
        self._grid_sort_dir.setCheckable(True)
        self._grid_sort_dir.setText("↑")
        self._grid_sort_dir.setToolTip(tooltip("Ascending. Click to sort descending."))
        row.addWidget(self._grid_sort_dir)
        row.addStretch(1)

        self._grid_filter_combo.currentIndexChanged.connect(self._on_grid_filter_changed)
        self._grid_sort_combo.currentIndexChanged.connect(self._on_grid_sort_changed)
        self._grid_sort_dir.toggled.connect(self._on_grid_sort_dir_toggled)
        return row

    def _on_grid_filter_changed(self) -> None:
        """Re-filter the visible grid when the Show choice changes."""
        self._grid_filter = self._grid_filter_combo.currentData()
        self._resort_grid()

    def _on_grid_sort_changed(self) -> None:
        """Re-sort the visible grid when the sort field changes."""
        self._grid_sort = self._grid_sort_combo.currentData()
        self._resort_grid()

    def _on_grid_sort_dir_toggled(self, descending: bool) -> None:  # noqa: FBT001 - Qt toggled slot
        """Flip the grid between ascending and descending, updating the toggle's arrow."""
        self._grid_sort_desc = descending
        self._grid_sort_dir.setText("↓" if descending else "↑")
        self._grid_sort_dir.setToolTip(
            tooltip("Descending. Click to sort ascending.")
            if descending
            else _("Ascending. Click to sort descending."),
        )
        self._resort_grid()

    def _resort_grid(self) -> None:
        """Rebuild the grid with the current filter and order; cached thumbnails are reused."""
        if self._grid_folder is not None and self._right.currentIndex() == _PAGE_GRID:
            self._show_grid(self._grid_folder)

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
        self._preview = QLabel(_("Select a photo to preview it."))
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
        existing_header.addWidget(self._section_label(_("Existing")))
        self._existing_source = QLabel("")
        self._existing_source.setObjectName("hint")
        self._existing_source.setToolTip(
            tooltip(
                "Where the existing metadata was read from: the image file, an XMP sidecar, "
                "or both.",
            ),
        )
        existing_header.addWidget(self._existing_source)
        existing_header.addStretch(1)
        grid.addLayout(existing_header, 0, 1)
        grid.addWidget(self._section_label(_("New (editable)")), 0, 2)

        self._existing_title = QLineEdit()
        self._existing_title.setReadOnly(True)
        self._title = QLineEdit()
        self._title.setToolTip(tooltip("The title to write. Edit freely before saving."))
        grid.addWidget(QLabel(_("Title")), 1, 0)
        grid.addWidget(self._existing_title, 1, 1)
        grid.addWidget(self._title, 1, 2)

        top = Qt.AlignmentFlag.AlignTop
        self._existing_description = _readonly_box(44)
        self._description = QPlainTextEdit()
        self._description.setToolTip(tooltip("The description to write."))
        # Descriptions are usually a sentence or two; grow the boxes with the text instead of
        # reserving a fixed block of the pane (textChanged also fires on programmatic fills).
        self._description.textChanged.connect(lambda: _fit_text_height(self._description))
        _fit_text_height(self._description)
        grid.addWidget(QLabel(_("Description")), 2, 0, top)
        grid.addWidget(self._existing_description, 2, 1)
        grid.addWidget(self._description, 2, 2)

        self._existing_keywords = _readonly_box(150)
        self._keywords = QPlainTextEdit()
        self._keywords.setMinimumHeight(150)
        self._keywords.setPlaceholderText(_("One per line. Use < for hierarchy (Duck<Bird<Animal)"))
        self._keywords.setToolTip(
            tooltip(
                "Keywords to write, one per line. Use '<' for a hierarchy "
                "(e.g. 'Duck<Bird<Animal'); the changes and resulting paths show below.",
            ),
        )
        self._keywords.textChanged.connect(self._refresh_derived)
        grid.addWidget(QLabel(_("Keywords")), 3, 0, top)
        grid.addWidget(self._existing_keywords, 3, 1)
        grid.addWidget(self._keywords, 3, 2)
        return grid

    def _build_details_section(self) -> QVBoxLayout:
        """Collapsible keyword-change details: the diff and the resulting hierarchy paths."""
        box = QVBoxLayout()
        self._details_toggle = QToolButton()
        self._details_toggle.setText(_("Keyword changes"))
        self._details_toggle.setCheckable(True)
        self._details_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self._details_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._details_toggle.setToolTip(
            tooltip(
                "Show exactly what saving will change: added and removed keywords, plus the "
                "resulting keyword tree.",
            ),
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
        self._diff.setToolTip(
            tooltip("Keyword changes a save will make: green added, red removed."),
        )
        self._hierarchy = _readonly_box(60)
        self._hierarchy.setObjectName("tree")  # monospace, so the branch guides line up
        self._hierarchy.setToolTip(
            tooltip(
                "The keyword tree that saving will write (stored as Lightroom hierarchy paths).",
            ),
        )
        form.addRow(_("Changes"), self._diff)
        form.addRow(_("Tree"), self._hierarchy)
        self._details_panel.hide()
        box.addWidget(self._details_panel)
        return box

    def _on_details_toggled(self, expanded: bool) -> None:  # noqa: FBT001 - Qt toggled(bool) slot.
        """Expand or collapse the keyword-change details under the disclosure arrow."""
        arrow = Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow
        self._details_toggle.setArrowType(arrow)
        self._details_panel.setVisible(expanded)

    def _build_save_options_menu(self) -> QMenu:
        """Build the menu deciding what a save writes: fields, merge mode, sidecar, backup."""
        menu = QMenu(self)
        menu.setToolTipsVisible(True)
        self._write_title = QAction(_("Write Title"), self)
        self._write_title.setToolTip(
            tooltip("Write the title. Uncheck to leave the existing title as is."),
        )
        self._write_description = QAction(_("Write Description"), self)
        self._write_description.setToolTip(
            tooltip("Write the description. Uncheck to leave the existing description as is."),
        )
        self._write_keywords = QAction(_("Write Keywords"), self)
        self._write_keywords.setToolTip(
            tooltip(
                "Write keywords. Uncheck to leave existing keywords untouched, e.g. to refresh "
                "only the title and description.",
            ),
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

        self._overwrite = QAction(_("Overwrite Existing Keywords"), self)
        self._overwrite.setToolTip(
            tooltip("Replace existing keywords instead of merging the new ones in."),
        )
        self._overwrite.toggled.connect(self._refresh_derived)
        self._embed = QAction(_("Embed in Photo"), self)
        self._embed.setToolTip(tooltip("Write into the image file instead of an XMP sidecar."))
        self._backup = QAction(_("Keep ExifTool Backup"), self)
        self._backup.setToolTip(
            tooltip(
                "Let ExifTool save the untouched file as *_original before writing. Uncheck to "
                "write in place, which leaves no extra copies filling up the disk on a large "
                "batch (make sure you have a backup elsewhere).",
            ),
        )
        for action, checked in (
            (self._overwrite, not output.preserve_keywords),
            (self._embed, not output.use_sidecar),
            (self._backup, output.backup_xmp),
        ):
            action.setCheckable(True)
            action.setChecked(checked)
            menu.addAction(action)
        # A config that starts with keywords off must also start with Overwrite grayed out.
        self._overwrite.setEnabled(self._write_keywords.isChecked())
        # Every toggle refreshes the Save buttons' option summary. Connected after the
        # setChecked calls above so construction never fires into the not-yet-built buttons.
        for action in (
            self._write_title,
            self._write_description,
            self._write_keywords,
            self._overwrite,
            self._embed,
            self._backup,
        ):
            action.toggled.connect(self._refresh_save_tooltips)
        return menu

    def _save_options_summary(self) -> str:
        """Describe what a save currently writes, for the Save buttons' tooltips."""
        fields = [
            name
            for action, name in (
                (self._write_title, _("title")),
                (self._write_description, _("description")),
                (self._write_keywords, _("keywords")),
            )
            if action.isChecked()
        ]
        parts = [", ".join(fields) if fields else _("nothing (pick a field in the arrow menu)")]
        if self._write_keywords.isChecked():
            parts.append(
                _("overwriting existing keywords")
                if self._overwrite.isChecked()
                else _("merging with existing keywords"),
            )
        parts.append(
            _("into the image file") if self._embed.isChecked() else _("to an XMP sidecar"),
        )
        parts.append(
            _("keeping a *_original backup")
            if self._backup.isChecked()
            else _("with no *_original backup"),
        )
        return ", ".join(parts)

    def _refresh_save_tooltips(self) -> None:
        """Keep both Save buttons' tooltips describing the currently chosen options."""
        summary = _("Currently writes {options}.").format(options=self._save_options_summary())
        self._save_button.setToolTip(
            tooltip(
                "Write this photo. {summary} Change what is written with the arrow.",
                summary=summary,
            ),
        )
        self._save_selected_button.setToolTip(
            tooltip(
                "Write the checked photos that have a generated proposal. {summary} "
                "Change what is written with the arrow.",
                summary=summary,
            ),
        )

    def _build_save_row(self) -> QHBoxLayout:
        """Per-photo actions at the bottom of the detail pane; batch actions live below."""
        row = QHBoxLayout()
        hint_label = QLabel(_("Hint for the AI"))
        self._hint = QLineEdit()
        self._hint.setPlaceholderText(_("e.g. 'The animal is a deer, not a boar'"))
        self._hint.setClearButtonEnabled(True)
        self._hint.setToolTip(
            tooltip(
                "A note about this photo that the model trusts over its own reading of the "
                "image; useful when it misidentifies the subject. It is sent along on every "
                "generation of this photo (a hinted photo skips the cached result) and is "
                "never written to the file. Press Enter to regenerate right away.",
            ),
        )
        hint_label.setToolTip(self._hint.toolTip())
        # returnPressed passes no argument, so the keyword-only default (use_cache=True) applies.
        self._hint.returnPressed.connect(self._generate_current)
        row.addWidget(hint_label)
        row.addWidget(self._hint, stretch=1)
        self._generate_one_button = QToolButton()
        self._generate_one_button.setObjectName("split")
        self._generate_one_button.setText(_("Generate This Photo"))
        self._generate_one_button.setToolTip(
            tooltip("Run the model on just this photo, regardless of which photos are checked."),
        )
        self._generate_one_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self._generate_one_button.clicked.connect(
            lambda: self._generate_current(),  # noqa: PLW0108  # drop Qt's clicked(checked) arg
        )
        self._generate_one_menu = QMenu(self)
        self._generate_one_menu.setToolTipsVisible(True)
        skip_one = self._generate_one_menu.addAction(
            _("Generate This Photo (Skip Cache)"),
            lambda: self._generate_current(use_cache=False),
        )
        skip_one.setToolTip(tooltip("One-time: call the model even when a cached result exists."))
        self._generate_one_button.setMenu(self._generate_one_menu)
        # The save options live on the button's own arrow, so what a save writes is
        # discoverable right where the save happens (both Save buttons share one menu).
        self._save_button = QToolButton()
        self._save_button.setObjectName("split")
        self._save_button.setText(_("Save This Photo"))
        self._save_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self._save_button.clicked.connect(
            lambda: self._save_current(),  # noqa: PLW0108  # drop Qt's clicked(checked) arg
        )
        self._save_button.setMenu(self._save_options_menu)
        row.addWidget(self._generate_one_button)
        row.addWidget(self._save_button)
        return row

    def _build_bottom_bar(self) -> QHBoxLayout:
        """Status on the left; the batch workflow (generate, then save) on the right."""
        self._status = QLabel(_("Drag photos or folders here to begin."))
        self._status.setObjectName("status")
        self._progress = QProgressBar()
        self._progress.setMaximumWidth(220)
        self._progress.setFormat("%v / %m")
        self._progress.setVisible(False)
        # Elapsed time (and, once a photo has finished, the estimate of what is left) beside the
        # bar. A long save used to show nothing at all, which looks like the app hung.
        self._timing = QLabel()
        self._timing.setObjectName("timing")
        self._timing.setVisible(False)
        # Repaints the readout between photos so the clock keeps moving during a slow one.
        self._timing_timer = QTimer(self)
        self._timing_timer.setInterval(_TIMING_TICK_MS)
        self._timing_timer.timeout.connect(self._update_timing)

        self._retry_button = QPushButton(_("Retry Failed"))
        self._retry_button.setToolTip(
            tooltip(
                "Re-run the model on every photo that failed to generate. Enabled once a photo "
                "has actually failed.",
            ),
        )
        self._retry_button.setEnabled(False)
        self._retry_button.clicked.connect(self._retry_failed)
        self._cancel_button = QPushButton(_("Cancel"))
        self._cancel_button.setToolTip(
            tooltip(
                "Stop the run in progress, generating or saving. The photo currently in flight "
                "finishes; the rest are left untouched so you can resume them later.",
            ),
        )
        self._cancel_button.setEnabled(False)
        self._cancel_button.clicked.connect(self._cancel_generation)

        # A split button: a click generates normally (cache included); the arrow offers the
        # one-time skip-cache run without changing the Settings toggle.
        self._generate_button = QToolButton()
        self._generate_button.setObjectName("primarysplit")
        self._generate_button.setText(_("Generate Selected"))
        self._generate_button.setToolTip(tooltip("Run the model on the checked photos."))
        self._generate_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self._generate_button.clicked.connect(
            lambda: self._generate(),  # noqa: PLW0108  # drop Qt's clicked(checked) arg
        )
        # Kept on self: Qt's menu() accessor returns a transient wrapper shiboken may delete.
        self._generate_menu = QMenu(self)
        self._generate_menu.setToolTipsVisible(True)
        skip_all = self._generate_menu.addAction(
            _("Generate Selected (Skip Cache)"),
            lambda: self._generate(use_cache=False),
        )
        skip_all.setToolTip(
            tooltip("One-time: call the model even for photos with cached results."),
        )
        self._generate_button.setMenu(self._generate_menu)

        self._save_selected_button = QToolButton()
        self._save_selected_button.setObjectName("primarysplit")
        self._save_selected_button.setText(_("Save Selected"))
        self._save_selected_button.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self._save_selected_button.clicked.connect(
            lambda: self._save_selected(),  # noqa: PLW0108  # drop Qt's clicked(checked) arg
        )
        self._save_selected_button.setMenu(self._save_options_menu)

        row = QHBoxLayout()
        row.addWidget(self._status, stretch=1)
        row.addWidget(self._timing)
        row.addWidget(self._progress)
        row.addWidget(self._retry_button)
        row.addWidget(self._cancel_button)
        row.addWidget(self._generate_button)
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
        filters = file_dialog_name_filters(self._extensions.text().strip())
        files, _filter = QFileDialog.getOpenFileNames(self, _("Add Photos"), "", ";;".join(filters))
        if files:
            self._add_inputs([Path(f) for f in files])

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, _("Add a folder of photos"))
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
        """Remove every selected row (files, and folders with their contents) from the list."""
        removed: list[str] = []
        for tree_item in self._tree.selectedItems():
            path = tree_item.data(0, _PATH_ROLE)
            if path is None:
                continue
            if bool(tree_item.data(0, _IS_DIR_ROLE)):
                prefix = Path(path)
                removed += [k for k in self._items if Path(k).is_relative_to(prefix)]
            else:
                removed.append(path)
        self._remove_items(removed)

    def _remove_items(self, keys: list[str]) -> None:
        """Drop the photos behind *keys* from the list and refresh the tree and grid."""
        if not keys:
            return
        for key in keys:
            self._items.pop(key, None)
            self._preview_cache.pop(key, None)
            self._thumb_cache.pop(key, None)
            if self._current is not None and str(self._current.path) == key:
                self._show_empty()
        self._rebuild_tree()
        # A visible grid still holds the removed thumbnails: dead items whose clicks and
        # checkboxes map to nothing. Rebuild it, or close it when its folder emptied out.
        if self._grid_folder is not None and self._right.currentIndex() == _PAGE_GRID:
            if paths_under(self._item_paths(), self._grid_folder):
                self._show_grid(self._grid_folder)
            else:
                self._grid_folder = None
                self._show_empty()
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
        self._grid_folder = None
        self._current = None
        self._rebuild_tree()
        self._show_empty()
        self._status.setText(_("Drag photos or folders here to begin."))
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
            self._status.setText(_("Add photos before deselecting."))
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
                ngettext(
                    "Deselected {n} photo with {phrase}; {selected} still selected.",
                    "Deselected {n} photos with {phrase}; {selected} still selected.",
                    changed,
                ).format(n=changed, phrase=_(phrase), selected=self._selected_count()),
            )
        else:
            self._status.setText(_("No checked photos have {phrase}.").format(phrase=_(phrase)))

    def _deselect_from_file(self) -> None:
        """Pick a skip-list file and uncheck the photos it names."""
        if not self._items:
            self._status.setText(_("Add photos before deselecting."))
            return
        chosen, _filter = QFileDialog.getOpenFileName(self, _("Choose a skip-list file"))
        if chosen:
            self._apply_skip_file(Path(chosen))

    def _apply_skip_file(self, skip_file: Path) -> None:
        """Uncheck every photo whose name or path is listed in *skip_file*."""
        try:
            entries = load_skip_list(skip_file)
        except DiscoveryError as exc:
            QMessageBox.warning(self, _("Could not read the skip list"), str(exc))
            return
        if not entries:
            # The file read fine but had nothing usable (empty, blank lines, or only comments).
            # Say so, rather than the ambiguous "Deselected 0" a real no-match would also show.
            self._status.setText(
                _("That skip list had no usable entries (empty or only comments)."),
            )
            return
        matched = skip_list_matches([item.path for item in self._items.values()], entries)
        changed = self._deselect(matched)
        if changed:
            self._status.setText(
                ngettext(
                    "Deselected {n} photo from the skip list; {selected} still selected.",
                    "Deselected {n} photos from the skip list; {selected} still selected.",
                    changed,
                ).format(n=changed, selected=self._selected_count()),
            )
        else:
            self._status.setText(_("No photos in the list matched the skip list."))

    def _export_csv(self) -> None:
        """Write a CSV report of every photo in the list to a chosen path."""
        if not self._items:
            self._status.setText(_("Add photos before exporting a CSV."))
            return
        chosen, _filter = QFileDialog.getSaveFileName(
            self,
            _("Export CSV report"),
            "photo-tagger-report.csv",
            _("CSV files (*.csv)"),
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
            QMessageBox.warning(self, _("Could not write the CSV"), str(exc))
            return
        self._status.setText(
            ngettext(
                "Exported {n} photo to {name}.",
                "Exported {n} photos to {name}.",
                len(rows),
            ).format(n=len(rows), name=target.name),
        )

    def _rebuild_tree(self) -> None:
        self._syncing = True
        # Build with sorting off so items do not shuffle on every insert; re-enabling at the
        # end re-applies whatever column/direction the header is currently set to.
        self._tree.setSortingEnabled(False)
        self._tree.clear()
        for node in build_tree([item.path for item in self._items.values()]):
            self._add_folder_node(self._tree, node)
        self._tree.setSortingEnabled(True)
        self._sync_grid_checks()
        # clear() above dropped the selection (_on_current_changed ignores it while _syncing,
        # so the right-hand pane kept whatever was open). Restore the highlight so bulk actions
        # (dragging photos in, Check All, Uncheck Already Tagged) do not kick the user out of
        # the photo or folder they are reviewing.
        if self._current is not None:
            self._select_tree_entry(str(self._current.path), is_dir=False)
        elif self._grid_folder is not None and self._right.currentIndex() == _PAGE_GRID:
            self._select_tree_entry(str(self._grid_folder), is_dir=True)
        self._syncing = False

    def _select_tree_entry(self, path: str, *, is_dir: bool) -> None:
        """Re-highlight the tree row for *path* (callers hold ``_syncing``)."""
        iterator = QTreeWidgetItemIterator(self._tree)
        while iterator.value():
            entry = iterator.value()
            if bool(entry.data(0, _IS_DIR_ROLE)) == is_dir and entry.data(0, _PATH_ROLE) == path:
                self._tree.setCurrentItem(entry)
                return
            iterator += 1

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
        self._sync_grid_checks()
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
        if self._syncing:
            # A programmatic rebuild (tree.clear() emits currentItemChanged(None)) is not the
            # user navigating; reacting would blank the pane they are working in.
            return
        # Keep the open photo's in-progress edits (title, description, keywords, hint) when the
        # user browses away; they are restored when it is opened again. Without this, hinting
        # several photos before one Generate Selected would be impossible: every navigation
        # would drop the hint just typed.
        self._commit_current()
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
        self._grid_folder = folder
        self._grid.clear()
        self._grid_items = {}
        under = [self._items[str(path)] for path in paths_under(self._item_paths(), folder)]
        items = sort_photos(
            filter_photos(under, self._grid_filter),
            self._grid_sort,
            descending=self._grid_sort_desc,
        )
        pending: list[Path] = []
        self._syncing = True
        for item in items:
            key = str(item.path)
            grid_item = QListWidgetItem(item.path.name)
            grid_item.setData(_PATH_ROLE, key)
            grid_item.setFlags(grid_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            self._grid.addItem(grid_item)
            self._grid_items[key] = grid_item
            self._update_grid_item(item, grid_item)
            if key not in self._thumb_cache:
                pending.append(item.path)
        self._syncing = False
        self._right.setCurrentIndex(_PAGE_GRID)
        self._status.setText(self._grid_status_text(folder, shown=len(items), total=len(under)))
        if pending:
            self._start_thumbs(pending)

    def _item_paths(self) -> list[Path]:
        """Return the path of every photo currently in the list."""
        return [item.path for item in self._items.values()]

    def _grid_status_text(self, folder: Path, *, shown: int, total: int) -> str:
        """Describe the grid's contents, noting when a filter is hiding some of the folder."""
        name = folder.name or folder
        if shown == total:
            return ngettext("{n} photo in {folder}.", "{n} photos in {folder}.", total).format(
                n=total,
                folder=name,
            )
        return _("Showing {shown} of {total} in {folder}.").format(
            shown=shown,
            total=total,
            folder=name,
        )

    def _update_grid_item(self, item: PhotoItem, grid_item: QListWidgetItem) -> None:
        """Refresh a grid thumbnail: image (or placeholder), state badges, and checkbox."""
        # Hold _syncing across the whole refresh: setIcon and setToolTip emit itemChanged just
        # like setCheckState, and an unguarded emission flips _grid_check_toggled, which then
        # swallows the user's next real click on a thumbnail (plus an O(n) tree walk per
        # streamed-in thumbnail).
        was_syncing = self._syncing
        self._syncing = True
        try:
            grid_item.setCheckState(_checked(item.selected))
            key = str(item.path)
            base = self._thumb_cache.get(key, self._placeholder_pixmap)
            badges = thumb_badges(item, has_sidecar=item.path.with_suffix(".xmp").exists())
            grid_item.setIcon(QIcon(_badged_pixmap(base, badges)))
            notes = [_(_BADGE_TEXT[name]) for name in badges]
            if item.status == FAILED and item.error:
                notes.append(item.error)
            grid_item.setToolTip("\n".join([item.path.name, *notes]))
        finally:
            self._syncing = was_syncing

    def _on_thumb_activated(self, item: QListWidgetItem) -> None:
        if self._grid_check_toggled:
            # This click landed on the checkbox: the toggle already happened, and jumping to
            # the detail page would yank the user out of the grid they are working in.
            self._grid_check_toggled = False
            return
        if _selection_modifiers_active():
            # Shift/Cmd clicks are building a multi-selection; navigating away would kill it.
            return
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
        # Stop the thread's event loop once every thumbnail is decoded; without this the idle
        # thread would keep running until the next navigation or the window closed.
        self._thumb_worker.finished.connect(self._thumb_thread.quit)
        self._thumb_thread.start()

    def _stop_thumbs(self) -> None:
        if self._thumb_worker is not None:
            self._thumb_worker.stop()
            self._thumb_worker.deleteLater()
        if self._thumb_thread is not None:
            self._thumb_thread.quit()
            self._thumb_thread.wait()
            self._thumb_thread.deleteLater()
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
        self._existing_source.setText(
            _("(from {sources})").format(sources=sources) if sources else _("(no metadata found)"),
        )
        self._existing_title.setText(item.existing_title or _(_NONE))
        self._existing_description.setPlainText(item.existing_description or _(_NONE))
        _fit_text_height(self._existing_description)
        existing_kw = format_existing_keywords(item.existing_keywords)
        self._existing_keywords.setPlainText(existing_kw or _(_NONE))
        self._title.setText(item.title)
        self._description.setPlainText(item.description)
        self._keywords.setPlainText(keywords_to_text(item.keywords))
        self._hint.setText(item.hint)
        self._show_detail(enabled=True)
        self._update_error_banner(item)
        self._refresh_derived()

    def _update_error_banner(self, item: PhotoItem) -> None:
        """Show the failure reason for a failed photo; hide the banner otherwise."""
        if item.status == FAILED and item.error:
            self._error_banner.setText(
                _(
                    "Generation failed: {error}\nUse 'Retry Failed' to try again, or "
                    "Help > Open Logs for the full traceback.",
                ).format(error=item.error),
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
            self._preview.setText(_("(no preview available)"))
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
            self._diff.setHtml(_("(keywords will not be written)"))
            self._hierarchy.setPlainText(_(_NONE))
            self._details_toggle.setText(_("Keyword changes (not written)"))
            return
        edited = parse_keyword_lines(self._keywords.toPlainText())
        overwrite = self._overwrite.isChecked()
        existing = self._current.existing_keywords
        paths = hierarchy_preview(existing, edited, overwrite=overwrite)
        self._hierarchy.setPlainText(paths or _(_NONE))
        diff = keyword_diff(existing, edited, overwrite=overwrite)
        self._diff.setHtml(_diff_html(diff))
        # A collapsed section still tells the user whether saving changes anything.
        added = sum(1 for _kw, state in diff if state == ADDED)
        removed = sum(1 for _kw, state in diff if state == REMOVED)
        summary = f"+{added} / -{removed}" if added or removed else _("no change")
        self._details_toggle.setText(_("Keyword changes ({summary})").format(summary=summary))

    def _show_detail(self, *, enabled: bool) -> None:
        for widget in (
            self._title,
            self._description,
            self._keywords,
            self._hint,
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
        self._empty_message.setText(_(_EMPTY_PICK) if self._items else _(_EMPTY_START))
        self._right.setCurrentIndex(_PAGE_EMPTY)

    def _commit_current(self) -> None:
        """Copy the visible editable fields back onto the selected item."""
        item = self._current
        if item is None:
            return
        item.title = self._title.text().strip()
        item.description = self._description.toPlainText().strip()
        item.keywords = parse_keyword_lines(self._keywords.toPlainText())
        item.hint = self._hint.text().strip()

    def _save_options(self) -> SaveOptions:
        """Snapshot the save toggles, so a background write never has to read a widget."""
        return SaveOptions(
            write_title=self._write_title.isChecked(),
            write_description=self._write_description.isChecked(),
            write_keywords=self._write_keywords.isChecked(),
            overwrite=self._overwrite.isChecked(),
            backup=self._backup.isChecked(),
            use_sidecar=not self._embed.isChecked(),
        )

    def _write_fields_chosen(self) -> bool:
        """Report whether at least one write toggle (Title/Description/Keywords) is on."""
        return self._save_options().any_field

    def _write_item(self, item: PhotoItem) -> bool:
        """
        Write the item's checked fields to disk; return success.

        Unchecked fields stay as is. This is the inline path for saving one photo, which is a single
        ExifTool call; a whole batch goes through :class:`SaveWorker` instead.
        """
        options = self._save_options()
        job = build_save_job(item, options)
        ok = write_metadata(
            job.path,
            job.keywords,
            description=job.description,
            title=job.title,
            backup=options.backup,
            use_sidecar=options.use_sidecar,
        )
        self._apply_write_result(item, job, ok=ok)
        return ok

    def _apply_write_result(self, item: PhotoItem, job: SaveJob, *, ok: bool) -> None:
        """Fold one finished write into its item: the status word and the Tagged column."""
        item.status = SAVED if ok else FAILED
        if ok:
            # The saved fields are now on the file, so the Tagged column can update without a
            # rescan.
            item.known_fields = (item.known_fields or set()) | job.fields
        self._refresh_status_cell(item)

    def _save_current(self) -> None:
        if self._current is None or self._busy():
            return
        if not self._write_fields_chosen():
            self._status.setText(
                _("Pick at least one field to write (Title, Description, or Keywords)."),
            )
            return
        self._commit_current()
        ok = self._write_item(self._current)
        name = self._current.path.name
        self._status.setText(
            _("Saved {name}.").format(name=name)
            if ok
            else _("Failed to save {name}.").format(name=name),
        )
        self._resort()
        self._update_status()

    def _save_selected(self) -> None:
        if not self._write_fields_chosen():
            self._status.setText(
                _("Pick at least one field to write (Title, Description, or Keywords)."),
            )
            return
        self._commit_current()
        targets = [item for item in self._items.values() if item.selected and item.has_proposal]
        if not targets:
            self._status.setText(_("No checked photos have a proposal to save."))
            return
        self._run_save(targets)

    def _run_save(self, items: list[PhotoItem]) -> None:
        """
        Write *items* on a background thread, reporting each file as it lands.

        Writing hundreds of photos inline blocked the event loop for minutes: the window stopped
        repainting and the OS showed its busy cursor, which is indistinguishable from a hang. The
        work now runs off the UI thread behind the same progress bar and clock a generation run
        uses.
        """
        if self._busy():
            return
        options = self._save_options()
        jobs = [build_save_job(item, options) for item in items]
        self._save_jobs = {str(job.path): job for job in jobs}
        self._saved_ok = 0
        for item in items:
            item.status = WORKING
            self._refresh_status_cell(item)
        self._cancelling = False
        self._set_running(running=True, total=len(jobs))
        self._status.setText(
            ngettext("Saving {n} photo...", "Saving {n} photos...", len(jobs)).format(n=len(jobs)),
        )

        self._save_thread = QThread(self)
        self._save_worker = SaveWorker(
            jobs,
            backup=options.backup,
            use_sidecar=options.use_sidecar,
        )
        self._save_worker.moveToThread(self._save_thread)
        self._save_thread.started.connect(self._save_worker.run)
        self._save_worker.file_done.connect(self._on_save_done)
        self._save_worker.finished.connect(self._on_save_finished)
        self._save_thread.start()

    def _on_save_done(self, path: str, ok: bool) -> None:  # noqa: FBT001 - Qt signal argument
        """Fold one finished write into its item and tick the progress bar."""
        item = self._items.get(path)
        job = self._save_jobs.get(path)
        if item is None or job is None:
            return
        self._apply_write_result(item, job, ok=ok)
        self._saved_ok += int(ok)
        self._advance_progress()
        self._update_status()

    def _on_save_finished(self) -> None:
        """Report the batch tally and put the window back into its idle state."""
        if self._closing:
            # closeEvent already tore the thread down; this queued signal must not touch widgets
            # on a window that is going away.
            return
        total = len(self._save_jobs)
        # A cancelled save leaves the un-written photos marked WORKING; free them so they look
        # ready-again rather than stuck.
        self._reset_working()
        self._save_jobs = {}
        self._resort()
        self._teardown_save_thread()
        # Last, so the tally is not overwritten by the per-file status refresh above.
        self._status.setText(
            ngettext(
                "Cancelled after saving {saved} photo.",
                "Cancelled after saving {saved} photos.",
                self._saved_ok,
            ).format(saved=self._saved_ok)
            if self._cancelling
            else ngettext(
                "Saved {saved} of {n} checked photo.",
                "Saved {saved} of {n} checked photos.",
                total,
            ).format(saved=self._saved_ok, n=total),
        )
        self._cancelling = False

    def _teardown_save_thread(self) -> None:
        """Join the save thread and release both it and its worker."""
        if self._save_thread is not None:
            self._save_thread.quit()
            self._save_thread.wait()
            self._save_thread.deleteLater()
            self._save_thread = None
        if self._save_worker is not None:
            self._save_worker.deleteLater()
        self._save_worker = None
        self._set_running(running=False)

    # --- generation ------------------------------------------------------------------------

    def _generate(self, *, use_cache: bool = True) -> None:
        selected = [item for item in self._items.values() if item.selected]
        if not selected:
            self._status.setText(_("Check at least one photo first."))
            return
        self._run_generation(selected, use_cache=use_cache)

    def _generate_current(self, *, use_cache: bool = True) -> None:
        if self._current is None:
            self._status.setText(_("Open a photo to generate it."))
            return
        self._run_generation([self._current], use_cache=use_cache)

    def _retry_failed(self) -> None:
        failed = [item for item in self._items.values() if item.status == FAILED]
        if not failed:
            self._status.setText(_("No failed photos to retry."))
            return
        self._run_generation(failed)

    def _run_generation(self, items: list[PhotoItem], *, use_cache: bool = True) -> None:
        if self._busy() or not items:
            return
        # Fold the open photo's visible edits (most importantly a just-typed hint) into its
        # state before the run reads it.
        self._commit_current()
        for item in items:
            item.status = WORKING
            self._refresh_status_cell(item)
        current = self._current
        if current is not None and current in items:
            # Clear a stale failure banner the moment its photo is re-queued.
            self._update_error_banner(current)
        self._cancelling = False
        self._set_running(running=True, total=len(items))
        self._status.setText(
            ngettext("Generating {n} photo...", "Generating {n} photos...", len(items)).format(
                n=len(items),
            ),
        )

        self._thread = QThread(self)
        self._worker = GenerateWorker(
            self._provider_name(),
            self._model.currentText().strip(),
            self._url.text().strip() or None,
            [item.path for item in items],
            api_key=self._api_key_value(),
            cache_file=self._active_cache_file() if use_cache else None,
            output_language=self._output_language,
            hints={str(item.path): item.hint for item in items if item.hint},
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
        self._session_tagged.add(str(item.path))
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
        """
        Ask whichever run is in flight to stop after the photo it is on.

        One Cancel button covers both runs (they never overlap): a model call and a metadata write
        are both uninterruptible once started, so cancelling takes effect at the next photo.
        """
        worker = self._worker or self._save_worker
        if worker is None:
            return
        self._cancelling = True
        worker.stop()
        self._cancel_button.setEnabled(False)
        self._status.setText(_("Cancelling after the current photo finishes..."))

    def _on_generate_finished(self) -> None:
        if self._closing:
            # closeEvent already tore the thread down; this queued signal arriving afterwards
            # must not touch widgets on a window that is going away, or tear down twice.
            return
        # A cancelled run leaves the un-started photos marked WORKING; free them so they look
        # queued-again rather than stuck, and report what actually got done.
        reset = self._reset_working()
        if self._cancelling:
            self._status.setText(
                ngettext(
                    "Cancelled. {n} photo not generated.",
                    "Cancelled. {n} photos not generated.",
                    reset,
                ).format(n=reset),
            )
        else:
            self._status.setText(_("Generation finished."))
        self._cancelling = False
        self._resort()
        self._teardown_thread()

    def _reset_working(self) -> int:
        """
        Free any photo still marked WORKING; return how many were reset.

        A photo that already has a proposal goes back to READY (the proposal is still there to
        save), anything else back to PENDING. Used after a cancelled generation or save, so an
        interrupted photo reads as queued-again rather than stuck.
        """
        reset = 0
        for item in self._items.values():
            if item.status == WORKING:
                item.status = READY if item.has_proposal else PENDING
                self._refresh_status_cell(item)
                reset += 1
        return reset

    def _busy(self) -> bool:
        """
        Whether a generation or save run is in flight.

        The two never overlap.
        """
        return self._thread is not None or self._save_thread is not None

    def _has_failures(self) -> bool:
        """Report whether any photo is currently in the failed state."""
        return any(item.status == FAILED for item in self._items.values())

    def _set_running(self, *, running: bool, total: int = 0) -> None:
        """Switch the window between idle and busy: buttons, progress bar, and the clock."""
        self._generate_button.setEnabled(not running)
        self._generate_one_button.setEnabled(not running)
        self._save_button.setEnabled(not running)
        self._save_selected_button.setEnabled(not running)
        self._retry_button.setEnabled(not running and self._has_failures())
        self._test_button.setEnabled(not running)
        self._cancel_button.setEnabled(running)
        if running:
            self._start_progress(total)
        else:
            self._stop_progress()

    def _start_progress(self, total: int) -> None:
        """Show the bar and start the clock for a run over *total* photos."""
        self._progress.setRange(0, total)
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._run_started = time.monotonic()
        self._timing.setVisible(True)
        self._update_timing()
        self._timing_timer.start()

    def _stop_progress(self) -> None:
        """Hide the bar and the clock; the run is over."""
        self._timing_timer.stop()
        self._run_started = None
        self._progress.setVisible(False)
        self._timing.setVisible(False)
        self._timing.clear()

    def _update_timing(self) -> None:
        """
        Repaint the elapsed/remaining readout from the progress bar's own counters.

        Driven by a timer as well as by each finished photo, so the elapsed time keeps moving while
        a single slow photo is in flight. That movement is the point: it is what tells the user the
        program is working rather than wedged.
        """
        if self._run_started is None:
            return
        self._timing.setText(
            progress_timing_text(
                self._progress.value(),
                self._progress.maximum(),
                time.monotonic() - self._run_started,
            ),
        )

    def _advance_progress(self) -> None:
        """Tick the run progress bar for one finished (or failed) photo."""
        if self._run_started is None:
            return
        self._progress.setValue(self._progress.value() + 1)
        self._update_timing()

    def _teardown_thread(self) -> None:
        # deleteLater releases the C++ side of the thread and the (parentless, moved) worker;
        # dropping only the Python refs would leak both once per generation run.
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait()
            self._thread.deleteLater()
            self._thread = None
        if self._worker is not None:
            self._worker.deleteLater()
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
            QMessageBox.warning(self, _("Could not list models"), str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        current = self._model.currentText()
        self._model.clear()
        self._model.addItems(rank_vision_models(models))
        self._model.setCurrentText(current)
        self._status.setText(
            ngettext(
                "Found {n} model on {provider}.",
                "Found {n} models on {provider}.",
                len(models),
            ).format(n=len(models), provider=PROVIDER_LABELS.get(self._provider_name(), "")),
        )

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
        box.setWindowTitle(_("Connection check"))
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
        label = _(_STATUS_LABEL[item.status]) if _STATUS_LABEL[item.status] else ""
        if item.status == READY and item.from_cache:
            label = _("ready (cached)")
        leaf.setText(_COL_STATUS, label)
        leaf.setData(_COL_STATUS, _STATUS_RANK_ROLE, status_sort_rank(item.status))
        color = _STATUS_COLOR.get(item.status)
        leaf.setData(
            _COL_STATUS,
            Qt.ItemDataRole.ForegroundRole,
            QBrush(color) if color is not None else None,
        )
        # Surface the failure reason on hover so it is discoverable straight from the tree. Model
        # errors can be a paragraph long, hence the wrap.
        leaf.setToolTip(_COL_STATUS, wrap_tooltip(item.error) if item.status == FAILED else "")
        if item.known_fields is not None:
            leaf.setText(_COL_TAGGED, tagged_summary(item.known_fields))
            leaf.setToolTip(_COL_TAGGED, wrap_tooltip(tagged_tooltip(item.known_fields)))

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
        if self._thread is None and self._save_thread is None:
            self._retry_button.setEnabled(self._has_failures())

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt override.
        """
        Stop any in-flight generation or save before closing.

        Asking the workers to stop first means closing mid-run only waits for the photo currently in
        flight, not the whole batch. Waiting on the threads keeps a running QThread from being
        destroyed under it.
        """
        self._closing = True
        if self._worker is not None:
            self._worker.stop()
        if self._save_worker is not None:
            self._save_worker.stop()
        self._stop_thumbs()
        self._stop_scan()
        self._teardown_thread()
        self._teardown_save_thread()
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
        box.setWindowTitle(_("Anonymous usage telemetry"))
        box.setText(_("Photo Tagger sends anonymous usage stats to guide development."))
        box.setInformativeText(
            _(
                "Collected: model name, batch size, OS, CPU/GPU model, RAM size, timing, and "
                "anonymous crash reports (error type and code location only).\n"
                "Never: photos, file paths, filenames, tags, error messages, or personal data.\n\n"
                "You can turn this off now, or anytime from Settings > Send Anonymous Telemetry.",
            ),
        )
        keep = box.addButton(_("Keep Enabled"), QMessageBox.ButtonRole.AcceptRole)
        box.addButton(_("Turn It Off"), QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(keep)
        box.exec()
        if box.clickedButton() is not keep:
            # Unchecking fires _on_telemetry_toggled, which persists the choice.
            self._telemetry_action.setChecked(False)

    def _emit_telemetry(self) -> None:
        """
        Fire a best-effort GUI usage beacon on close; never raises.

        The process exits right after closeEvent, so a fire-and-forget daemon thread would be killed
        mid-send and the beacon silently lost. block=True waits for the flush; a healthy send is
        tens of milliseconds and the bad case (no network) is capped by emit's timeout.
        """
        telemetry.emit(
            telemetry.RunInfo(
                interface="gui",
                provider=self._provider_name(),
                model=self._model.currentText().strip(),
                batch_size=len(self._session_tagged),
                duration_seconds=time.monotonic() - self._session_start,
                output_language=self._output_language,
                ui_language=i18n.current_language(),
                file_types=telemetry.file_types_summary(Path(key) for key in self._session_tagged),
                success_count=len(self._session_tagged),
            ),
            enabled=self._telemetry_enabled,
            block=True,
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
    BADGE_FAILED: gettext_noop("generation failed"),
    BADGE_SAVED: gettext_noop("saved"),
    BADGE_UNSAVED: gettext_noop("generated, not saved yet"),
    BADGE_METADATA: gettext_noop("already has metadata"),
    BADGE_SIDECAR: gettext_noop("has an XMP sidecar"),
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


def _selection_modifiers_active() -> bool:
    """Report whether a multi-select modifier (Shift or Ctrl/Cmd) is held right now."""
    modifiers = QApplication.keyboardModifiers()
    return bool(
        modifiers & (Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier),
    )


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
    return "<br>".join(rows) or _("(no change)")


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
    # Resolve the UI language before any widget is built: strings are baked at construction.
    # Qt's system locale is the hint of last resort; it knows the OS language even when no LANG
    # is exported (a Finder/Dock launch).
    configured = load_config().get("language")
    language = i18n.activate(
        str(configured) if configured else None,
        system_hint=QLocale.system().name(),
    )
    if language != "en":
        # Also translate Qt's own stock strings (file dialogs, standard buttons). Best-effort:
        # PySide6 wheels may not ship every qtbase catalog.
        qt_translator = QTranslator(app)
        translations_dir = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
        if qt_translator.load(QLocale(language), "qtbase", "_", translations_dir):
            app.installTranslator(qt_translator)

    def _gui_telemetry_enabled() -> bool:
        # The window's live toggle is authoritative once it exists; before that, fall back to the
        # persisted GUI preference or the config default (emit_crash enforces the env opt-outs).
        pref = telemetry.read_gui_pref()
        return load_defaults().telemetry.enabled if pref is None else pref

    # Qt swallows exceptions raised inside slots (they reach sys.excepthook and the loop keeps
    # running), so an unhandled slot crash never propagates out of app.exec(). Chain a hook that
    # fires an anonymous crash beacon (type + in-app code location, never the message) first.
    previous_hook = sys.excepthook

    def _crash_hook(exc_type: type[BaseException], exc: BaseException, tb: object) -> None:
        telemetry.emit_crash(exc, interface="gui", enabled=_gui_telemetry_enabled(), block=False)
        previous_hook(exc_type, exc, tb)  # type: ignore[arg-type]

    sys.excepthook = _crash_hook

    try:
        window = MainWindow()
        window.show()
        window.maybe_show_telemetry_notice()
        return app.exec()
    except Exception as exc:
        # A crash outside the event loop (startup, teardown) kills the process; flush the beacon.
        telemetry.emit_crash(exc, interface="gui", enabled=_gui_telemetry_enabled(), block=True)
        raise
