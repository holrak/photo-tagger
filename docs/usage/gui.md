---
icon: lucide/app-window
---

# Desktop GUI

photo-tagger ships an optional desktop GUI for a point-and-click, **review-before-write** workflow
over the same pipeline the CLI uses. You drag in photos, generate proposals with the vision model,
then inspect and edit each photo's title, description, and keywords before saving.

The GUI is purely additive: it reads existing metadata, runs the model, and writes with ExifTool
through the same building blocks as the CLI, so what you learn in the
[CLI reference](cli-reference.md) maps directly onto it. See
[Architecture](../architecture/index.md#frontends) for how the frontends share one pipeline.

## Install and launch

The GUI lives behind the `gui` extra so the base CLI stays free of the Qt dependency:

```bash
uv tool install 'photo-tagger[gui]'
photo-tagger gui
```

On conda-forge it is a separate package, `photo-tagger-gui`, that bundles PySide6:

```bash
conda install -c conda-forge photo-tagger-gui
photo-tagger gui
```

Run without the extra (or without the `photo-tagger-gui` package) and the command prints an install
hint instead of a traceback:

```console
$ photo-tagger gui
The desktop GUI needs PySide6, which is not installed.
Install the optional extra with:
    pip install 'photo-tagger[gui]'
```

!!! note

    The GUI needs the same prerequisites as the CLI: ExifTool on your `PATH` and a reachable model
    server. Use **Test connection** in the window to verify both before a run.

## The workflow

### 1. Add photos

Drag photos or folders anywhere onto the window, or use the **Add photos...** button: a click opens
the file picker, and its arrow menu holds **Add Folder...** plus the folder-scan options (the **File
types** list, comma-separated and case-insensitive, and **Include subfolders**), exactly like the
CLI's `--ext` and `--recursive`. Photos appear in a nested tree on the left: subfolders are grouped
under their parent folder, so a deep shoot stays organized rather than flattened.

!!! note "File types are case-insensitive but variant-aware"

    `jpg` matches `.JPG` (case-insensitive), but it does **not** cover `.jpeg`: those are distinct
    extensions, so the default list includes both. Hover any control for a tooltip explaining it.

### 2. Choose what to process

Every folder and file has a checkbox; uncheck a folder to exclude everything under it, or uncheck
individual files. Only checked photos are generated. Rows support multi-selection the usual way
(++shift++-click for a range, ++cmd++/++ctrl++-click to add single rows, ++shift++ + arrow keys from
the keyboard); right-clicking inside the selection then offers **Check / Uncheck / Generate /
Remove** for all of it at once, and this works in the thumbnail grid too. To take items off the list
entirely (rather than just deselect them), select them and click **Remove**, press ++delete++ /
++backspace++, or use **Remove From List** in the right-click menu; **File > Clear List** empties
the whole list.

The tree has four resizable, sortable columns (click a header to sort; click again to reverse;
folders stay grouped above their sibling files either way):

- **Photos**: the file name.
- **Type**: the extension, with `+xmp` appended when an XMP sidecar sits next to the file.
- **Status**: blank (pending), `working...`, `ready` (or `ready (cached)` when the proposal was
    replayed from the result cache), `saved ✓` (green), or `failed ✗` (red). Sorting uses the
    processing stage, not the label text, so clicking it groups all the failures or all the ready
    photos together.
- **Tagged**: which metadata the file already carries, filled in by a background scan after you add
    photos: `T` title, `D` description, `K` keywords, `-` for none (hover for the full list).

The **Select** menu checks and unchecks photos in bulk, so you do not have to hunt through a large
list by hand:

- **Check All** / **Uncheck All** flip every checkbox at once.
- **Uncheck Already Tagged** opens a menu of criteria for what counts as "already done", the GUI's
    field-aware [`--skip-tagged`](cli-reference.md). It reads the metadata in one pass (so a large
    folder pauses briefly) and unchecks the matching photos:
    - **Has any metadata** unchecks any photo that has a title, description, *or* keywords (the broad,
        original behavior).
    - **Has a title**, **Has a description**, **Has keywords**, or **Has a title and a description**
        target specific fields. The combined criteria require *all* of their fields, so a photo that
        only has keywords survives **Has a title and a description** and stays selected, which is what
        you want when filling in the title and description on photos that are missing them.
- **Uncheck From Skip List...** lets you pick a plain-text file listing photos to skip, one per line
    (by bare filename or full path, `#` comments allowed), and unchecks the ones it names. This is
    the GUI's [`--skip-from`](cli-reference.md), and it pairs with the CLI's
    `--append-to-skip-file`: point it at the list a CLI run wrote to resume the same work in the
    window.

Deselecting only unchecks: the photos stay in the list, so you can see what was skipped and re-check
any of them. The status bar reports how many photos were deselected and how many are still selected.

Selecting a **folder** (rather than a file) shows a **thumbnail grid** of its photos on the right,
like a contact sheet. Thumbnails load in the background, so a large folder of RAW files stays
responsive. Click any thumbnail to open that photo's detail, tick its checkbox to select or deselect
it (kept in sync with the tree), or right-click it for the same menu as its row in the tree (retry,
skip cache, reveal, remove). Each thumbnail carries small badges (hover for the explanation): the
top-right one tracks the lifecycle (red ✗ failed, green ✓ saved, indigo dot for
generated-but-not-saved), and the top-left ones flag a photo that already has metadata (`M`) or an
XMP sidecar (`S`).

### 3. Generate proposals

Pick a **Provider** (Ollama, LM Studio, llama.cpp, or OpenAI) and a **Model** in the header. Press
**Refresh** to query the provider for the models it currently serves and pick from the dropdown
instead of typing; likely vision-capable models are listed first. **Connection...** opens the
settings that rarely change: a custom **Base URL**, a masked **API key** (leave it blank to fall
back to the provider's environment variable), and the **Test connection** check for ExifTool and the
model.

**Generate selected** (in the bottom bar) then runs the model on the checked photos on a background
thread, building the same contextual prompt as the CLI (existing keywords, location, GPS, camera).
Results stream in, the tree status updates per photo, and a progress bar in the bottom bar counts
the batch down.

Results are **cached by default** (see [Configuration](#configuration)): re-running a batch after a
crash, or generating a folder you partly processed before, reuses the earlier answers for unchanged
photos instead of calling the model again. Toggle **Settings > Cache AI Results** to turn that off
for the session, or use the arrow on either **Generate** button (or a photo's right-click menu) for
a one-time **Skip Cache** run.

To regenerate a single photo without touching your selection, open it and press **Generate this
photo** in the detail pane, or right-click it in the tree and choose **Generate** (a failed photo
shows **Retry Generation** there instead). If several photos ended up `failed ✗`, **Retry failed**
in the bottom bar re-runs the model on all of them at once (the button stays disabled while nothing
has failed).

To stop a run early, press **Cancel** (next to *Generate selected*). The photo already in flight
finishes (a model request cannot be interrupted mid-call), then the run stops and the un-started
photos return to `pending` so you can resume them later with another **Generate selected**. Anything
already generated keeps its proposal.

### 4. Review, edit, and save

Click a photo to open it on the right. The detail pane is **side-by-side** for easy comparison: an
**Existing** column (read-only) next to a **New (editable)** column.

- a **preview** (RAW files are decoded just like a real run),
- **Existing** vs **New** Title, Description, and Keywords lined up row by row, with the New side
    editable and seeded from the proposal. The Existing header notes where that metadata was read
    from (the image file, an XMP sidecar, or both). The description boxes grow with their content
    instead of reserving space, and existing keyword hierarchies display in the same `<` notation
    you type,
- a collapsible **Keyword changes** section. Its header always summarizes what a save would do
    (`+3 / -1`, or `no change`); expand it for the colored diff (green added, red struck-through
    removed, grey unchanged) and the resulting keyword **tree**, drawn with `tree`-style branch
    guides (`├─`/`└─`).

Keywords support hierarchy with `<` (specific to general), for example `Eagle<Bird<Animal`; the
summary, diff, and tree update live as you edit. Adjust anything, then press **Save this photo** to
write just the open one, or **Save selected** (bottom bar) to write the **checked** photos that have
a proposal. The save scope matches *Generate selected* (the same checkboxes), so check everything to
save everything.

The arrow on either **Save** button opens the save options, which choose what every save writes
(both buttons share them, and their tooltips always spell out the current choice):

- **Write Title**, **Write Description**, **Write Keywords** (all on by default), the GUI's
    equivalent of the CLI's `--no-write-title` / `--no-write-description` / `--no-write-keywords`.
    Uncheck one to leave that field on the photo untouched, for example uncheck **Write Keywords**
    to refresh only the title and description while keeping a curated Lightroom keyword list as is
    (turning it off also disables **Overwrite Existing Keywords**, since there is nothing to write).
- **Overwrite Existing Keywords** replaces the existing keywords instead of merging the new ones in.
- **Embed in Photo** writes into the image file instead of the default XMP sidecar.

You can edit and save a photo even without generating a proposal first: the editable fields then
start from the existing values.

!!! tip "API keys: field or environment"

    The **Connection...** dialog has a masked **API key** field. Leave it blank to use the provider's
    environment variable (`OPENAI_API_KEY`, `LM_STUDIO_API_KEY`, or `OLLAMA_API_KEY`), which keeps the
    secret out of the app entirely; or type a key to use it for this session only. A typed key is held
    in memory for the run and is never written to disk (not even by **Save Settings as Defaults**).
    OpenAI requires a key; local Ollama and LM Studio servers usually do not. To set it from the
    environment instead, launch with, for example, `OPENAI_API_KEY=sk-... photo-tagger gui`.

### 5. When a photo fails

A photo that the model could not process is marked `failed ✗` in the status column. To find out why:

- **Hover** the `failed ✗` cell for a tooltip with the error, or
- **open** the photo: a red banner above the preview shows the reason (for example
    `model unreachable` or a decode error).

Once you have addressed the cause (start the model server, fix the URL, free up memory), click
**Retry failed** to re-run every failed photo, or right-click one and choose **Retry Generation** to
retry just that one. A successful retry clears the banner and flips the status back to `ready`.

For the full traceback behind a failure, use **Help > Open Logs**. The GUI writes a timestamped,
rotating log file to `~/.photo-tagger/logs/` on every run and the action opens that folder in your
file browser.

### Export a CSV report

**File > Export CSV Report...** writes a spreadsheet with **one row per photo in the list**, the
same report the CLI's [`--csv-file`](cli-reference.md#csv-report) produces. Each row gathers
everything the GUI knows about a photo: its status, the working title/description/keywords (the
keywords as a save would write them, honoring the **Overwrite** toggle), the
title/description/keywords already on the file, the camera/location EXIF read when it was generated,
and the per-photo token usage and timing. Photos you have not generated yet still get a row, with
the generated and usage columns left blank.

Any unsaved edits in the open photo are folded into its row first, so the export reflects exactly
what you see. Pick a destination in the save dialog (a `.csv` suffix is added if you omit one); the
status bar confirms how many photos were written. This is handy for reviewing a batch in a
spreadsheet or sharing the results without opening every photo.

## Configuration

The GUI reads the same TOML config file and environment variables as the CLI and pre-fills the
provider, model, URL, extensions, and save options from them. It can also write that file:
**Settings > Save Settings as Defaults...** updates the config file in effect (or creates
`~/.config/photo-tagger/config.toml`) with the current provider, model, URL, file types, recursion,
and save options. The save **merges**: comments, ordering, and every setting the GUI does not manage
are preserved, and the API key is never written. For everything the GUI does not surface (prompt
file, sampling, workers, filters), **Settings > Edit Config File...** opens the file in your editor.
See [Configuration](../getting-started/configuration.md) for the format.

Generated results are cached in the same SQLite format as the CLI's
[`--cache-file`](cli-reference.md): the GUI uses the configured `cache_file` if the config sets one,
and `~/.photo-tagger/cache.sqlite` otherwise. **Settings > Cache AI Results** turns the cache off
for a session; a photo's right-click menu offers **Generate (Skip Cache)** for one-off fresh
results.

The window surfaces the most common options; an ExifTool backup is always kept and inference
settings use their defaults. For the full set of flags (custom prompts, date-range filters,
sampling, logging), use the [CLI](cli-reference.md).

## Limitations

!!! warning

    Cancelling (or closing the window mid-run) stops at the next photo boundary: the photo **already in
    flight runs to completion** before the run halts, because a model request cannot be interrupted
    mid-call. With a slow model that one photo can take a while.

- Loading a photo's preview and existing metadata is synchronous, so selecting a large RAW file may
    pause briefly the first time (results are cached afterwards).
- On macOS, `./packaging/build_macos_app.sh` builds a double-clickable (unsigned) `Photo Tagger.app`
    with PyInstaller; elsewhere launch the GUI with `photo-tagger gui`.

For headless machines, scripting, scheduling, or piping results into other tools, use the CLI with
[`--json`](cli-reference.md#display); the GUI is meant for interactive, local use.
