# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `--vocabulary PATH`: restrict generated keywords to a controlled vocabulary, so a run cannot seed
  a curated Lightroom catalog with near-duplicates of keywords it already has. Reads either shape of
  Lightroom's *Metadata > Export Keywords*, the `.txt` (indented, with `{synonyms}`) and the `.csv`
  (the same list behind four option columns), or a plain list of terms and `Animal|Bird|Osprey`
  paths. A comment needs a space after the hash, so a keyword such as `#Diversity` stays a keyword.
  Matching ignores case, punctuation, and plurals, with a conservative fuzzy pass for typos, and
  every match is rewritten to the file's own spelling **and** hierarchy. The vocabulary is also
  listed in the prompt, so the model prefers those terms in the first place; that listing is part of
  the cache namespace, so swapping files starts a fresh cache slice. Past 5,000 terms the run warns
  that the fuzzy pass is off and the prompt listing is truncated.
- `--vocabulary-strict`: drop generated keywords the vocabulary does not cover instead of writing
  them as-is. The run summary gains `vocabulary_mapped` (how many keywords were rewritten) and
  `vocabulary_dropped` (each rejected term and how often it came up), so the vocabulary can grow on
  purpose rather than by accident.
- `--session-gap MINUTES`: group the batch into shoots (by EXIF capture time, falling back to file
  mtime) and harmonize each one, so forty frames of the same bird stop landing in the catalog as
  `Osprey`, `Ospreys`, `Bird|Osprey`, and `Wildlife|Raptor|Osprey`. A session is analyzed in full
  before anything is written; its own output then becomes its vocabulary, with the majority spelling
  and the majority hierarchy winning. Deriving it from the finished results rather than nudging the
  model keeps it deterministic: the same batch harmonizes the same way at any `--workers` setting.

## [0.7.0] - 2026-08-08

### Added

- GUI: the bottom bar now shows the time **elapsed** and the estimated time **left** next to the
  progress bar, for both generating and saving (`2:30 elapsed · 8:10 left`).
- GUI: a **Keep ExifTool Backup** toggle in the Save options menu (the CLI's `--no-backup-xmp`).
  Saving used to always leave a `*_original` copy of every photo behind, which on a large batch
  doubles the disk space used. It defaults to on and persists via *Save Settings as Defaults*.

### Changed

- GUI: **Save Selected** now writes on a background thread (sharing one ExifTool process for the
  batch) instead of blocking the window. Saving a large batch used to freeze the interface behind
  the OS busy cursor with no sign of progress, which looked like a crash. It now reports per photo,
  and **Cancel** stops it after the file in flight.

## [0.6.0] - 2026-07-11

### Added

- Anonymous crash reports: an unhandled exception sends the exception *type* and its code location
  inside photo-tagger (`module:function:line`), never the error message (messages can embed paths).
  Same opt-outs as the usage telemetry; capped per process. The one-time telemetry notice re-shows
  once for existing installs so the expanded disclosure is seen.
- Telemetry now reports coarse hardware facts (CPU/GPU model, logical cores, RAM in whole GB; probed
  once and cached), per-run outcome counters (successes, failures, cache hits, retry recoveries,
  workers, token totals, model time, dry-run flag), and failure buckets (fixed labels like `timeout`
  or `metadata-write`) so breakage is measurable, not anecdotal.
- The telemetry dashboard grew sections for adoption, usage, hardware & performance, and reliability
  & crashes, plus a global interface filter (all/CLI/GUI).
- The inference cache prunes entries older than 180 days when it opens, so `cache.sqlite` stops
  growing forever (every settings change orphans its old namespace).
- A short pause before the retry pass, so an overloaded model server is not immediately re-hit.

### Changed

- `--append-to-skip-file` records full paths instead of bare filenames, so duplicate camera names
  across folders (`A/IMG_0001.CR3`, `B/IMG_0001.CR3`) can no longer make a resumed run silently skip
  unprocessed photos. Old name-only skip files keep working.
- Config-file values now pass through the same conversion and validation as CLI flags, so a mistyped
  value fails with a clean error at startup instead of a traceback mid-run.
- Photos with identical pixel content in one concurrent run share a single model call instead of
  each paying their own.
- CSV reports are written as `utf-8-sig` so Excel renders accented titles and keywords correctly.

### Fixed

- Passing any CLI flag from an option group no longer silently resets the group's other config-file
  values to built-in defaults (e.g. `--temperature` used to revert a configured `max_tokens`).
- `--dry-run` no longer appends previewed photos to `--append-to-skip-file`, which made later real
  runs skip photos that never got metadata.
- The description mirror is written to `XMP-tiff:ImageDescription`; the previous `XMP-exif` tag was
  silently rejected by ExifTool, so the mirror never reached the file.
- Keyword capitalization no longer corrupts acronyms and apostrophes (`NYC` stays `NYC` instead of
  becoming `Nyc`; `bird's nest` no longer becomes `Bird'S Nest`).
- One corrupt or zero-byte photo no longer makes the whole folder read as untagged (and get
  re-tagged): the batched ExifTool reads salvage the healthy files, and `--skip-tagged` and the
  GUI's Tagged column now distinguish "could not read" from "no metadata".
- Windows: file locking works again (the PID note into the held lock file raised on every
  acquisition), and the batched metadata reads no longer miss every file over path-separator
  differences.
- Ctrl-C during concurrent batch start-up is honored instead of grinding through every queued photo;
  never-submitted photos are reported as pending.
- A single over-long or blank keyword from the model no longer burns the whole validation retry
  budget (full vision calls); items are cleaned instead.
- GUI: streamed-in thumbnails no longer swallow the next grid click; a non-domain error during
  generation no longer leaves the window stuck in the running state; removing photos prunes the
  visible grid and thumbnail cache; bulk actions no longer kick you out of the photo or folder being
  reviewed.
- The `gui` command only suggests installing the `[gui]` extra when PySide6/shiboken6 is actually
  missing; other import errors surface as themselves.
- A failed cache initialization no longer leaks its SQLite connection.

## [0.5.0] - 2026-07-09

### Added

- The GUI is translated at runtime (gettext) and ships a complete Brazilian Portuguese (`pt_BR`)
  translation. Pick a language with **Settings > Language**, the `language` config key, or
  `PHOTO_TAGGER_LANG`; the default follows the OS locale.
- New `llamacpp` provider for llama.cpp's `llama-server`. Default endpoint
  `http://localhost:8080/v1`; env vars `LLAMA_CPP_BASE_URL` and `LLAMA_CPP_API_KEY` (the key is only
  needed when `llama-server` was started with `--api-key`).
- Anonymous, opt-out usage telemetry with a one-time first-run notice. Disable it with
  `--no-telemetry`, `PHOTO_TAGGER_NO_TELEMETRY=1` (or `DO_NOT_TRACK=1`), `enabled = false` under
  `[telemetry]` in the config file, or the GUI's **Settings > Send Anonymous Telemetry** toggle.
- GUI redesign: split **Generate**/**Save** buttons with option menus, thumbnail badges,
  multi-selection with bulk actions, tree columns for file type, status, and already-tagged,
  collapsible keyword-change details, reveal-in-file-manager on Windows, Linux, and macOS, and CSV
  export.
- The GUI now caches results by default (sharing the CLI's cache format), with skip-cache actions
  for forcing a fresh generation.
- **Settings > Save Settings as Defaults** merges into the existing config file instead of rewriting
  it, preserving comments and unknown keys.
- Windowed `photo-tagger-gui` entry point, so the desktop app launches without a console window.
- `exiftool_path` config key and `PHOTO_TAGGER_EXIFTOOL` env var for ExifTool installs not on
  `PATH`. A GUI launched from Finder also inherits the login shell's `PATH` automatically.
- `packaging/build_macos_app.sh` builds a standalone, double-clickable macOS app with PyInstaller.

### Changed

- Console logging is capped at `INFO` from import (so nothing above it leaks before the CLI flags
  apply), and file logs are serialized as JSON lines.
- System prompt now demands English-only, single-script output and at most one hierarchy chain per
  keyword.

### Fixed

- The GUI's telemetry beacon is flushed before the process exits instead of being lost.
- The GUI result cache is keyed on the image content hash, so **Embed in Photo** no longer
  invalidates it.
- Hierarchical keywords written with `>` separators are parsed as hierarchies instead of being
  mangled, and model keywords are deduplicated.
- Ctrl-C during a concurrent batch no longer miscounts photos that finished while the pool drained.

## [0.4.0] - 2026-06-26

### Changed

- The `--cache-file` inference cache is now keyed on the image data only (ExifTool's
  `ImageDataHash`) instead of a hash of the whole file. Writing metadata into a photo no longer
  changes the key, so a rerun over the same folder hits the cache even after `--embed-in-photo`
  wrote tags on the first pass. Formats ExifTool cannot hash that way fall back to the whole-file
  hash. Existing cache files are simply repopulated on the next run.

## [0.3.0] - 2026-06-22

### Added

- Optional desktop GUI (`photo-tagger gui`), via the `gui` extra
  (`pip install 'photo-tagger[gui]'`). A PySide6 review-before-write frontend: drag in photos or
  folders, generate proposals, then review and edit each photo's title, description, and keywords
  (with a live Lightroom-hierarchy preview) before saving. PySide6 is imported lazily, so the base
  CLI never depends on Qt.
- New `openai` provider for any hosted OpenAI-compatible endpoint. Set `OPENAI_BASE_URL` / `--url`
  and `OPENAI_API_KEY` / `--api-key`; fails fast when no key is configured.
- New `photo-tagger doctor` command: a pre-flight check that ExifTool is on PATH and the provider is
  reachable and serves the requested model, exiting non-zero on failure.
- `--csv-file PATH` on the `tag` command writes a per-photo CSV report (existing/written metadata,
  camera, location, GPS, usage, timing), alongside `--summary-file` and `--json`.
- `--write-keywords` / `--no-write-keywords` (default on) refreshes the title and description while
  leaving existing keywords on disk untouched.
- A PEP 561 `py.typed` marker so downstream projects can consume the package's type hints.

### Changed

- Backends now live in a `photo_tagger.providers` registry; adding a backend is a single entry.
- Existing keywords use a typed `KeywordSet` value object instead of a bare `dict[str, list[str]]`.
- The package version is read from installed distribution metadata, so `pyproject.toml` is the only
  code-side source of truth.
- CLI option groups moved out of `main.py` into `photo_tagger.cli_options`.

### Fixed

- Hierarchical keywords are generated reliably again: the model schema now has a dedicated
  `hierarchies` field for taxonomy chains (`Golden Eagle<Bird of Prey<Animal`) instead of expecting
  `<` embedded in the flat `keywords` list, which the model had stopped doing. (The CLI cache is
  keyed on the user prompt only, so delete a stale `--cache-file` to pick this up on
  already-processed photos; the GUI never caches.)

## [0.2.2] - 2026-05-30

### Added

- MIT License. The `LICENSE` file now ships in the sdist via PEP 639 `license-files`; the deprecated
  `License :: OSI Approved :: MIT License` classifier was dropped.

### Changed

- System prompt now forbids emitting the camera body, lens model, or capture timestamp as keywords;
  these describe equipment, not subject content. Earlier runs sometimes copied literal EXIF strings
  (e.g. `Canon Eos R5M2`, `Rf200-800Mm F6.3-9 Is Usm`) into the keyword list.

### Fixed

- `create_agent` no longer has a code path where the provider could be left unbound for a value
  outside the supported set. Provider construction moved into `_build_provider`, which returns from
  each branch and ends in `assert_never`, keeping the match exhaustive for the type checker.

## [0.2.1] - 2026-05-27

### Fixed

- `AttributeError: 'str' object has no attribute 'parent'` when writing the summary file after a
  successful run if `summary_file` (or any other `Path` field) was set via the TOML config rather
  than the CLI. `apply_overrides` now coerces string TOML values to `Path` for fields annotated as
  `Path` or `Path | None`.

## [0.2.0] - 2026-05-26

### Added

- TOML config file. Search order: `$PHOTO_TAGGER_CONFIG`, `./.photo-tagger.toml`,
  `~/.config/photo-tagger/config.toml`. CLI flags still win. See
  [`.photo-tagger.example.toml`](.photo-tagger.example.toml) for a template.
- `--workers N` for thread-pool concurrency (default 1; the model server is usually the bottleneck).
- `--cache-file PATH` SQLite cache of model outputs keyed by image content hash plus model, prompt,
  and sampling settings. Reruns skip the model entirely when nothing relevant changed. WAL mode is
  enabled.
- `--lock-file PATH` exclusive file lock that refuses to start if another `photo-tagger` already
  holds it. Cross-platform (Linux, macOS, Windows).
- `--summary-file PATH` writes a JSON run summary on completion (success counts, failed files, token
  usage, wall time). Atomic write; parent dir is created.
- `--json` emits one NDJSON line per processed photo on stdout. Logs and progress stay on stderr so
  `| jq` works.
- `--skip-tagged` skips files whose image or sidecar already has keywords, description, or title
  (catches photos tagged in Lightroom or by hand).
- `--append-to-skip-file PATH` records each successful filename so a later run with
  `--skip-from PATH` resumes where this one stopped.
- `--newer-than` / `--older-than` ISO 8601 mtime filters. Naive timestamps are read as local time.
- `--prompt-file PATH` replaces the default user prompt with file contents; existing photo metadata
  is still appended.
- `--max-keywords N` caps the AI keyword count per photo before merging with existing tags.
- `--dry-run` runs the model and logs the proposed metadata without writing.
- `--timeout-seconds` per-image hard cap; the retry loop handles the abort.
- `--frequency-penalty` (default 0.5) suppresses chant-style token loops observed with Qwen3-VL at
  low temperature.
- `--progress` / `--no-progress` rich progress bar (auto-disabled on non-tty stderr).
- Graceful Ctrl-C in batch runs.
- Token usage tracking per call (`InferenceResult`) and per batch (`BatchTotals`).

### Changed

- System prompt rewritten: anchors on visible image content, treats EXIF/GPS as corroborative
  evidence only, refuses to copy existing keywords as filler.
- Default console log level is now `INFO` (was `DEBUG`).
- Default `MAX_TOKENS` raised to 1200.
- Progress bar routed to stderr; stays clean alongside `--json`.
- Existing keywords are de-duplicated case-insensitively at read time.
- `WeightedFlatSubject` is now written back when persisting merged keywords.
- `--ext` matches case-insensitively; default aligned with the README.

### Fixed

- Lock leak when the PID file write failed after lock acquire.
- `parse_hierarchical_keyword` returning `['']` for empty input.
- StubAgent exposing `usage` as a callable instead of an attribute.
- Pydantic-AI deprecation: access `result.usage` as a property.
- EXIF orientation now honored; PIL file handles closed eagerly.
- Over-long keyword lists are truncated rather than failing validation.
- Cache and lock startup errors degrade to warnings instead of failing the run.
- Cache I/O errors are treated as warnings, not photo failures.
- Skip-list appender is now thread-safe under `--workers > 1`.
- Numeric env vars (`JPEG_QUALITY`, `TEMPERATURE`, etc.) parse safely with a warning instead of
  crashing.

### Security

- `--api-key` warns that CLI args are visible in process listings; prefer env vars
  (`OLLAMA_API_KEY`, `LM_STUDIO_API_KEY`, `OPENAI_API_KEY`).
- Lock file permissions tightened to `0o600`.

## [0.1.0] - 2026-02-16

Initial release.

### Added

- `photo-tagger` CLI that asks a vision-language model to analyze each photo and writes
  Lightroom-compatible metadata (title, one-sentence description, and hierarchical keywords).
- RAW and standard image support: CR3, CR2, NEF, JPG, PNG, and more.
- XMP sidecars by default; `--embed-in-photo` writes metadata directly into the image instead.
- Keyword merging with existing metadata by default; `--overwrite-keywords` replaces.
  `--no-write-title` / `--no-write-description` skip those fields.
- `--no-backup-xmp` to skip the ExifTool `_original` snapshot.
- Provider support for Ollama and LM Studio via their OpenAI-compatible APIs. Selected with
  `--provider`; endpoint and credentials via `--url` / `--api-key` or env vars (`OLLAMA_BASE_URL`,
  `OLLAMA_API_KEY`, `LM_STUDIO_BASE_URL`, `LM_STUDIO_API_KEY`, `OPENAI_API_KEY`).
- Repeatable `-i/--input` accepting files and directories; `--ext` filters by extension,
  `-r/--recursive` walks subdirectories.
- Inference knobs: `-m/--model`, `--temperature`, `--max-tokens`, `--retries`, `--jpeg-dimensions`,
  `--jpeg-quality`. Each also has an env-var override (`MODEL_NAME`, `TEMPERATURE`, etc.).
- In-memory JPEG conversion to keep token usage low.
- Structured log files for debugging and auditing.

[0.1.0]: https://github.com/jbsilva/photo-tagger/releases/tag/v0.1.0
[0.2.0]: https://github.com/jbsilva/photo-tagger/compare/v0.1.0...v0.2.0
[0.2.1]: https://github.com/jbsilva/photo-tagger/compare/v0.2.0...v0.2.1
[0.2.2]: https://github.com/jbsilva/photo-tagger/compare/v0.2.1...v0.2.2
[0.3.0]: https://github.com/jbsilva/photo-tagger/compare/v0.2.2...v0.3.0
[0.4.0]: https://github.com/jbsilva/photo-tagger/compare/v0.3.0...v0.4.0
[0.5.0]: https://github.com/jbsilva/photo-tagger/compare/v0.4.0...v0.5.0
