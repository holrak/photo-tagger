---
icon: lucide/square-terminal
---

# CLI reference

photo-tagger runs as a single command, `photo-tagger`, built with
[cyclopts](https://github.com/BrianPugh/cyclopts). Its flags are grouped into logical option groups;
this page documents every flag, its default, the matching environment variable (or `-` when there is
none), and what it does.

Any flag you pass on the command line overrides the corresponding config-file value and
environment-informed default. See [Configuration](../getting-started/configuration.md) for the full
precedence rules and TOML layout.

## Commands

Running `photo-tagger` with image inputs tags them (the default command). Three subcommands exist:

| Command                   | Description                                                                                               |
| ------------------------- | --------------------------------------------------------------------------------------------------------- |
| `photo-tagger`            | Tag the given images (default). Documented by the option groups below.                                    |
| `photo-tagger doctor`     | Pre-flight check: verifies ExifTool is on `PATH` and the provider serves the model, then exits 0/1.       |
| `photo-tagger vocabulary` | Build a keyword file from a library's own keywords (see [Building a vocabulary](#building-a-vocabulary)). |
| `photo-tagger gui`        | Launch the optional desktop GUI. Requires the `gui` extra; see [Desktop GUI](gui.md).                     |

`doctor` accepts `--provider`, `-m/--model`, `-u/--url`, and `-k/--api-key` (same meanings as below)
and honors the same config file and environment variables. Run it first when a tagging run will not
start:

```console
$ photo-tagger doctor --provider lmstudio --model qwen/qwen3-vl-30b
photo-tagger 0.5.0 environment check

  OK    ExifTool: /usr/bin/exiftool
  OK    Model 'qwen/qwen3-vl-30b' on lmstudio: available at http://localhost:1234/v1

All checks passed.
```

## Input and scanning

`-i/--input` is required and repeatable: pass it once per file or directory you want to process.

| Flag                         | Default    | Env var | Description                                                                                   |
| ---------------------------- | ---------- | ------- | --------------------------------------------------------------------------------------------- |
| `-i`, `--input` PATH         | (required) | `-`     | One or more files and/or directories; repeat the flag.                                        |
| `--ext`, `--extensions` LIST | `cr3,jpg`  | `-`     | Comma-separated extensions used when scanning directories (case-insensitive).                 |
| `-r`, `--recursive`          | `false`    | `-`     | Recurse into subdirectories while scanning input directories.                                 |
| `-w`, `--workers` N          | `1`        | `-`     | Process N photos concurrently with a thread pool. The model server is usually the bottleneck. |
| `--skip-from` PATH           | none       | `-`     | Skip filenames listed in PATH (one per line; lines starting with `#` are comments).           |
| `--append-to-skip-file` PATH | none       | `-`     | Append each successfully tagged filename to PATH as the run progresses (created if missing).  |

## Provider

The provider group selects the backend and how to reach it. Prefer the API-key environment variables
over `--api-key` so the key never lands in your shell history.

| Flag                  | Default                                                                                                            | Env var                                                                             | Description                                                        |
| --------------------- | ------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `--provider` NAME     | `lmstudio`                                                                                                         | `-`                                                                                 | Backend: `ollama`, `lmstudio`, `llamacpp`, or `openai`.            |
| `-m`, `--model` NAME  | `qwen/qwen3-vl-30b`                                                                                                | `MODEL_NAME`                                                                        | Vision-language model identifier.                                  |
| `-u`, `--url` URL     | `http://localhost:1234/v1` (lmstudio), `http://localhost:11434/v1` (ollama), `http://localhost:8080/v1` (llamacpp) | `LM_STUDIO_BASE_URL` / `OLLAMA_BASE_URL` / `LLAMA_CPP_BASE_URL` / `OPENAI_BASE_URL` | Provider API base URL.                                             |
| `-k`, `--api-key` KEY | none                                                                                                               | `OLLAMA_API_KEY` / `LM_STUDIO_API_KEY` / `LLAMA_CPP_API_KEY` / `OPENAI_API_KEY`     | API key; prefer the env vars over the flag. Required for `openai`. |
| `--retries` N         | `5`                                                                                                                | `RETRIES`                                                                           | Automatic retries when the model output fails schema validation.   |

## Inference

These flags tune sampling and the image sent to the model. Lower temperature and a frequency penalty
keep the output focused; the JPEG settings control how much detail the model sees.

| Flag                               | Default   | Env var             | Description                                                                                                                                                                                                             |
| ---------------------------------- | --------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--output-language`, `--lang` NAME | `English` | `-`                 | Language of the generated title, description, and keywords (any language name the model understands, e.g. `German`, `"Brazilian Portuguese"`).                                                                          |
| `--hint` TEXT                      | none      | `-`                 | A note about every photo in the run that the model trusts over its own reading of the image, e.g. `"The animal in these photos is a deer"`. Changes the cache namespace, so hinted runs never replay hint-less results. |
| `--temperature` FLOAT              | `0.2`     | `TEMPERATURE`       | Sampling temperature.                                                                                                                                                                                                   |
| `--max-tokens` N                   | `1200`    | `MAX_TOKENS`        | Maximum tokens to generate.                                                                                                                                                                                             |
| `--timeout-seconds` FLOAT          | `60.0`    | `TIMEOUT_SECONDS`   | Per-image inference timeout; on timeout the retry loop steps in.                                                                                                                                                        |
| `--frequency-penalty` FLOAT        | `0.5`     | `FREQUENCY_PENALTY` | Penalty on repeated tokens; discourages repetitive output loops.                                                                                                                                                        |
| `--jpeg-dimensions` N              | `1280`    | `JPEG_DIMENSIONS`   | Max dimension (px) of the JPEG sent to the model.                                                                                                                                                                       |
| `--jpeg-quality` N                 | `80`      | `JPEG_QUALITY`      | JPEG quality (1-100) of the image sent to the model.                                                                                                                                                                    |

## Output

The output group decides what metadata is written and where. By default photo-tagger writes an XMP
sidecar next to each image and leaves the original untouched.

| Flag                                             | Default           | Env var | Description                                                                                         |
| ------------------------------------------------ | ----------------- | ------- | --------------------------------------------------------------------------------------------------- |
| `--preserve-keywords` / `--overwrite-keywords`   | preserve (`true`) | `-`     | Merge with existing keywords vs replace them.                                                       |
| `--write-title` / `--no-write-title`             | write (`true`)    | `-`     | Generate and write a title.                                                                         |
| `--write-description` / `--no-write-description` | write (`true`)    | `-`     | Generate and write a description.                                                                   |
| `--write-keywords` / `--no-write-keywords`       | write (`true`)    | `-`     | Write keywords (merged per `--preserve-keywords`); `--no-write-keywords` leaves existing ones.      |
| `--write-sidecar` / `--embed-in-photo`           | sidecar (`true`)  | `-`     | Write an XMP sidecar (default) vs embed metadata into the image file.                               |
| `--backup-xmp` / `--no-backup-xmp`               | backup (`true`)   | `-`     | Keep ExifTool's `*_original` backup before writing; `--no-backup-xmp` passes `-overwrite_original`. |
| `--max-keywords` N                               | none (keep all)   | `-`     | Cap AI-generated keywords kept per photo before merging.                                            |
| `--vocabulary` PATH                              | none              | `-`     | Restrict generated keywords to the terms in PATH (see below).                                       |
| `--vocabulary-strict`                            | `false`           | `-`     | Drop generated keywords the vocabulary does not cover instead of writing them as-is.                |
| `--session-gap` MINUTES                          | `0` (off)         | `-`     | Group photos into shoots and make each shoot's keywords agree with itself (see below).              |
| `--dry-run`                                      | `false`           | `-`     | Run the model and log the proposed metadata, but write nothing.                                     |

### Controlled vocabulary

`--vocabulary` points at the keyword list your catalog already uses, so a run cannot seed it with
near-duplicates of keywords you have curated by hand. Two file shapes are accepted:

- A **Lightroom keyword-list export** (_Metadata > Export Keywords_), in either shape that menu
    offers: the `.txt` (_Exclude Keyword Tag Options_) is one keyword per line, children indented
    under their parent, `{braces}` for synonyms, `[brackets]` for keywords marked "do not export";
    the `.csv` (_Include Keyword Tag Options_) is the same list behind four option columns, and the
    keyword column is lifted out of it automatically.
- A **plain list**: one term per line, optionally as a full path in either `Animal|Bird|Osprey` or
    `Osprey<Bird<Animal` form. Blank lines are ignored, and so are `#` comments: a comment needs a
    space after the hash, so a hashtag-style keyword such as `#Diversity` is kept as a keyword.

```text
Animal
	Bird
		Osprey
		{Sea Hawk}
	Mammal
Landscape
```

Every generated keyword is matched against the file, ignoring case and punctuation, with a
conservative fuzzy pass for typos and longer variants. A match is rewritten to the file's own
spelling **and hierarchy**, so `ospreys`, `Sea Hawk`, and `Osprey<Raptor<Wildlife` all land as
`Animal|Bird|Osprey`. The file's spelling is used exactly as written, so a catalog that keeps its
keywords in lower case stays that way. Keywords the file does not cover pass through untouched
unless `--vocabulary-strict` is set, in which case they are dropped and reported: the run summary's
`vocabulary_dropped` names every rejected term and how often it came up, which is the list to work
from when growing the vocabulary. `vocabulary_mapped` counts the rewrites.

#### Languages other than English

Write the vocabulary in the same language you generate in ([`--output-language`](#inference)); a
German run cannot match an English catalog whatever the matcher does.

Matching itself is language-neutral except in one place: folding a plural onto its singular
(`Ospreys` → `Osprey`) uses English rules, so it is applied for English output and skipped for every
other language. Applying it to German would merge `Alles` into `Alle`, and it can say nothing at all
about `птицы`. Everything else works in any script: case folding (including `Straße` and `Strasse`),
punctuation and spacing, and the fuzzy pass, which still unifies longer inflections such as
`Landschaften` with `Landschaft` or `Закаты` with `Закат`.

For the short inflected forms no ratio can safely catch, declare them in the file with the synonym
syntax, which is exact and needs no guessing:

```text
Животное
	Птица
	{птицы}
	{птиц}
```

The vocabulary is listed in the prompt as well, so the model prefers your terms in the first place
instead of being corrected afterwards. That listing is part of the cache namespace: swapping
vocabulary files starts a fresh cache slice rather than replaying keywords chosen under the old one.

### Sessions

Each photo is analyzed on its own, so forty frames of the same bird can come back as `Osprey` here
and `Ospreys` there, filed under `Bird<Animal` on one frame and `Raptor<Wildlife` on the next.
Lightroom then shows four keywords where there is one subject.

`--session-gap MINUTES` fixes that without needing a vocabulary file. The batch is split into shoots
wherever the capture time (EXIF `DateTimeOriginal`, falling back to file mtime) jumps by more than
the given gap. Every photo in a shoot is analyzed first, then the session's own output becomes its
vocabulary: the spelling most of the session used wins, and so does the hierarchy most of it used.
Only then is anything written. It reads `--output-language` for the same reason the vocabulary file
does, so a German or Russian shoot is harmonized under that language's rules rather than English
ones, and it folds close-enough variants together on top of that: a shoot that said `Закаты` twice
and `Закат` once writes `Закаты` throughout, and one that said `Landschaften` and `Landschaft`
settles on one of them. Short words are never folded this way, so `Alle` and `Alles` stay two
keywords.

```bash
photo-tagger -i ~/Pictures/Trip -r --session-gap 60
```

Because the vocabulary is derived from the finished results rather than fed to the model, the
outcome does not depend on which photo finished first: the same batch harmonizes the same way every
time, at any `--workers` setting. It composes with `--vocabulary`, which is applied per photo first.

Two details worth knowing:

- Photos are processed in capture order rather than the order they were listed.
- Sessions run one after another (the photos inside one still run in parallel), and a session's
    writes happen in a burst at its end. If a write fails there it is reported as failed and **not**
    retried: the model work is already done and harmonized, and an ExifTool write that failed for a
    filesystem reason is not the kind of failure a second attempt clears. Analysis failures are
    still retried as usual, and a photo recovered by the retry pass lands on its shoot's agreed
    terms.

## Filter

Filters narrow the resolved batch before any model call. Timestamps use ISO 8601, such as
`2024-01-01` or `2024-01-01T14:30`; naive timestamps use local time.

| Flag                   | Default | Env var | Description                                                                                       |
| ---------------------- | ------- | ------- | ------------------------------------------------------------------------------------------------- |
| `--skip-tagged`        | `false` | `-`     | Skip files that already have keywords, a title, or a description in the image or its XMP sidecar. |
| `--newer-than` ISO8601 | none    | `-`     | Drop files whose mtime is on/before this timestamp.                                               |
| `--older-than` ISO8601 | none    | `-`     | Drop files whose mtime is on/after this timestamp.                                                |

## Log

photo-tagger writes a timestamped log file and mirrors messages to stderr, so stdout stays clean for
[`--json`](#display) output.

| Flag                        | Default | Env var | Description                                                        |
| --------------------------- | ------- | ------- | ------------------------------------------------------------------ |
| `--console-log-level` LEVEL | `INFO`  | `-`     | `DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL`/`OFF`. `OFF` disables. |
| `--file-log-level` LEVEL    | `DEBUG` | `-`     | Same levels; `OFF` disables the file log.                          |
| `--log-folder` PATH         | `logs`  | `-`     | Folder for timestamped log files.                                  |

## Display

The display group controls the progress bar and machine-readable output.

| Flag                           | Default           | Env var | Description                                                                                                                                                                                                                                  |
| ------------------------------ | ----------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--progress` / `--no-progress` | progress (`true`) | `-`     | Live rich progress bar; auto-disabled when stderr is not a TTY.                                                                                                                                                                              |
| `--json`                       | `false`           | `-`     | Emit one NDJSON line per processed photo to stdout (`file`, `status`, `from_cache`, `retry`, `title`, `description`, `keywords`, input/output/total tokens, `seconds`). Logs and progress stay on stderr, so stdout pipes cleanly into `jq`. |

## Artifacts

The artifacts group points at side files: a custom prompt, a run summary, a per-photo CSV report, a
result cache, and a lock.

| Flag                  | Default | Env var | Description                                                                                                                                                                                                         |
| --------------------- | ------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--prompt-file` PATH  | none    | `-`     | Replace the default user prompt with the contents of PATH; existing photo metadata is still appended automatically.                                                                                                 |
| `--summary-file` PATH | none    | `-`     | Write a JSON run summary (success/failure counts, failed files, token usage, wall time) on completion.                                                                                                              |
| `--csv-file` PATH     | none    | `-`     | Write a CSV report with one row per photo (see below). Rows stream as photos finish, so a stopped run still leaves a valid file.                                                                                    |
| `--cache-file` PATH   | none    | `-`     | SQLite cache of model outputs, keyed on an image-data hash that ignores metadata (so it survives `--embed-in-photo`). Reruns skip the model call when nothing relevant changed. Created if missing; safe to delete. |
| `--lock-file` PATH    | none    | `-`     | Acquire an exclusive file lock before running; refuse to start if another photo-tagger already holds it. Works on Linux, macOS, and Windows.                                                                        |

### CSV report

Where `--summary-file` writes one JSON object for the whole run and `--json` streams NDJSON to
stdout, `--csv-file` writes a spreadsheet-friendly table with **one row per photo**. It is the
single file that gathers everything extracted and computed for each image:

- `filename`, `file`, `status`
- `title`, `description`, `keywords` (the keywords actually written), `hierarchical_keywords`
- `existing_keywords` (what was already on the file)
- `camera_model`, `lens_model`, `capture_date`, `gps_position`, `city`, `country` (read EXIF)
- `input_tokens`, `output_tokens`, `total_tokens`, `seconds`, `from_cache`, `retry`

Multi-value cells (the keyword lists) are joined with a semicolon and a space. Rows are flushed as
each photo completes, so interrupting the run with Ctrl-C still leaves a complete, openable CSV of
the work done so far. `--csv-file` and `--json` can be used together; both observe every photo. A
`--dry-run` still fills the report, which makes it handy for previewing a batch before writing any
metadata.

## Building a vocabulary

`--vocabulary-strict` is what stops a catalog sprawling, and it needs a keyword file worth
enforcing. `photo-tagger vocabulary` writes one from the library you already have:

```bash
photo-tagger vocabulary -i ~/Pictures -r -o vocabulary.txt --report dropped.csv
```

It reads the keywords your photos already carry, counts how often each one is used, and keeps the
ones that earn their place. The read goes through exiftool, so **the application does not matter**:
digiKam, darktable, Immich, PhotoPrism, Synology Photos, Piwigo and the rest all write XMP/IPTC
keywords, and only Lightroom offers a keyword-list export at all. `XMP-lr:HierarchicalSubject`
carries the hierarchy those photos really use, so the generated file keeps it.

Nothing is written to your photos or your catalog. The output is a text file to read and edit.

| Flag                  | Default    | Description                                                                               |
| --------------------- | ---------- | ----------------------------------------------------------------------------------------- |
| `-i`, `--input` PATH  | none       | Photos or folders to read keywords from; repeat the flag. Honors `--ext` and `-r`.        |
| `-o`, `--output` PATH | (required) | Where to write the vocabulary file.                                                       |
| `--from-export` PATH  | none       | Read a Lightroom keyword export (`.txt` or `.csv`) instead of, or as well as, the photos. |
| `--min-uses` N        | `2`        | Keep a keyword only when the library uses it at least this often.                         |
| `--max-terms` N       | `4800`     | Cap the file, dropping the least-used first. `0` means no cap.                            |
| `--allow-digits`      | `false`    | Keep keywords containing digits (dropped by default as measurements and model numbers).   |
| `--flat`              | `false`    | Write bare keywords instead of their hierarchies.                                         |
| `--report` PATH       | none       | Write a CSV of every dropped keyword, its count, and the rule that cut it.                |

### Which source to use

Prefer the photos. An export has no usage data at all, so `--from-export` counts *occurrences in the
keyword tree* instead: a term filed under forty parents scores forty, however many photos carry it.
It is a usable proxy for a catalog that is not on this machine, not the same measure. Pass both and
the counts are added together.

### What the rules do

Applied in this order, each one reported in `--report`:

1. **Shape.** A keyword with digits, odd punctuation, more than three words, or over 30 characters
    is a measurement, a path, or a sentence, not a subject.
2. **Rarity.** `--min-uses` drops the one-offs. In a catalog an AI has been writing to this is the
    rule that does the work: most keywords are used exactly once.
3. **Variants.** One concept keeps one spelling, the most-used one. Nothing is lost by this, since
    vocabulary matching folds case, punctuation, and plurals anyway: a photo tagged `Animals` still
    snaps onto `Animal`.
4. **The cap.** `--max-terms` removes the least-used survivors last, so it never cuts a term an
    earlier rule would have kept.

The result is deterministic: the same library gives the same file, ties broken alphabetically.

!!! tip

    Read the file before you trust it, and tune from the report rather than by guesswork. If a keyword
    you care about was dropped, `dropped.csv` names it with the count that would have kept it. The
    hierarchy is worth a look too: a catalog a tool has been writing to can file `Beach` under `Sand`,
    and a vocabulary imposes its hierarchy on every photo it matches. `--flat` drops the hierarchies
    when the source is not worth keeping.

## Skipping and resuming

Three flags cooperate to skip work you have already done and to resume a run that stopped partway
through:

- `--skip-from PATH` reads a list of filenames (one per line, `#` comments allowed) and drops any
    matching files from the batch before processing starts.
- `--append-to-skip-file PATH` appends each successfully tagged filename to PATH as the run
    progresses, creating the file if it does not exist.
- `--skip-tagged` inspects each file's existing metadata and skips anything that already has
    keywords, a title, or a description (in the image or its XMP sidecar). Use it when you want the
    skip decision to come from the files themselves rather than from a list.

For resume-on-failure, pass the **same path** to both `--skip-from` and `--append-to-skip-file`. The
first run appends every success to the file; if the run dies partway through, re-running with the
same arguments reads that file back through `--skip-from` and continues from where it left off,
without re-tagging the photos that already succeeded.

!!! tip

    Combine the skip file with `--cache-file` for an even cheaper resume: the skip file removes finished
    photos from the batch entirely, while the cache avoids re-calling the model for any photo that does
    slip back in unchanged.

See [Recipes](recipes.md) for runnable resume and skip examples.
