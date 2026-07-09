---
icon: lucide/languages
---

# Translations

The desktop GUI is translated at runtime with GNU gettext. English is the source language; Brazilian
Portuguese (`pt_BR`) ships complete. The CLI's `--help` text and the structured log events
intentionally stay in English so scripts and bug reports remain grep-able.

## Choosing a language

By default the GUI follows the operating system's language and falls back to English when there is
no matching translation. To pick one explicitly, use any of these (first match wins):

1. The `PHOTO_TAGGER_LANG` environment variable, for a one-off run:

    ```bash
    PHOTO_TAGGER_LANG=pt_BR photo-tagger gui
    ```

2. **Settings > Language** in the GUI. The choice is written to the config file (comments and other
    settings are preserved) and applies on the next launch; **System Default** removes it again.

3. The top-level `language` key in the [config file](../getting-started/configuration.md):

    ```toml
    language = "pt_BR"
    ```

Accepted values are `en`, `pt_BR` (any common spelling works: `pt-BR`, `pt_BR.UTF-8`, plain `pt`),
or `auto` for the OS default. Unknown values quietly fall back to English.

## Contributing a translation

Catalogs live in `src/photo_tagger/locale/`; the editable `.po` files are committed next to the
compiled `.mo` files gettext loads. To add a language (say German):

```bash
# 1. Create the catalog from the current template
uv run pybabel init -i src/photo_tagger/locale/photo_tagger.pot \
    -d src/photo_tagger/locale -D photo_tagger -l de

# 2. Fill in the msgstr entries in src/photo_tagger/locale/de/LC_MESSAGES/photo_tagger.po

# 3. Compile the runtime catalog (a pre-commit hook fails if you forget)
uv run scripts/compile_translations.py
```

Then add the language code and its native-language label to `SUPPORTED_LANGUAGES` in
`src/photo_tagger/i18n.py` so it appears in the GUI's Language menu, and add a catalog-loading test
to `tests/test_i18n.py`.

After changing translatable strings in the source, refresh the template and merge it into the
existing catalogs:

```bash
uv run pybabel extract -k gettext_noop --project photo-tagger \
    -o src/photo_tagger/locale/photo_tagger.pot src/photo_tagger
uv run pybabel update -i src/photo_tagger/locale/photo_tagger.pot \
    -d src/photo_tagger/locale -D photo_tagger
uv run scripts/compile_translations.py
```
