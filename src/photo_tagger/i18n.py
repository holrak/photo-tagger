"""
Runtime translations for user-facing strings (GNU gettext).

The GUI is the translated surface: every label, tooltip, and status message goes through :func:`_`
(or :func:`ngettext` when the wording depends on a count). The CLI's structured log events and
``--help`` text intentionally stay in English.

Catalogs live in ``locale/<code>/LC_MESSAGES/photo_tagger.mo`` next to this module; the editable
``.po`` sources sit beside them and are compiled by ``scripts/compile_translations.py``. Language
resolution order: the ``PHOTO_TAGGER_LANG`` environment variable, the top-level ``language`` key in
the config file, the OS locale, then English.
"""

import gettext as _gettext_module
import os
from pathlib import Path


LOCALE_DIR = Path(__file__).parent / "locale"
DOMAIN = "photo_tagger"

# The config value meaning "follow the OS locale" (the default when the key is absent).
AUTO = "auto"

# Languages with a shipped catalog (plus the source language). Values are the native-language
# labels the GUI shows in its Language menu, so a user who cannot read the current language can
# still find their own.
SUPPORTED_LANGUAGES: dict[str, str] = {
    "en": "English",
    "pt_BR": "Português (Brasil)",
}

_FALLBACK = "en"

# The active catalog. NullTranslations returns every msgid unchanged, so before activate() runs
# (and in the test suite) every string is its English source.
_active: _gettext_module.NullTranslations = _gettext_module.NullTranslations()


def _(message: str) -> str:
    """Translate *message* under the active catalog (identity for English or a missing entry)."""
    return _active.gettext(message)


def ngettext(singular: str, plural: str, count: int) -> str:
    """Translate a count-dependent message, honoring the target language's plural rules."""
    return _active.ngettext(singular, plural, count)


def gettext_noop(message: str) -> str:
    """
    Mark *message* for catalog extraction without translating it here.

    For module-level constants (dicts of labels, template strings): the literal must be visible to
    the extractor at its definition, but the lookup has to happen at display time, after activate()
    has installed a catalog. Definition sites wrap with this; use sites call :func:`_` on the value.
    """
    return message


def normalize_language(tag: str | None) -> str | None:
    """
    Map a locale tag ("pt-BR", "pt_BR.UTF-8", "pt") to a supported code, or None.

    Accepts BCP-47 and POSIX spellings. A bare language falls back to the first supported regional
    variant ("pt" resolves to "pt_BR"), so an approximate OS locale still gets a usable catalog.
    """
    if not tag:
        return None
    code = tag.replace("-", "_").split(".")[0].split("@")[0]
    for supported in SUPPORTED_LANGUAGES:
        if supported.casefold() == code.casefold():
            return supported
    base = code.split("_")[0].casefold()
    for supported in SUPPORTED_LANGUAGES:
        if supported.split("_")[0].casefold() == base:
            return supported
    return None


def detect_language(configured: str | None = None, *, system_hint: str | None = None) -> str:
    """
    Resolve which supported language to use.

    ``PHOTO_TAGGER_LANG`` wins (an explicit per-run override), then the config file's ``language``
    key (*configured*), then the POSIX locale variables, then *system_hint* (the GUI passes Qt's
    system locale, which knows the OS language even when no LANG is exported, e.g. a Finder launch).
    Anything unset, "auto", or unsupported falls through; the final fallback is English.
    """
    for candidate in (os.getenv("PHOTO_TAGGER_LANG"), configured):
        if candidate and candidate != AUTO:
            resolved = normalize_language(candidate)
            if resolved is not None:
                return resolved
    for variable in ("LC_ALL", "LC_MESSAGES", "LANG"):
        resolved = normalize_language(os.getenv(variable))
        if resolved is not None:
            return resolved
    resolved = normalize_language(system_hint)
    return resolved if resolved is not None else _FALLBACK


def activate(configured: str | None = None, *, system_hint: str | None = None) -> str:
    """
    Install the catalog for the resolved language process-wide; return the resolved code.

    ``fallback=True`` means a missing catalog degrades to English rather than raising, so a partial
    install (or the source tree before compilation) still runs.
    """
    global _active  # noqa: PLW0603 - one process-wide catalog is the point of gettext.
    language = detect_language(configured, system_hint=system_hint)
    _active = _gettext_module.translation(
        DOMAIN,
        localedir=LOCALE_DIR,
        languages=[language],
        fallback=True,
    )
    return language
