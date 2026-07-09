"""Tests for the gettext-based translation layer (language resolution + the pt_BR catalog)."""

from collections.abc import Iterator

import pytest

from photo_tagger import i18n
from photo_tagger.i18n import _, activate, detect_language, gettext_noop, ngettext


@pytest.fixture(autouse=True)
def _reset_language(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Clear the language env vars and restore English after each test (activation is global)."""
    for variable in ("PHOTO_TAGGER_LANG", "LC_ALL", "LC_MESSAGES", "LANG"):
        monkeypatch.delenv(variable, raising=False)
    yield
    activate("en")


# --- resolution --------------------------------------------------------------------------------


def test_detect_language_defaults_to_english() -> None:
    """Nothing configured anywhere resolves to English."""
    assert detect_language() == "en"
    assert detect_language(i18n.AUTO) == "en"


def test_detect_language_prefers_the_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """PHOTO_TAGGER_LANG wins over the config value."""
    monkeypatch.setenv("PHOTO_TAGGER_LANG", "pt_BR")
    assert detect_language("en") == "pt_BR"


def test_detect_language_uses_the_config_value() -> None:
    """The config file's language key applies when no env override is set."""
    assert detect_language("pt_BR") == "pt_BR"


def test_detect_language_falls_back_to_posix_locale(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no explicit choice, LANG decides; unsupported locales fall through to English."""
    monkeypatch.setenv("LANG", "pt_BR.UTF-8")
    assert detect_language() == "pt_BR"
    monkeypatch.setenv("LANG", "de_DE.UTF-8")
    assert detect_language() == "en"


def test_detect_language_uses_the_system_hint_last(monkeypatch: pytest.MonkeyPatch) -> None:
    """The GUI's Qt locale hint applies only when nothing else resolved (a Finder launch)."""
    assert detect_language(system_hint="pt-BR") == "pt_BR"
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    assert detect_language(system_hint="pt-BR") == "en"


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("pt_BR", "pt_BR"),
        ("pt-BR", "pt_BR"),
        ("pt_BR.UTF-8", "pt_BR"),
        ("pt", "pt_BR"),
        ("PT_br", "pt_BR"),
        ("en_GB", "en"),
        ("de_DE", None),
        ("", None),
        (None, None),
    ],
)
def test_normalize_language(tag: str | None, expected: str | None) -> None:
    """Locale tags in any common spelling map onto the supported codes."""
    assert i18n.normalize_language(tag) == expected


def test_unknown_config_value_degrades_to_english() -> None:
    """A bogus configured language never crashes; it just falls back."""
    assert detect_language("tlh") == "en"


# --- activation and the shipped catalog ---------------------------------------------------------


def test_activate_english_is_identity() -> None:
    """Under English every msgid comes back unchanged."""
    activate("en")
    assert _("Generate Selected") == "Generate Selected"
    assert ngettext("{n} file", "{n} files", 1) == "{n} file"


def test_activate_loads_the_brazilian_portuguese_catalog() -> None:
    """The shipped pt_BR catalog translates a known GUI string."""
    assert activate("pt_BR") == "pt_BR"
    assert _("Generate Selected") == "Gerar Selecionadas"


def test_pt_br_plural_rules() -> None:
    """Portuguese pluralization: one singular, everything else plural."""
    activate("pt_BR")
    assert ngettext("{n} file", "{n} files", 1).format(n=1) == "1 arquivo"
    assert ngettext("{n} file", "{n} files", 2).format(n=2) == "2 arquivos"


def test_untranslated_message_falls_back_to_the_source() -> None:
    """A msgid missing from the catalog is shown in English rather than blank."""
    activate("pt_BR")
    assert _("this string is not in any catalog") == "this string is not in any catalog"


def test_gettext_noop_returns_the_message_unchanged() -> None:
    """The extraction marker never translates."""
    activate("pt_BR")
    assert gettext_noop("Generate Selected") == "Generate Selected"
