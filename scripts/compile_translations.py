#!/usr/bin/env python3
# ruff: noqa: T201 - print() is this build script's output channel, as in check_version_sync.py.
"""
Compile every locale's .po catalog into the .mo file gettext loads at runtime.

The compiled .mo files are committed (so editable installs and the plain wheel build need no build
step); this script keeps them in sync with the editable .po sources. Run it after changing a .po:

uv run scripts/compile_translations.py

``--check`` recompiles in memory and fails when a committed .mo is stale or missing, which is how
the pre-commit hook guards against editing a .po and forgetting to recompile.
"""

import sys
from io import BytesIO
from pathlib import Path

from babel.messages.mofile import write_mo
from babel.messages.pofile import read_po


LOCALE_DIR = Path(__file__).parent.parent / "src" / "photo_tagger" / "locale"
DOMAIN = "photo_tagger"


def compiled_bytes(po_path: Path) -> bytes:
    """Return the .mo bytes for *po_path*'s catalog."""
    with po_path.open("rb") as fh:
        catalog = read_po(fh)
    buffer = BytesIO()
    write_mo(buffer, catalog)
    return buffer.getvalue()


def main() -> int:
    """Compile (or with --check, verify) every catalog; return the exit code."""
    check = "--check" in sys.argv[1:]
    po_files = sorted(LOCALE_DIR.glob(f"*/LC_MESSAGES/{DOMAIN}.po"))
    if not po_files:
        print(f"no catalogs found under {LOCALE_DIR}", file=sys.stderr)
        return 1

    stale: list[Path] = []
    for po_path in po_files:
        mo_path = po_path.with_suffix(".mo")
        expected = compiled_bytes(po_path)
        current = mo_path.read_bytes() if mo_path.exists() else None
        if current == expected:
            continue
        if check:
            stale.append(mo_path)
        else:
            mo_path.write_bytes(expected)
            print(f"compiled {mo_path.relative_to(LOCALE_DIR.parent.parent.parent)}")

    if stale:
        names = ", ".join(str(path) for path in stale)
        print(
            f"stale compiled catalog(s): {names}\nrun: uv run scripts/compile_translations.py",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
