# PyInstaller spec for the Photo Tagger macOS app bundle.
# Build via packaging/build_macos_app.sh (which also generates the icon).
#
# Not a normal module: Analysis/PYZ/EXE/COLLECT/BUNDLE and SPECPATH are injected by PyInstaller,
# so this file is neither imported nor linted/type-checked as application code.
from pathlib import Path

from PyInstaller.utils.hooks import copy_metadata

_HERE = Path(SPECPATH)  # noqa: F821 - injected by PyInstaller
_ROOT = _HERE.parent
_ICNS = _HERE / "icon.icns"

# Some packages read their own version via importlib.metadata at import time (genai_prices, pulled
# in by pydantic-ai, hard-fails without it). Bundle that metadata. photo-tagger's own metadata is
# included too so the app reports its real version instead of "0.0.0+unknown".
_datas = [
    (str(_ROOT / "src" / "photo_tagger" / "resources"), "photo_tagger/resources"),
    # Compiled gettext catalogs; without them the frozen app is English-only.
    (str(_ROOT / "src" / "photo_tagger" / "locale"), "photo_tagger/locale"),
]
_datas += copy_metadata("pydantic_ai", recursive=True)
_datas += copy_metadata("photo-tagger")

a = Analysis(  # noqa: F821
    [str(_HERE / "photo_tagger_gui.py")],
    pathex=[str(_ROOT / "src")],
    datas=_datas,
    hiddenimports=[],
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Photo Tagger",
    console=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="Photo Tagger")  # noqa: F821

app = BUNDLE(  # noqa: F821
    coll,
    name="Photo Tagger.app",
    icon=str(_ICNS) if _ICNS.exists() else None,
    bundle_identifier="photo.tagger.app",
    info_plist={
        "CFBundleName": "Photo Tagger",
        "CFBundleDisplayName": "Photo Tagger",
        "NSHighResolutionCapable": True,
        "LSApplicationCategoryType": "public.app-category.photography",
    },
)
