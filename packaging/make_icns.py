# mypy: ignore-errors
"""
Render src/photo_tagger/resources/icon.svg into a macOS ``icon.icns`` for the app bundle.

Uses Qt (already a GUI build dependency) to rasterize the SVG at the standard iconset sizes, then
``iconutil`` to pack them. Best-effort: build_macos_app.sh treats a missing icon.icns as "use the
default icon", so a failure here never blocks the build.
"""

import os
import subprocess
import sys
from pathlib import Path


# A complete macOS iconset: (pixel size, iconutil-required filename suffix).
_ICONSET = (
    (16, "16x16"),
    (32, "16x16@2x"),
    (32, "32x32"),
    (64, "32x32@2x"),
    (128, "128x128"),
    (256, "128x128@2x"),
    (256, "256x256"),
    (512, "256x256@2x"),
    (512, "512x512"),
    (1024, "512x512@2x"),
)


def main() -> int:
    """Rasterize the SVG into an iconset and pack it into icon.icns; return a process exit code."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # render without a display
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer
    from PySide6.QtWidgets import QApplication

    here = Path(__file__).resolve().parent
    svg = here.parent / "src" / "photo_tagger" / "resources" / "icon.svg"
    if not svg.is_file():
        sys.stderr.write(f"icon source not found: {svg}\n")
        return 1

    QApplication([])  # QImage/QPainter need a Q*Application instance
    renderer = QSvgRenderer(str(svg))
    iconset = here / "icon.iconset"
    iconset.mkdir(parents=True, exist_ok=True)
    for size, suffix in _ICONSET:
        image = QImage(size, size, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        painter = QPainter(image)
        renderer.render(painter)
        painter.end()
        image.save(str(iconset / f"icon_{suffix}.png"), "PNG")

    icns = here / "icon.icns"
    subprocess.run(["/usr/bin/iconutil", "-c", "icns", str(iconset), "-o", str(icns)], check=True)
    sys.stdout.write(f"wrote {icns}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
