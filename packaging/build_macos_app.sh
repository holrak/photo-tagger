#!/usr/bin/env bash
# Build "Photo Tagger.app" (unsigned) for macOS with PyInstaller.
#
#   ./packaging/build_macos_app.sh
#
# Output: dist/Photo Tagger.app  (PyInstaller's build/ and dist/ are git-ignored).
# Unsigned, so the first launch needs a right-click > Open (Gatekeeper asks once). The app finds a
# Homebrew-installed exiftool automatically; install it with `brew install exiftool` if missing.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

RUN=(uv run --extra gui --group package)

echo "==> Generating the app icon (best-effort)..."
if ! "${RUN[@]}" python packaging/make_icns.py; then
  echo "    icon generation failed; bundling with PyInstaller's default icon."
fi

echo "==> Building the app bundle..."
"${RUN[@]}" pyinstaller --noconfirm --clean packaging/photo-tagger.spec \
  --distpath "$ROOT/dist" --workpath "$ROOT/build"

APP="$ROOT/dist/Photo Tagger.app"
echo "==> Verifying the bundle (import self-test)..."
"$APP/Contents/MacOS/Photo Tagger" --selftest

echo
echo "==> Done: $APP"
echo "    Launch it the first time with right-click > Open (it is unsigned)."
