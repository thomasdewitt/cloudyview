#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/cloudyview-uv-cache}"
export UV_CACHE_DIR

uv run --extra dev --extra interactive python packaging/make_icons.py
uv run --extra dev --extra interactive pyinstaller \
    --clean --noconfirm packaging/soar.spec

BUNDLE="$ROOT_DIR/dist/cloudyview-soar"
EXECUTABLE="$BUNDLE/cloudyview-soar"
ICON="$BUNDLE/_internal/packaging/icon_512.png"
test -x "$EXECUTABLE"
test -f "$ICON"
echo "Built $BUNDLE"

if command -v appimagetool >/dev/null 2>&1; then
    APPDIR="$ROOT_DIR/build/appimage/CloudyView-Soar.AppDir"
    rm -rf "$APPDIR"
    install -d "$APPDIR/usr/lib" "$APPDIR/usr/share/applications"
    cp -a "$BUNDLE" "$APPDIR/usr/lib/cloudyview-soar"
    ln -s "usr/lib/cloudyview-soar/cloudyview-soar" "$APPDIR/AppRun"
    install -m 0644 packaging/icon_512.png "$APPDIR/cloudyview-soar.png"
    sed \
        -e 's|@EXECUTABLE@|cloudyview-soar|g' \
        -e 's|@ICON@|cloudyview-soar|g' \
        packaging/cloudyview-soar.desktop \
        > "$APPDIR/cloudyview-soar.desktop"
    cp "$APPDIR/cloudyview-soar.desktop" \
        "$APPDIR/usr/share/applications/cloudyview-soar.desktop"
    appimagetool "$APPDIR" "$ROOT_DIR/dist/CloudyView-Soar.AppImage"
else
    echo "appimagetool not found; skipped AppImage creation."
fi

if [[ "${1:-}" == "--install-desktop" ]]; then
    APPLICATIONS_DIR="$HOME/.local/share/applications"
    install -d "$APPLICATIONS_DIR"
    sed \
        -e "s|@EXECUTABLE@|$EXECUTABLE|g" \
        -e "s|@ICON@|$ICON|g" \
        packaging/cloudyview-soar.desktop \
        > "$APPLICATIONS_DIR/cloudyview-soar.desktop"
    echo "Installed $APPLICATIONS_DIR/cloudyview-soar.desktop"
else
    echo "Desktop launcher not installed; rerun with --install-desktop to install locally."
fi
