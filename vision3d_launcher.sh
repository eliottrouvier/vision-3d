#!/usr/bin/env bash
# Vision-3D Studio Launcher (supports symlinks)

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do
  DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
DIR="$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)"

# Fix Qt QPA platform plugin loader when path contains colons or special chars
QT_SHARE_PLUGINS="$HOME/.local/share/vision3d/plugins"
if [ ! -f "$QT_SHARE_PLUGINS/platforms/libqcocoa.dylib" ]; then
  mkdir -p "$QT_SHARE_PLUGINS"
  PYSIDE_PLUGINS="$(find "$DIR/.venv" -type d -path "*/PySide6/Qt/plugins" 2>/dev/null | head -n 1)"
  if [ -n "$PYSIDE_PLUGINS" ] && [ -d "$PYSIDE_PLUGINS" ]; then
    cp -R "$PYSIDE_PLUGINS/"* "$QT_SHARE_PLUGINS/" 2>/dev/null || true
  fi
fi

if [ -d "$QT_SHARE_PLUGINS" ]; then
  export QT_PLUGIN_PATH="$QT_SHARE_PLUGINS"
  export QT_QPA_PLATFORM_PLUGIN_PATH="$QT_SHARE_PLUGINS/platforms"
fi

exec "$DIR/.venv/bin/python" "$DIR/main.py" "$@"
