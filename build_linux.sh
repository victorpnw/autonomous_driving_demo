#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo " Building Autonomous Driving Demo (Linux)"
echo "=========================================="
echo

PYTHON_BIN="python3"
PIP_BIN="pip3"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
  PIP_BIN=".venv/bin/pip"
fi

if ! "$PYTHON_BIN" -m PyInstaller --version >/dev/null 2>&1; then
  echo "PyInstaller not found. Installing..."
  "$PIP_BIN" install pyinstaller
  echo
fi

echo "Running PyInstaller..."
"$PYTHON_BIN" -m PyInstaller game.spec --distpath dist --workpath build --clean -y

echo
echo "=========================================="
echo " Build complete!"
echo " Output: dist/AutonomousDrivingDemo"
echo "=========================================="
