#!/bin/bash
# Starts the Padel Analyzer and opens it in your browser.
# The first run installs everything into a local .venv folder (a few minutes).
cd "$(dirname "$0")" || exit 1
EXTRAS="${PADELFUSION_EXTRAS:-yolo,llm}"

fail() { echo; echo "$1"; echo; read -r -p "Press Enter to close this window." _; exit 1; }

PY="$(command -v python3 || true)"
[ -n "$PY" ] || fail "Python 3.10 or newer is needed. Install it from https://www.python.org/downloads/ and double-click this file again."
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
  || fail "Your Python is too old ($("$PY" --version)). Install 3.10 or newer from https://www.python.org/downloads/"

if [ ! -x .venv/bin/python ]; then
  echo "First run: setting up the analyzer. This takes a few minutes..."
  "$PY" -m venv .venv || fail "Could not create the .venv folder."
fi
if [ ! -f .venv/.installed ] || [ pyproject.toml -nt .venv/.installed ]; then
  .venv/bin/python -m pip install --upgrade pip >/dev/null
  if ! .venv/bin/python -m pip install -e ".[$EXTRAS]"; then
    echo "Full install failed; installing without the YOLO detector instead."
    .venv/bin/python -m pip install -e ".[llm]" || fail "Installation failed. See the messages above."
  fi
  touch .venv/.installed
fi
exec .venv/bin/python -m padelfusion ui
