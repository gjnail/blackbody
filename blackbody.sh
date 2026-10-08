#!/usr/bin/env bash
# Blackbody launcher for macOS and Linux: sets up a private Python environment on first run (and again when
# requirements.txt changes), then opens the app.
set -e
cd "$(dirname "$0")"
# (the pinned packages have ready-built wheels for 64-bit Python 3.12 and 3.13 only)
check='import sys; sys.exit(not ((3, 12) <= sys.version_info[:2] <= (3, 13) and sys.maxsize > 2**32))'
# .venv/.installed is a copy of requirements.txt, written only once an install has worked, so a failed or
# interrupted install is tried again next time
if ! cmp -s requirements.txt .venv/.installed; then
  rm -f .venv/.installed
  if [ -e .venv ] && ! { .venv/bin/python -c "$check" && .venv/bin/python -c 'import pip'; } >/dev/null 2>&1; then
    echo "The Python environment in .venv cannot be used, so Blackbody is making a new one."
    rm -rf .venv
  fi
  if [ ! -e .venv ]; then
    py=
    for p in python3.13 python3.12 python3; do
      if command -v "$p" >/dev/null 2>&1 && "$p" -c "$check" >/dev/null 2>&1; then py=$p; break; fi
    done
    if [ -z "$py" ]; then
      echo "Blackbody needs 64-bit Python 3.12 or 3.13: https://www.python.org/downloads/" >&2
      echo "Newer versions are not supported yet, as some of the packages it uses have no builds for them." >&2
      echo "python3 here is: $(python3 --version 2>/dev/null || echo none)" >&2
      exit 1
    fi
    "$py" -m venv .venv || { rm -rf .venv; exit 1; }
  fi
  echo "Setting up Blackbody. The first time, this downloads about 400 MB and takes a few minutes..."
  if ! { .venv/bin/python -m pip install --upgrade pip && .venv/bin/python -m pip install -r requirements.txt; }; then
    echo "Installing the dependencies failed: see the messages above. Running ./blackbody.sh again tries again." >&2
    exit 1
  fi
  cp requirements.txt .venv/.installed
fi
exec .venv/bin/python -m blackbody "$@"
