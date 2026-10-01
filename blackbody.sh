#!/usr/bin/env bash
# Blackbody launcher for macOS and Linux: sets up a private Python environment on first run.
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Setting up Blackbody for the first time (about 400 MB)..."
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi
exec .venv/bin/python -m blackbody "$@"
