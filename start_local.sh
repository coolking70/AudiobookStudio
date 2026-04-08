#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Force direct network access for any child process or library that still
# honors environment proxies. The LLM client already uses trust_env=False,
# but clearing these here keeps ad-hoc scripts consistent too.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ] && [ -n "${AUDIOBOOKSTUDIO_PYTHON:-}" ]; then
  PYTHON_BIN="$AUDIOBOOKSTUDIO_PYTHON"
fi
if [ -z "$PYTHON_BIN" ] && [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
fi
if [ -z "$PYTHON_BIN" ] && [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
fi
if [ -z "$PYTHON_BIN" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "Python not found. Set PYTHON_BIN or install Python 3.10+."
    exit 1
  fi
fi

exec "$PYTHON_BIN" start_local.py
