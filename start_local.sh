#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")"

# Force direct network access for any child process or library that still
# honors environment proxies. The LLM client already uses trust_env=False,
# but clearing these here keeps ad-hoc scripts consistent too.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

if [ ! -x ".venv/bin/python" ]; then
  echo "Missing virtual environment: .venv"
  echo "Create it with: /opt/homebrew/bin/python3.11 -m venv .venv"
  exit 1
fi

exec .venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
