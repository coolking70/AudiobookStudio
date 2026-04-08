from __future__ import annotations

import os

import uvicorn


def clear_proxy_env() -> None:
    for key in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ.pop(key, None)


def main() -> None:
    clear_proxy_env()
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    reload_enabled = os.getenv("RELOAD", "").strip().lower() in {"1", "true", "yes", "on"}
    uvicorn.run("app:app", host=host, port=port, reload=reload_enabled)


if __name__ == "__main__":
    main()
