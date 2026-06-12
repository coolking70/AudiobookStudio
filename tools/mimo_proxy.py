"""OpenAI-compatible proxy for mimocode (mimo-auto free model).

Converts POST /v1/chat/completions → `mimo run --attach` subprocess call, so
any tool that expects an OpenAI endpoint (e.g. book_voice_parser/audit.py
hetero_llm) can use mimo without a native API.

Architecture:
  1. On start, launches `mimo serve` as a background process on an internal port.
  2. Each /v1/chat/completions request calls `mimo run --attach <internal>` to
     reuse the already-warm server (faster than cold subprocess: ~7s vs ~12s).
  3. Returns an OpenAI-compatible JSON response.

Usage:
    python tools/mimo_proxy.py [--port 19999]

Then configure hetero_llm as:
    base_url: http://127.0.0.1:19999/v1
    model:    mimo-auto
    api_key:  local

Only non-streaming /v1/chat/completions is supported (what audit.py uses).
Sessions are one-shot: each request is a fresh message, no history.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

_serve_proc: subprocess.Popen | None = None
_mimo_server_url: str = ""


def _start_mimo_server(internal_port: int) -> None:
    global _serve_proc, _mimo_server_url
    _mimo_server_url = f"http://127.0.0.1:{internal_port}"
    _serve_proc = subprocess.Popen(
        ["zsh", "-i", "-c", f"mimo serve --port {internal_port}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # wait until port is accepting connections
    import socket
    for _ in range(30):
        try:
            with socket.create_connection(("127.0.0.1", internal_port), timeout=1):
                break
        except OSError:
            time.sleep(0.5)
    else:
        sys.stderr.write("[mimo-proxy] WARNING: mimo serve may not have started\n")


def _run_mimo(prompt: str, timeout: int = 120) -> str:
    cmd = f"mimo run --attach {_mimo_server_url} --format json {json.dumps(prompt)}"
    result = subprocess.run(
        ["zsh", "-i", "-c", cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    texts: list[str] = []
    for line in result.stdout.splitlines():
        try:
            ev = json.loads(line)
            if ev.get("type") == "text":
                texts.append(ev["part"].get("text", ""))
        except Exception:
            pass
    return "".join(texts)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        messages = body.get("messages", [])
        user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
        prompt = user_msgs[-1] if user_msgs else ""
        if not prompt:
            self._error(400, "no user message")
            return
        try:
            t0 = time.time()
            text = _run_mimo(prompt, timeout=120)
            elapsed = time.time() - t0
        except subprocess.TimeoutExpired:
            self._error(504, "mimo run timed out")
            return
        except Exception as e:
            self._error(500, str(e))
            return
        if not text:
            self._error(502, "mimo returned empty response")
            return
        sys.stderr.write(f"[mimo-proxy] {elapsed:.1f}s  {repr(text[:80])}\n")
        resp = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "mimo-auto"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        data = json.dumps(resp, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, code: int, msg: str):
        body = json.dumps({"error": {"message": msg, "type": "proxy_error"}}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=19999, help="proxy listen port (default 19999)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--internal-port", type=int, default=19998,
                    help="interno mimo serve port (default 19998)")
    args = ap.parse_args()

    sys.stderr.write("[mimo-proxy] starting mimo serve on internal port "
                     f"{args.internal_port}…\n")
    _start_mimo_server(args.internal_port)
    sys.stderr.write(f"[mimo-proxy] mimo serve ready at {_mimo_server_url}\n")

    server = HTTPServer((args.host, args.port), _Handler)
    sys.stderr.write(
        f"[mimo-proxy] proxy listening on http://{args.host}:{args.port}/v1\n"
        f"  hetero_llm config:\n"
        f"    base_url: http://{args.host}:{args.port}/v1\n"
        f"    model:    mimo-auto\n"
        f"    api_key:  local\n"
        "  Ctrl-C to stop\n"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if _serve_proc:
            _serve_proc.terminate()


if __name__ == "__main__":
    main()
