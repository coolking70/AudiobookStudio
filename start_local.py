from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import uvicorn


REQUIRED_IMPORTS = ("fastapi", "uvicorn", "httpx", "soundfile", "pydantic", "numpy")
OPTIONAL_AUDIO_IMPORTS = ("torch", "torchaudio", "omnivoice", "imageio_ffmpeg")
TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}
FALSY_ENV_VALUES = {"0", "false", "no", "n", "off"}


def clear_proxy_env() -> None:
    for key in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        os.environ.pop(key, None)


def _run_pid_command(command: list[str]) -> set[int]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except Exception:
        return set()
    if completed.returncode not in {0, 1}:
        return set()

    pids: set[int] = set()
    for token in (completed.stdout or "").replace(",", " ").split():
        if token.isdigit():
            pid = int(token)
            if pid != os.getpid():
                pids.add(pid)
    return pids


def find_listening_pids(port: int) -> set[int]:
    if sys.platform.startswith("win"):
        try:
            completed = subprocess.run(
                ["netstat", "-ano", "-p", "tcp"],
                capture_output=True,
                text=True,
                check=False,
                timeout=8,
            )
        except Exception:
            return set()
        pids: set[int] = set()
        needle = f":{port}"
        for line in (completed.stdout or "").splitlines():
            parts = line.split()
            if len(parts) < 5 or parts[0].upper() != "TCP":
                continue
            local_addr, state, pid_text = parts[1], parts[3].upper(), parts[4]
            if state == "LISTENING" and local_addr.endswith(needle) and pid_text.isdigit():
                pid = int(pid_text)
                if pid != os.getpid():
                    pids.add(pid)
        return pids

    pids = _run_pid_command(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"])
    if pids:
        return pids
    return _run_pid_command(["fuser", f"{port}/tcp"])


def describe_pid(pid: int) -> str:
    if sys.platform.startswith("win"):
        try:
            completed = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except Exception:
            return str(pid)
        line = (completed.stdout or "").strip().splitlines()
        if not line or "No tasks" in line[0]:
            return str(pid)
        return f"{pid} {line[0]}"

    try:
        completed = subprocess.run(
            ["ps", "-p", str(pid), "-o", "pid=,command="],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except Exception:
        return str(pid)
    return (completed.stdout or "").strip() or str(pid)


def terminate_processes(pids: set[int]) -> None:
    if sys.platform.startswith("win"):
        for pid in sorted(pids):
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False)
        return

    for pid in sorted(pids):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.time() + 5
    remaining = set(pids)
    while remaining and time.time() < deadline:
        remaining = {pid for pid in remaining if _pid_exists(pid)}
        if remaining:
            time.sleep(0.2)

    for pid in sorted(remaining):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def resolve_kill_existing_preference() -> bool | None:
    raw = os.getenv("AUDIOBOOKSTUDIO_KILL_EXISTING", "").strip().lower()
    if raw in TRUTHY_ENV_VALUES:
        return True
    if raw in FALSY_ENV_VALUES:
        return False
    return None


def ensure_port_available(host: str, port: int) -> None:
    pids = find_listening_pids(port)
    if not pids:
        return

    print(f"[AudiobookStudio] Detected existing listener on {host}:{port}:")
    for pid in sorted(pids):
        print(f"  - {describe_pid(pid)}")

    preference = resolve_kill_existing_preference()
    if preference is None:
        if not sys.stdin.isatty():
            print(
                "No interactive terminal is available. Set AUDIOBOOKSTUDIO_KILL_EXISTING=1 "
                "to stop the existing process automatically, or change PORT."
            )
            raise SystemExit(1)
        answer = input("Kill the existing process(es) and continue? [y/N] ").strip().lower()
        preference = answer in TRUTHY_ENV_VALUES

    if not preference:
        print("Startup cancelled. Change PORT or stop the existing process manually.")
        raise SystemExit(1)

    terminate_processes(pids)
    remaining = find_listening_pids(port)
    if remaining:
        print(f"Could not stop all process(es) on port {port}: {sorted(remaining)}")
        raise SystemExit(1)
    print(f"Stopped existing process(es) on port {port}.")


def candidate_python_binaries() -> list[Path]:
    script_dir = Path(__file__).resolve().parent
    candidates: list[Path] = []

    env_python = os.getenv("AUDIOBOOKSTUDIO_PYTHON", "").strip()
    if env_python:
        candidates.append(Path(env_python).expanduser())

    conda_prefix = os.getenv("CONDA_PREFIX", "").strip()
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "python.exe")
        candidates.append(Path(conda_prefix) / "bin" / "python")

    candidates.extend(
        [
            Path(sys.executable),
            Path(r"I:\conda_envs\omnivoice\python.exe"),
            script_dir / ".venv" / "Scripts" / "python.exe",
            script_dir / ".venv" / "bin" / "python",
        ]
    )

    resolved: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            normalized = str(candidate.resolve())
        except Exception:
            normalized = str(candidate)
        if normalized in seen or not candidate.exists():
            continue
        seen.add(normalized)
        resolved.append(candidate)
    return resolved


def inspect_python_environment(python_bin: Path) -> dict[str, object] | None:
    probe = f"""
import importlib
import json
import sys

required = {REQUIRED_IMPORTS!r}
optional = {OPTIONAL_AUDIO_IMPORTS!r}

def check(names):
    result = {{}}
    for name in names:
        try:
            mod = importlib.import_module(name)
            result[name] = getattr(mod, "__file__", True) or True
        except Exception:
            result[name] = False
    return result

print(json.dumps({{
    "executable": sys.executable,
    "version": sys.version.split()[0],
    "required": check(required),
    "optional": check(optional),
}}, ensure_ascii=False))
"""
    try:
        completed = subprocess.run(
            [str(python_bin), "-c", probe],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
    except Exception:
        return None

    stdout = (completed.stdout or "").strip().splitlines()
    if not stdout:
        return None
    try:
        return json.loads(stdout[-1])
    except json.JSONDecodeError:
        return None


def score_python_environment(info: dict[str, object]) -> tuple[int, int]:
    required = info.get("required", {})
    optional = info.get("optional", {})
    required_score = sum(1 for ok in required.values() if ok)
    optional_score = sum(1 for ok in optional.values() if ok)
    return required_score, optional_score


def maybe_switch_to_best_python() -> None:
    if os.getenv("AUDIOBOOKSTUDIO_PYTHON_LOCKED", "").strip().lower() in {"1", "true", "yes", "on"}:
        return

    inspections: list[tuple[Path, dict[str, object]]] = []
    for candidate in candidate_python_binaries():
        info = inspect_python_environment(candidate)
        if info is not None:
            inspections.append((candidate, info))

    if not inspections:
        return

    current_python = Path(sys.executable).resolve()
    best_candidate, best_info = max(
        inspections,
        key=lambda item: (
            *score_python_environment(item[1]),
            str(item[0]).lower() == str(Path(r"I:\conda_envs\omnivoice\python.exe")).lower(),
        ),
    )
    best_required, best_optional = score_python_environment(best_info)
    current_info = next((info for candidate, info in inspections if Path(candidate).resolve() == current_python), None)
    current_required, current_optional = score_python_environment(current_info or {"required": {}, "optional": {}})

    if Path(best_candidate).resolve() == current_python:
        return
    if best_required < len(REQUIRED_IMPORTS):
        return
    if (best_required, best_optional) <= (current_required, current_optional):
        return

    env = os.environ.copy()
    env["AUDIOBOOKSTUDIO_PYTHON_LOCKED"] = "1"
    env["AUDIOBOOKSTUDIO_PYTHON"] = str(best_candidate)
    raise SystemExit(subprocess.call([str(best_candidate), str(Path(__file__).resolve())], env=env))


def resolve_hf_snapshot(repo_dir: Path) -> Path | None:
    if not repo_dir.exists():
        return None
    direct_model = repo_dir / "model.safetensors"
    if direct_model.exists():
        return repo_dir

    refs_main = repo_dir / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot_dir = repo_dir / "snapshots" / revision
        if snapshot_dir.exists():
            return snapshot_dir

    snapshot_root = repo_dir / "snapshots"
    if snapshot_root.exists():
        snapshots = sorted(
            (item for item in snapshot_root.iterdir() if item.is_dir()),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for snapshot_dir in snapshots:
            if (snapshot_dir / "model.safetensors").exists():
                return snapshot_dir
    return None


def auto_configure_runtime_paths() -> None:
    cache_candidates = [
        Path(os.getenv("HUGGINGFACE_HUB_CACHE", "")).expanduser() if os.getenv("HUGGINGFACE_HUB_CACHE") else None,
        Path(os.getenv("HF_HUB_CACHE", "")).expanduser() if os.getenv("HF_HUB_CACHE") else None,
        Path(r"I:\hf_cache"),
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    cache_root = next((path for path in cache_candidates if path and path.exists()), None)

    if cache_root is not None:
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root))
        os.environ.setdefault("HF_HUB_CACHE", str(cache_root))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root))

        if not os.getenv("OMNIVOICE_MODEL_PATH"):
            model_path = resolve_hf_snapshot(cache_root / "models--k2-fsa--OmniVoice")
            if model_path is not None:
                os.environ["OMNIVOICE_MODEL_PATH"] = str(model_path)

        if not os.getenv("OMNIVOICE_ASR_MODEL_PATH"):
            asr_path = resolve_hf_snapshot(cache_root / "models--openai--whisper-large-v3-turbo")
            if asr_path is not None:
                os.environ["OMNIVOICE_ASR_MODEL_PATH"] = str(asr_path)


def main() -> None:
    clear_proxy_env()
    maybe_switch_to_best_python()
    auto_configure_runtime_paths()
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    reload_enabled = os.getenv("RELOAD", "").strip().lower() in {"1", "true", "yes", "on"}
    ensure_port_available(host, port)
    uvicorn.run("app:app", host=host, port=port, reload=reload_enabled)


if __name__ == "__main__":
    main()
