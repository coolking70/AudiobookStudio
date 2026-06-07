import json
import os
import sys
from pathlib import Path


def _suppress_windows_error_dialogs() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes

        sem_fail_critical_errors = 0x0001
        sem_no_gp_fault_error_box = 0x0002
        sem_no_open_file_error_box = 0x8000
        ctypes.windll.kernel32.SetErrorMode(
            sem_fail_critical_errors | sem_no_gp_fault_error_box | sem_no_open_file_error_box
        )
    except Exception:
        pass


_suppress_windows_error_dialogs()

from pipeline import VoxCPMPipeline


def _read_payload() -> dict:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    return json.loads(raw)


def _print_json(data: dict) -> None:
    result_path = str(os.getenv("VOXCPM_BRIDGE_RESULT_PATH") or "").strip()
    if result_path:
        try:
            target = Path(result_path).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
    sys.stdout.write(json.dumps(data, ensure_ascii=False))
    sys.stdout.flush()


def run_status(_: dict) -> dict:
    return {
        "ok": True,
        "python_executable": sys.executable,
        "force_dtype": os.getenv("VOXCPM_FORCE_DTYPE", ""),
    }


def run_load(payload: dict) -> dict:
    pipeline = VoxCPMPipeline(
        model_name=payload.get("model_name") or "",
        device=payload.get("device") or None,
    )
    pipeline.load()
    try:
        return {
            "ok": True,
            "device": pipeline.device,
            "model_name": pipeline.model_name,
            "python_executable": sys.executable,
            "force_dtype": os.getenv("VOXCPM_FORCE_DTYPE", ""),
        }
    finally:
        pipeline.unload()


def run_tts(payload: dict) -> dict:
    output_path = Path(str(payload.get("output_path") or "")).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline = VoxCPMPipeline(
        model_name=payload.get("model_name") or "",
        device=payload.get("device") or None,
    )
    pipeline.synthesize_segment(
        text=str(payload.get("text") or ""),
        output_path=output_path,
        instruct=payload.get("instruct"),
        ref_audio=payload.get("ref_audio"),
        ref_text=payload.get("ref_text"),
        cfg_value=float(payload.get("cfg_value") if payload.get("cfg_value") is not None else 2.0),
        inference_timesteps=int(payload.get("inference_timesteps") if payload.get("inference_timesteps") is not None else 10),
    )
    return {
        "ok": True,
        "file": str(output_path),
        "device": pipeline.device,
        "model_name": pipeline.model_name,
        "python_executable": sys.executable,
        "force_dtype": os.getenv("VOXCPM_FORCE_DTYPE", ""),
    }


def main() -> int:
    action = (sys.argv[1] if len(sys.argv) > 1 else "status").strip().lower()
    payload = _read_payload()
    handlers = {
        "status": run_status,
        "load": run_load,
        "tts": run_tts,
    }
    if action not in handlers:
        _print_json({"ok": False, "error": f"unsupported action: {action}"})
        return 2
    try:
        result = handlers[action](payload)
        _print_json(result)
        return 0
    except Exception as exc:
        _print_json(
            {
                "ok": False,
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
