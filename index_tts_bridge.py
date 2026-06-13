import json
import os
import sys
from contextlib import redirect_stdout
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

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_INDEX_TTS_ROOT = ROOT_DIR.parent / "index-tts"
INDEX_TTS_ROOT = Path(os.getenv("INDEX_TTS_ROOT") or DEFAULT_INDEX_TTS_ROOT).expanduser().resolve()
if INDEX_TTS_ROOT.exists():
    sys.path.insert(0, str(INDEX_TTS_ROOT))
    sys.path.insert(0, str(INDEX_TTS_ROOT / "indextts"))

from indextts.infer_v2 import IndexTTS2

_MODEL: IndexTTS2 | None = None
_MODEL_KEY: tuple[str, bool, bool, bool, bool] | None = None


def _read_payload() -> dict:
    raw = sys.stdin.read().lstrip("\ufeff")
    if not raw.strip():
        return {}
    return json.loads(raw)


def _print_json(data: dict) -> None:
    result_path = str(os.getenv("INDEX_TTS_BRIDGE_RESULT_PATH") or "").strip()
    if result_path:
        try:
            target = Path(result_path).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
    sys.stdout.write(json.dumps(data, ensure_ascii=False))
    sys.stdout.flush()


def _resolve_model_dir(model_name: object = None) -> Path:
    configured = str(model_name or os.getenv("INDEX_TTS_MODEL_DIR") or "").strip()
    model_dir = Path(configured).expanduser() if configured else INDEX_TTS_ROOT / "checkpoints"
    return model_dir.resolve()


def _bool_payload(payload: dict, key: str, env_key: str, default: bool = False) -> bool:
    if payload.get(key) is not None:
        return str(payload.get(key)).strip().lower() in {"1", "true", "yes", "on"}
    raw = os.getenv(env_key, "").strip().lower()
    if raw:
        return raw in {"1", "true", "yes", "on"}
    return default


def _int_payload(payload: dict, key: str, env_key: str, default: int) -> int:
    value = payload.get(key)
    if value is None:
        value = os.getenv(env_key)
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def _float_payload(payload: dict, key: str, env_key: str, default: float) -> float:
    value = payload.get(key)
    if value is None:
        value = os.getenv(env_key)
    if value is None or str(value).strip() == "":
        return default
    return float(value)


def _resolve_load_options(payload: dict) -> dict[str, bool]:
    return {
        "use_fp16": _bool_payload(payload, "use_fp16", "INDEX_TTS_USE_FP16", True),
        "use_cuda_kernel": _bool_payload(payload, "use_cuda_kernel", "INDEX_TTS_USE_CUDA_KERNEL", False),
        "use_deepspeed": _bool_payload(payload, "use_deepspeed", "INDEX_TTS_USE_DEEPSPEED", False),
        "use_torch_compile": _bool_payload(payload, "use_torch_compile", "INDEX_TTS_USE_TORCH_COMPILE", True),
    }


def _resolve_generation_options(payload: dict) -> dict[str, int | float | bool]:
    return {
        "max_text_tokens_per_segment": _int_payload(payload, "max_text_tokens_per_segment", "INDEX_TTS_MAX_TEXT_TOKENS", 120),
        "use_random": _bool_payload(payload, "use_random", "INDEX_TTS_USE_RANDOM", False),
        "num_beams": _int_payload(payload, "num_beams", "INDEX_TTS_NUM_BEAMS", 1),
        "temperature": _float_payload(payload, "temperature", "INDEX_TTS_TEMPERATURE", 0.8),
        "top_p": _float_payload(payload, "top_p", "INDEX_TTS_TOP_P", 0.8),
        "top_k": _int_payload(payload, "top_k", "INDEX_TTS_TOP_K", 20),
    }


def _load_model(payload: dict) -> IndexTTS2:
    global _MODEL, _MODEL_KEY
    model_dir = _resolve_model_dir(payload.get("model_name"))
    cfg_path = model_dir / "config.yaml"
    load_options = _resolve_load_options(payload)
    key = (
        str(model_dir),
        bool(load_options["use_fp16"]),
        bool(load_options["use_cuda_kernel"]),
        bool(load_options["use_deepspeed"]),
        bool(load_options["use_torch_compile"]),
    )
    if _MODEL is None or _MODEL_KEY != key:
        _MODEL = IndexTTS2(
            cfg_path=str(cfg_path),
            model_dir=str(model_dir),
            use_fp16=bool(load_options["use_fp16"]),
            use_cuda_kernel=bool(load_options["use_cuda_kernel"]),
            use_deepspeed=bool(load_options["use_deepspeed"]),
            use_torch_compile=bool(load_options["use_torch_compile"]),
        )
        _MODEL_KEY = key
    return _MODEL


def run_status(_: dict) -> dict:
    payload: dict = {}
    return {
        "ok": True,
        "python_executable": sys.executable,
        "index_tts_root": str(INDEX_TTS_ROOT),
        "model_dir": str(_resolve_model_dir()),
        "load_options": _resolve_load_options(payload),
        "generation_options": _resolve_generation_options(payload),
    }


def run_load(payload: dict) -> dict:
    model = _load_model(payload)
    return {
        "ok": True,
        "python_executable": sys.executable,
        "index_tts_root": str(INDEX_TTS_ROOT),
        "model_dir": str(_resolve_model_dir(payload.get("model_name"))),
        "sample_rate": getattr(model, "sampling_rate", None),
        "load_options": _resolve_load_options(payload),
        "generation_options": _resolve_generation_options(payload),
    }


def run_tts(payload: dict) -> dict:
    text = str(payload.get("text") or "").strip()
    if not text:
        raise RuntimeError("IndexTTS 缺少待合成文本。")
    ref_audio = str(payload.get("ref_audio") or "").strip()
    if not ref_audio:
        raise RuntimeError("IndexTTS2 需要 ref_audio 音色参考音频。请先为角色或片段选择参考音频。")

    output_path = Path(str(payload.get("output_path") or "")).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = _load_model(payload)

    style = str(payload.get("instruct") or "").strip()
    emo_audio = str(payload.get("emo_audio") or "").strip() or None
    emo_alpha = float(payload.get("emo_alpha") if payload.get("emo_alpha") is not None else 0.6)
    generation_options = _resolve_generation_options(payload)

    kwargs = {
        "spk_audio_prompt": ref_audio,
        "text": text,
        "output_path": str(output_path),
        "emo_audio_prompt": emo_audio,
        "emo_alpha": emo_alpha,
        "use_random": bool(generation_options["use_random"]),
        "verbose": _bool_payload(payload, "verbose", "INDEX_TTS_VERBOSE"),
        "max_text_tokens_per_segment": int(generation_options["max_text_tokens_per_segment"]),
        "num_beams": int(generation_options["num_beams"]),
        "temperature": float(generation_options["temperature"]),
        "top_p": float(generation_options["top_p"]),
        "top_k": int(generation_options["top_k"]),
    }
    if style:
        kwargs["use_emo_text"] = True
        kwargs["emo_text"] = style

    result = model.infer(**kwargs)
    return {
        "ok": True,
        "file": str(output_path),
        "result": str(result) if result is not None else str(output_path),
        "model_dir": str(_resolve_model_dir(payload.get("model_name"))),
        "python_executable": sys.executable,
        "load_options": _resolve_load_options(payload),
        "generation_options": generation_options,
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
        with redirect_stdout(sys.stderr):
            result = handlers[action](payload)
        _print_json(result)
        return 0
    except Exception as exc:
        _print_json({"ok": False, "error": f"{exc.__class__.__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
