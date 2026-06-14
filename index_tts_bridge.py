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
_MODEL_KEY: tuple[str, bool, bool, bool, bool, bool] | None = None
# Effective use_accel after any fallback (may differ from the requested value if
# flash_attn was unavailable at load time). Reported back in status/load/tts.
_ACCEL_EFFECTIVE: bool = False


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
        # GPT acceleration engine (needs the self-built flash_attn). ~3.3x faster.
        # If flash_attn is unavailable at load time, _load_model falls back to False.
        "use_accel": _bool_payload(payload, "use_accel", "INDEX_TTS_USE_ACCEL", True),
        # BigVGAN fused CUDA kernel: tiny end-to-end gain, only worth it for large
        # batch jobs; off by default. Loads from the prebuilt .pyd (patched load.py).
        "use_cuda_kernel": _bool_payload(payload, "use_cuda_kernel", "INDEX_TTS_USE_CUDA_KERNEL", False),
        "use_deepspeed": _bool_payload(payload, "use_deepspeed", "INDEX_TTS_USE_DEEPSPEED", False),
        "use_torch_compile": _bool_payload(payload, "use_torch_compile", "INDEX_TTS_USE_TORCH_COMPILE", True),
    }


def _resolve_generation_options(payload: dict) -> dict[str, int | float | bool]:
    return {
        "max_text_tokens_per_segment": _int_payload(payload, "max_text_tokens_per_segment", "INDEX_TTS_MAX_TEXT_TOKENS", 120),
        "use_random": _bool_payload(payload, "use_random", "INDEX_TTS_USE_RANDOM", False),
        "num_beams": _int_payload(payload, "num_beams", "INDEX_TTS_NUM_BEAMS", 1),
        # Recommended fast path (measured ~3.3x with accel, quality accepted):
        # greedy decode + reduced s2mel diffusion. All overridable via payload/env.
        "do_sample": _bool_payload(payload, "do_sample", "INDEX_TTS_DO_SAMPLE", False),
        "diffusion_steps": _int_payload(payload, "diffusion_steps", "INDEX_TTS_DIFFUSION_STEPS", 16),
        "inference_cfg_rate": _float_payload(payload, "inference_cfg_rate", "INDEX_TTS_INFERENCE_CFG_RATE", 0.3),
        "temperature": _float_payload(payload, "temperature", "INDEX_TTS_TEMPERATURE", 0.8),
        "top_p": _float_payload(payload, "top_p", "INDEX_TTS_TOP_P", 0.8),
        "top_k": _int_payload(payload, "top_k", "INDEX_TTS_TOP_K", 20),
    }


def _build_model(cfg_path: Path, model_dir: Path, load_options: dict, use_accel: bool) -> IndexTTS2:
    return IndexTTS2(
        cfg_path=str(cfg_path),
        model_dir=str(model_dir),
        use_fp16=bool(load_options["use_fp16"]),
        use_cuda_kernel=bool(load_options["use_cuda_kernel"]),
        use_deepspeed=bool(load_options["use_deepspeed"]),
        use_accel=bool(use_accel),
        use_torch_compile=bool(load_options["use_torch_compile"]),
    )


def _load_model(payload: dict) -> IndexTTS2:
    global _MODEL, _MODEL_KEY, _ACCEL_EFFECTIVE
    model_dir = _resolve_model_dir(payload.get("model_name"))
    cfg_path = model_dir / "config.yaml"
    load_options = _resolve_load_options(payload)
    requested_accel = bool(load_options["use_accel"])
    key = (
        str(model_dir),
        bool(load_options["use_fp16"]),
        requested_accel,
        bool(load_options["use_cuda_kernel"]),
        bool(load_options["use_deepspeed"]),
        bool(load_options["use_torch_compile"]),
    )
    if _MODEL is None or _MODEL_KEY != key:
        try:
            _MODEL = _build_model(cfg_path, model_dir, load_options, requested_accel)
            _ACCEL_EFFECTIVE = requested_accel
        except Exception as exc:
            # Graceful fallback: flash_attn (or the accel engine) unavailable ->
            # rebuild without acceleration so generation still works.
            if not requested_accel:
                raise
            print(
                f">> use_accel=True failed ({exc.__class__.__name__}: {exc}); "
                f"falling back to use_accel=False.",
                file=sys.stderr,
            )
            _MODEL = _build_model(cfg_path, model_dir, load_options, False)
            _ACCEL_EFFECTIVE = False
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
    load_options = _resolve_load_options(payload)
    load_options["use_accel_effective"] = _ACCEL_EFFECTIVE
    return {
        "ok": True,
        "python_executable": sys.executable,
        "index_tts_root": str(INDEX_TTS_ROOT),
        "model_dir": str(_resolve_model_dir(payload.get("model_name"))),
        "sample_rate": getattr(model, "sampling_rate", None),
        "load_options": load_options,
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
        "do_sample": bool(generation_options["do_sample"]),
        "diffusion_steps": int(generation_options["diffusion_steps"]),
        "inference_cfg_rate": float(generation_options["inference_cfg_rate"]),
        "temperature": float(generation_options["temperature"]),
        "top_p": float(generation_options["top_p"]),
        "top_k": int(generation_options["top_k"]),
    }
    if style:
        kwargs["use_emo_text"] = True
        kwargs["emo_text"] = style

    result = model.infer(**kwargs)
    load_options = _resolve_load_options(payload)
    load_options["use_accel_effective"] = _ACCEL_EFFECTIVE
    return {
        "ok": True,
        "file": str(output_path),
        "result": str(result) if result is not None else str(output_path),
        "model_dir": str(_resolve_model_dir(payload.get("model_name"))),
        "python_executable": sys.executable,
        "load_options": load_options,
        "generation_options": generation_options,
    }


HANDLERS = {
    "status": run_status,
    "load": run_load,
    "tts": run_tts,
}


def run_serve() -> int:
    """Persistent mode: load the model once, then serve newline-delimited JSON
    requests on stdin and write newline-delimited JSON responses on stdout.

    This avoids reloading the ~6GB model (and re-running torch.compile / accel
    warmup) for every segment, and lets IndexTTS2's single-slot reference-audio
    cache survive across segments so that ref-audio grouping actually pays off.

    Protocol (one JSON object per line):
      request : {"_id": <any>, "_action": "tts"|"load"|"status"|"shutdown", ...payload}
      response: {"_id": <same>, "ok": true/false, ...}
    All model/torch chatter is redirected to stderr; stdout carries only responses.
    """
    real_stdout = sys.stdout
    # Announce readiness on stderr (stdout is reserved for the JSON protocol).
    print(">> index_tts_bridge serve: ready", file=sys.stderr, flush=True)
    for raw in sys.stdin:
        line = raw.strip().lstrip("﻿")
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as exc:
            real_stdout.write(json.dumps({"ok": False, "error": f"bad request json: {exc}"}) + "\n")
            real_stdout.flush()
            continue
        req_id = req.get("_id")
        action = str(req.get("_action") or "tts").strip().lower()
        if action == "shutdown":
            real_stdout.write(json.dumps({"_id": req_id, "ok": True, "shutdown": True}) + "\n")
            real_stdout.flush()
            return 0
        handler = HANDLERS.get(action)
        if handler is None:
            result: dict = {"ok": False, "error": f"unsupported action: {action}"}
        else:
            try:
                with redirect_stdout(sys.stderr):
                    result = handler(req)
            except Exception as exc:
                result = {"ok": False, "error": f"{exc.__class__.__name__}: {exc}"}
        result["_id"] = req_id
        real_stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
        real_stdout.flush()
    return 0


def main() -> int:
    action = (sys.argv[1] if len(sys.argv) > 1 else "status").strip().lower()
    if action == "serve":
        return run_serve()
    payload = _read_payload()
    if action not in HANDLERS:
        _print_json({"ok": False, "error": f"unsupported action: {action}"})
        return 2
    try:
        with redirect_stdout(sys.stderr):
            result = HANDLERS[action](payload)
        _print_json(result)
        return 0
    except Exception as exc:
        _print_json({"ok": False, "error": f"{exc.__class__.__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
