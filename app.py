import base64
import json
import mimetypes
import os
import platform
import re
import traceback
from pathlib import Path
from threading import Lock
from urllib.parse import urljoin
from uuid import uuid4

import httpx
import psutil
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from audio_utils import build_lrc, join_wavs
from llm_client import OpenAICompatibleClient
from pipeline import OmniVoicePipeline
from role_analyzer import analyze_chunks_with_llm, analyze_text_with_llm, intelligent_segment_text, optimize_chunks_for_tts, sanitize_segments
from schemas import AnalyzeChunksRequest, AutoNarrateRequest, BatchImportVoiceLibraryRequest, ChunkOptimizeRequest, ConnectivityTestRequest, ImportSegmentsRequest, MergeRequest, ModelLoadRequest, NarrateRequest, ParseRequest, SegmentPlanRequest, TTSRequest, UploadRefAudioRequest


app = FastAPI(title="OmniVoice Reader Studio")
pipeline_cache: dict[str, OmniVoicePipeline] = {}
pipeline_lock = Lock()

Path("outputs").mkdir(parents=True, exist_ok=True)
Path("static").mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

OUTPUT_DIR = Path("outputs")
REF_AUDIO_DIR = OUTPUT_DIR / "ref_audio"
AI_HELPER_CONFIG_PATH = Path(__file__).resolve().parent.parent / "tts_text_parse" / "ai-helper" / "ai-api.md"


def sanitize_upload_filename(filename: str) -> str:
    name = Path(filename or "ref_audio.wav").name
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return safe or "ref_audio.wav"


def build_unique_upload_filename(filename: str) -> str:
    safe_name = sanitize_upload_filename(filename)
    path = Path(safe_name)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem) or "ref_audio"
    suffix = path.suffix or ".wav"
    return f"{stem}_{uuid4().hex[:10]}{suffix}"


def save_ref_audio_from_base64(filename: str, content_base64: str) -> Path:
    REF_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = build_unique_upload_filename(filename)
    target = REF_AUDIO_DIR / safe_name

    raw = content_base64
    if "," in raw:
        raw = raw.split(",", 1)[1]

    content = base64.b64decode(raw)
    target.write_bytes(content)
    return target


def build_lyric_lines(segments: list[dict]) -> list[str]:
    lines = []
    for seg in segments:
        speaker = str(seg.get("speaker", "") or "").strip()
        text = str(seg.get("text", "") or "").strip()
        if not text:
            lines.append("")
            continue
        if speaker and speaker != "旁白":
            lines.append(f"{speaker}：{text}")
        else:
            lines.append(text)
    return lines


def sanitize_output_name(name: str, fallback: str = "result") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name or "").strip())
    cleaned = cleaned.strip("._")
    return cleaned or fallback


def build_segment_wav_paths(base_name: str, segment_count: int) -> list[Path]:
    return [OUTPUT_DIR / f"{base_name}_{idx:03d}.wav" for idx in range(1, segment_count + 1)]


def raise_api_error(exc: Exception) -> None:
    traceback.print_exc()
    raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


def load_ai_helper_llm_defaults() -> dict[str, object]:
    if not AI_HELPER_CONFIG_PATH.exists():
        raise FileNotFoundError(f"找不到 ai-helper 配置文件: {AI_HELPER_CONFIG_PATH}")

    text = AI_HELPER_CONFIG_PATH.read_text(encoding="utf-8")
    api_key_match = re.search(r"API Key[：:]\s*(\S+)", text)
    base_url_match = re.search(r"Base URL[：:]\s*(\S+)", text)
    models_match = re.search(r"可用model[：:]\s*(.+)", text)

    if not api_key_match or not base_url_match:
        raise RuntimeError(f"无法从 {AI_HELPER_CONFIG_PATH} 解析 API Key 或 Base URL")

    available_models = []
    if models_match:
        available_models = [item.strip() for item in models_match.group(1).split(";") if item.strip()]

    preferred_model = "qwen3-max-2026-01-23"
    model = preferred_model if preferred_model in available_models else (available_models[0] if available_models else preferred_model)

    return {
        "base_url": base_url_match.group(1),
        "api_key": api_key_match.group(1),
        "model": model,
        "temperature": 0.3,
        "max_tokens": 8000,
        "analysis_mode": "balanced",
        "compatibility_mode": "chat_compat",
        "system_prompt": None,
        "source": str(AI_HELPER_CONFIG_PATH),
        "available_models": available_models,
    }


def normalize_remote_base_url(base_url: str | None) -> str:
    normalized = str(base_url or "").strip().rstrip("/")
    if not normalized:
        raise ValueError("远程 API Base URL 不能为空")
    return normalized


def absolutize_remote_file_url(base_url: str, file_url: str | None) -> str | None:
    if not file_url:
        return file_url
    if file_url.startswith("http://") or file_url.startswith("https://"):
        return file_url
    return urljoin(f"{base_url}/", file_url.lstrip("/"))


def remote_post_json(base_url: str, path: str, payload: dict, api_key: str | None = None) -> dict:
    url = f"{normalize_remote_base_url(base_url)}{path}"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    timeout = httpx.Timeout(connect=30.0, read=300.0, write=60.0, pool=30.0)
    with httpx.Client(timeout=timeout, trust_env=False) as client:
        resp = client.post(url, headers=headers, json=payload)
        data = resp.json()
        if not resp.is_success:
            raise RuntimeError(data.get("detail") or data)
        return data


def remote_get_json(base_url: str, path: str, api_key: str | None = None) -> dict:
    url = f"{normalize_remote_base_url(base_url)}{path}"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    timeout = httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0)
    with httpx.Client(timeout=timeout, trust_env=False) as client:
        resp = client.get(url, headers=headers)
        data = resp.json()
        if not resp.is_success:
            raise RuntimeError(data.get("detail") or data)
        return data


def normalize_inference_device(device: str | None) -> str:
    normalized = str(device or "").strip().lower()
    if not normalized or normalized == "auto":
        return OmniVoicePipeline.detect_device()
    if normalized == "cuda:0":
        return "cuda"
    if normalized in {"cuda", "mps", "cpu"}:
        return normalized
    raise ValueError(f"不支持的推理设备: {device}")


def get_pipeline(device: str | None = None) -> OmniVoicePipeline:
    resolved_device = normalize_inference_device(device)
    with pipeline_lock:
        pipeline = pipeline_cache.get(resolved_device)
        if pipeline is None:
            pipeline = OmniVoicePipeline(device=resolved_device)
            pipeline_cache[resolved_device] = pipeline
        return pipeline


def unload_all_audio_pipelines() -> None:
    with pipeline_lock:
        pipelines = list(pipeline_cache.values())
    for pipeline in pipelines:
        pipeline.unload()


def resolve_audio_backend(req_backend, fallback_device: str | None = None) -> tuple[str, str | None, str | None, str | None]:
    backend = req_backend or {}
    mode = str(backend.get("mode") or "local").strip().lower()
    remote_base_url = backend.get("remote_base_url")
    remote_api_key = backend.get("remote_api_key")
    inference_device = backend.get("inference_device") or fallback_device
    return mode, remote_base_url, remote_api_key, inference_device


def is_mps_oom_error(exc: Exception) -> bool:
    message = str(exc or "").lower()
    return "mps backend out of memory" in message or ("out of memory" in message and "mps" in message)


def synthesize_local_tts_with_fallback(
    *,
    inference_device: str | None,
    text: str,
    output_path: Path,
    instruct: str | None = None,
    ref_audio: str | None = None,
    ref_text: str | None = None,
) -> dict:
    resolved_device = normalize_inference_device(inference_device)
    pipeline = get_pipeline(resolved_device)
    try:
        pipeline.synthesize_segment(
            text=text,
            output_path=output_path,
            instruct=instruct,
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
        return {"device": resolved_device, "fallback_device": None}
    except Exception as exc:
        if resolved_device != "mps" or not is_mps_oom_error(exc):
            raise
        pipeline.unload()
        cpu_pipeline = get_pipeline("cpu")
        cpu_pipeline.synthesize_segment(
            text=text,
            output_path=output_path,
            instruct=instruct,
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
        return {"device": "cpu", "fallback_device": "cpu", "original_device": resolved_device}


def synthesize_local_narration_with_fallback(
    *,
    inference_device: str | None,
    segments: list[dict],
    output_name: str,
    silence_ms: int,
    role_profiles: dict,
) -> dict:
    resolved_device = normalize_inference_device(inference_device)
    pipeline = get_pipeline(resolved_device)
    try:
        merged_path = pipeline.synthesize_segments(
            segments=segments,
            base_name=output_name,
            silence_ms=silence_ms,
            role_profiles=role_profiles,
        )
        return {"merged_path": merged_path, "device": resolved_device, "fallback_device": None}
    except Exception as exc:
        if resolved_device != "mps" or not is_mps_oom_error(exc):
            raise
        pipeline.unload()
        cpu_pipeline = get_pipeline("cpu")
        merged_path = cpu_pipeline.synthesize_segments(
            segments=segments,
            base_name=output_name,
            silence_ms=silence_ms,
            role_profiles=role_profiles,
        )
        return {
            "merged_path": merged_path,
            "device": "cpu",
            "fallback_device": "cpu",
            "original_device": resolved_device,
        }


def get_audio_model_status() -> dict:
    with pipeline_lock:
        loaded_devices = sorted([device for device, pipeline in pipeline_cache.items() if pipeline.model is not None])
        cached_devices = sorted(list(pipeline_cache.keys()))
    return {
        "loaded_devices": loaded_devices,
        "cached_devices": cached_devices,
        "detected_device": OmniVoicePipeline.detect_device(),
    }


def get_system_status() -> dict:
    memory = psutil.virtual_memory()
    process = psutil.Process(os.getpid())
    cpu_temp = None
    cpu_temp_source = None
    try:
        temps = psutil.sensors_temperatures()
        for name, entries in (temps or {}).items():
            if not entries:
                continue
            entry = entries[0]
            cpu_temp = getattr(entry, "current", None)
            cpu_temp_source = name
            if cpu_temp is not None:
                break
    except Exception:
        pass

    gpu = {
        "cuda": {
            "available": torch.cuda.is_available(),
        },
        "mps": {
            "available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        },
    }
    if gpu["cuda"]["available"]:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        gpu["cuda"].update(
            {
                "free_gb": round(free_bytes / (1024 ** 3), 2),
                "total_gb": round(total_bytes / (1024 ** 3), 2),
                "used_gb": round((total_bytes - free_bytes) / (1024 ** 3), 2),
            }
        )
    if gpu["mps"]["available"] and hasattr(torch, "mps"):
        try:
            gpu["mps"].update(
                {
                    "current_allocated_gb": round(torch.mps.current_allocated_memory() / (1024 ** 3), 3),
                    "driver_allocated_gb": round(torch.mps.driver_allocated_memory() / (1024 ** 3), 3),
                    "recommended_max_gb": round(torch.mps.recommended_max_memory() / (1024 ** 3), 3),
                }
            )
        except Exception:
            pass

    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "cpu": {
            "logical_count": psutil.cpu_count(logical=True),
            "usage_percent": psutil.cpu_percent(interval=0.2),
            "load_avg": [round(x, 2) for x in os.getloadavg()] if hasattr(os, "getloadavg") else None,
            "temperature_c": cpu_temp,
            "temperature_source": cpu_temp_source,
        },
        "memory": {
            "total_gb": round(memory.total / (1024 ** 3), 2),
            "used_gb": round(memory.used / (1024 ** 3), 2),
            "available_gb": round(memory.available / (1024 ** 3), 2),
            "usage_percent": memory.percent,
            "process_rss_gb": round(process.memory_info().rss / (1024 ** 3), 3),
        },
        "gpu": gpu,
    }


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = Path("static/index.html")
    if not html_path.exists():
        return "<h1>index.html 不存在</h1>"
    return html_path.read_text(encoding="utf-8")


@app.get("/api/llm-defaults")
def llm_defaults():
    try:
        return {
            "ok": True,
            "llm": load_ai_helper_llm_defaults(),
        }
    except Exception as exc:
        raise_api_error(exc)


@app.get("/favicon.ico")
def favicon():
    icon_path = Path("static/favicon.svg")
    if not icon_path.exists():
        raise HTTPException(status_code=404, detail="favicon 不存在")
    return FileResponse(icon_path, media_type="image/svg+xml", filename="favicon.svg")


@app.post("/api/tts")
def tts(req: TTSRequest):
    try:
        output_name = sanitize_output_name(req.output_name, "tts_result")
        mode, remote_base_url, remote_api_key, inference_device = resolve_audio_backend(
            req.backend.model_dump() if req.backend else None,
            req.inference_device,
        )

        if mode == "remote":
            remote_resp = remote_post_json(
                remote_base_url,
                "/api/tts",
                {
                    "text": req.text,
                    "instruct": req.instruct,
                    "ref_audio": req.ref_audio,
                    "ref_text": req.ref_text,
                    "output_name": output_name,
                    "inference_device": inference_device,
                },
                remote_api_key,
            )
            remote_resp["audio_url"] = absolutize_remote_file_url(remote_base_url, remote_resp.get("audio_url"))
            return remote_resp

        output_path = OUTPUT_DIR / f"{output_name}.wav"
        result = synthesize_local_tts_with_fallback(
            inference_device=inference_device,
            text=req.text,
            output_path=output_path,
            instruct=req.instruct,
            ref_audio=req.ref_audio,
            ref_text=req.ref_text,
        )

        return {
            "ok": True,
            "file": str(output_path),
            "audio_url": f"/file/{output_path.name}",
            "device": result.get("device"),
            "fallback_device": result.get("fallback_device"),
            "original_device": result.get("original_device"),
        }
    except Exception as exc:
        raise_api_error(exc)


@app.post("/api/parse")
def parse_text(req: ParseRequest):
    try:
        segments = analyze_text_with_llm(req.text, req.llm)
        return {
            "ok": True,
            "segments": segments,
        }
    except Exception as exc:
        raise_api_error(exc)


@app.post("/api/segment-plan")
def segment_plan(req: SegmentPlanRequest):
    try:
        chunks = intelligent_segment_text(req.text, req.llm)
        return {
            "ok": True,
            "chunks": chunks,
        }
    except Exception as exc:
        raise_api_error(exc)


@app.post("/api/optimize-chunks")
def optimize_chunks(req: ChunkOptimizeRequest):
    try:
        chunks = [item.model_dump() for item in req.chunks]
        optimized = optimize_chunks_for_tts(chunks, req.llm)
        return {
            "ok": True,
            "chunks": optimized,
        }
    except Exception as exc:
        raise_api_error(exc)


@app.post("/api/analyze-chunks")
def analyze_chunks(req: AnalyzeChunksRequest):
    try:
        chunks = [item.model_dump() for item in req.chunks]
        segments = analyze_chunks_with_llm(chunks, req.llm)
        return {
            "ok": True,
            "segments": segments,
        }
    except Exception as exc:
        raise_api_error(exc)


@app.post("/api/narrate")
def narrate(req: NarrateRequest):
    try:
        output_name = sanitize_output_name(req.output_name, "narration")
        segments = [seg.model_dump() for seg in req.segments]
        role_profiles = {k: v.model_dump() for k, v in (req.role_profiles or {}).items()}
        mode, remote_base_url, remote_api_key, inference_device = resolve_audio_backend(
            req.backend.model_dump() if req.backend else None,
            req.inference_device,
        )

        if mode == "remote":
            remote_resp = remote_post_json(
                remote_base_url,
                "/api/narrate",
                {
                    "segments": segments,
                    "silence_ms": req.silence_ms,
                    "output_name": output_name,
                    "role_profiles": role_profiles,
                    "inference_device": inference_device,
                },
                remote_api_key,
            )
            remote_resp["audio_url"] = absolutize_remote_file_url(remote_base_url, remote_resp.get("audio_url"))
            remote_resp["lrc_url"] = absolutize_remote_file_url(remote_base_url, remote_resp.get("lrc_url"))
            return remote_resp

        result = synthesize_local_narration_with_fallback(
            inference_device=inference_device,
            segments=segments,
            output_name=output_name,
            silence_ms=req.silence_ms,
            role_profiles=role_profiles,
        )
        merged_path = result["merged_path"]
        wav_paths = build_segment_wav_paths(output_name, len(segments))
        lrc_path = OUTPUT_DIR / f"{output_name}_merged.lrc"
        build_lrc(wav_paths, build_lyric_lines(segments), lrc_path, silence_ms=req.silence_ms)
        return {
            "ok": True,
            "segments": len(segments),
            "file": merged_path,
            "audio_url": f"/file/{Path(merged_path).name}",
            "lrc_url": f"/file/{lrc_path.name}",
            "device": result.get("device"),
            "fallback_device": result.get("fallback_device"),
            "original_device": result.get("original_device"),
        }
    except Exception as exc:
        raise_api_error(exc)


@app.post("/api/auto-narrate")
def auto_narrate(req: AutoNarrateRequest):
    try:
        output_name = sanitize_output_name(req.output_name, "auto_narration")
        segments = analyze_text_with_llm(req.text, req.llm)
        role_profiles = {k: v.model_dump() for k, v in (req.role_profiles or {}).items()}
        mode, remote_base_url, remote_api_key, inference_device = resolve_audio_backend(
            req.backend.model_dump() if req.backend else None,
            req.inference_device,
        )

        if mode == "remote":
            remote_resp = remote_post_json(
                remote_base_url,
                "/api/narrate",
                {
                    "segments": segments,
                    "silence_ms": req.silence_ms,
                    "output_name": output_name,
                    "role_profiles": role_profiles,
                    "inference_device": inference_device,
                },
                remote_api_key,
            )
            remote_resp["segments"] = segments
            remote_resp["audio_url"] = absolutize_remote_file_url(remote_base_url, remote_resp.get("audio_url"))
            remote_resp["lrc_url"] = absolutize_remote_file_url(remote_base_url, remote_resp.get("lrc_url"))
            return remote_resp

        result = synthesize_local_narration_with_fallback(
            inference_device=inference_device,
            segments=segments,
            output_name=output_name,
            silence_ms=req.silence_ms,
            role_profiles=role_profiles,
        )
        merged_path = result["merged_path"]
        wav_paths = build_segment_wav_paths(output_name, len(segments))
        lrc_path = OUTPUT_DIR / f"{output_name}_merged.lrc"
        build_lrc(wav_paths, build_lyric_lines(segments), lrc_path, silence_ms=req.silence_ms)
        return {
            "ok": True,
            "segments": segments,
            "file": merged_path,
            "audio_url": f"/file/{Path(merged_path).name}",
            "lrc_url": f"/file/{lrc_path.name}",
            "device": result.get("device"),
            "fallback_device": result.get("fallback_device"),
            "original_device": result.get("original_device"),
        }
    except Exception as exc:
        raise_api_error(exc)


@app.post("/api/import-segments")
def import_segments(req: ImportSegmentsRequest):
    try:
        parsed = json.loads(req.content)
        if isinstance(parsed, dict):
            segments = parsed.get("segments", [])
        elif isinstance(parsed, list):
            segments = parsed
        else:
            raise ValueError("导入内容必须是 JSON 对象或数组")

        if not isinstance(segments, list):
            raise ValueError("导入内容中的 segments 必须是数组")

        return {
            "ok": True,
            "segments": sanitize_segments(segments),
        }
    except Exception as exc:
        raise_api_error(exc)


@app.post("/api/test-connectivity")
def test_connectivity(req: ConnectivityTestRequest):
    try:
        client = OpenAICompatibleClient(req.llm)
        content = client.chat_text(
            [
                {"role": "system", "content": "Reply in one short sentence."},
                {"role": "user", "content": "Say hello briefly."},
            ]
        )
        return {
            "ok": True,
            "message": content,
        }
    except Exception as exc:
        raise_api_error(exc)


@app.post("/api/upload-ref-audio")
def upload_ref_audio(req: UploadRefAudioRequest):
    try:
        target = save_ref_audio_from_base64(req.filename, req.content_base64)

        return {
            "ok": True,
            "saved_path": str(target.resolve()),
            "audio_url": f"/file/{target.relative_to(Path('outputs')).as_posix()}",
            "file_name": target.name,
        }
    except Exception as exc:
        raise_api_error(exc)


@app.post("/api/batch-import-voice-library")
def batch_import_voice_library(req: BatchImportVoiceLibraryRequest):
    try:
        imported = []
        for item in req.files:
            target = save_ref_audio_from_base64(item.filename, item.content_base64)
            imported.append(
                {
                    "name": Path(item.filename).stem,
                    "style": "",
                    "ref_text": "",
                    "ref_audio": str(target.resolve()),
                    "ref_audio_url": f"/file/{target.relative_to(Path('outputs')).as_posix()}",
                    "file_name": target.name,
                }
            )

        return {
            "ok": True,
            "voices": imported,
        }
    except Exception as exc:
        raise_api_error(exc)


@app.post("/api/model/load-audio")
def load_audio_model(req: ModelLoadRequest):
    try:
        backend = req.backend.model_dump() if req.backend else {}
        mode = str(backend.get("mode") or "local").strip().lower()
        if mode == "remote":
            data = remote_get_json(backend.get("remote_base_url"), "/api/health", backend.get("remote_api_key"))
            return {"ok": True, "mode": "remote", "health": data}

        unload_all_audio_pipelines()
        inference_device = backend.get("inference_device")
        pipeline = get_pipeline(inference_device)
        pipeline.load()
        return {
            "ok": True,
            "mode": "local",
            "device": pipeline.device,
            "model_name": pipeline.model_name,
        }
    except Exception as exc:
        raise_api_error(exc)


@app.post("/api/model/unload-audio")
def unload_audio_model(req: ModelLoadRequest):
    try:
        backend = req.backend.model_dump() if req.backend else {}
        mode = str(backend.get("mode") or "local").strip().lower()
        if mode == "remote":
            return {"ok": True, "mode": "remote", "message": "远程 API 模式由远端自行管理，当前不会在本地卸载。"}

        inference_device = normalize_inference_device(backend.get("inference_device"))
        unload_all_audio_pipelines()
        return {
            "ok": True,
            "mode": "local",
            "device": inference_device,
            "status": get_audio_model_status(),
        }
    except Exception as exc:
        raise_api_error(exc)

@app.get("/api/model/status")
def model_status():
    try:
        return {
            "ok": True,
            "audio": get_audio_model_status(),
        }
    except Exception as exc:
        raise_api_error(exc)


@app.get("/api/system/status")
def system_status():
    try:
        return {
            "ok": True,
            "system": get_system_status(),
            "models": {
                "audio": get_audio_model_status(),
            },
        }
    except Exception as exc:
        raise_api_error(exc)


@app.post("/api/merge")
def merge_audio(req: MergeRequest):
    try:
        output_name = sanitize_output_name(req.output_name, "merged")
        wav_paths = []

        for file_name in req.wav_files:
            wav_path = OUTPUT_DIR / Path(file_name).name
            if not wav_path.exists():
                raise FileNotFoundError(f"找不到分段音频: {wav_path.name}")
            wav_paths.append(wav_path)

        output_path = OUTPUT_DIR / f"{output_name}_merged.wav"
        merged_path = join_wavs(wav_paths, output_path, silence_ms=req.silence_ms)
        lrc_url = None
        if req.lyrics:
            lrc_path = OUTPUT_DIR / f"{output_name}_merged.lrc"
            lyric_lines = []
            for line in req.lyrics:
                speaker = (line.speaker or "").strip()
                text = line.text.strip()
                lyric_lines.append(f"{speaker}：{text}" if speaker and speaker != "旁白" else text)
            build_lrc(wav_paths, lyric_lines, lrc_path, silence_ms=req.silence_ms)
            lrc_url = f"/file/{lrc_path.name}"
        return {
            "ok": True,
            "segments": len(wav_paths),
            "file": merged_path,
            "audio_url": f"/file/{Path(merged_path).name}",
            "lrc_url": lrc_url,
        }
    except Exception as exc:
        raise_api_error(exc)


@app.get("/api/health")
def health():
    loaded_devices = sorted(device for device, pipeline in pipeline_cache.items() if pipeline.model is not None)
    return {
        "ok": True,
        "app": app.title,
        "model_loaded": bool(loaded_devices),
        "loaded_devices": loaded_devices,
        "default_device": OmniVoicePipeline.detect_device(),
        "outputs_dir": str(OUTPUT_DIR.resolve()),
    }


@app.get("/file/{name:path}")
def get_file(name: str):
    path = Path("outputs") / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=path.name)
