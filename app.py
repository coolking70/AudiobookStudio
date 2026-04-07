import base64
import json
import mimetypes
import re
import traceback
from pathlib import Path
from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from audio_utils import build_lrc, join_wavs
from llm_client import OpenAICompatibleClient
from pipeline import OmniVoicePipeline
from role_analyzer import analyze_chunks_with_llm, analyze_text_with_llm, intelligent_segment_text, optimize_chunks_for_tts, sanitize_segments
from schemas import AnalyzeChunksRequest, AutoNarrateRequest, ChunkOptimizeRequest, ConnectivityTestRequest, ImportSegmentsRequest, MergeRequest, NarrateRequest, ParseRequest, SegmentPlanRequest, TTSRequest, UploadRefAudioRequest


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
        output_path = OUTPUT_DIR / f"{output_name}.wav"
        pipeline = get_pipeline(req.inference_device)

        pipeline.synthesize_segment(
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
        pipeline = get_pipeline(req.inference_device)

        merged_path = pipeline.synthesize_segments(
            segments=segments,
            base_name=output_name,
            silence_ms=req.silence_ms,
            role_profiles=role_profiles,
        )
        wav_paths = build_segment_wav_paths(output_name, len(segments))
        lrc_path = OUTPUT_DIR / f"{output_name}_merged.lrc"
        build_lrc(wav_paths, build_lyric_lines(segments), lrc_path, silence_ms=req.silence_ms)
        return {
            "ok": True,
            "segments": len(segments),
            "file": merged_path,
            "audio_url": f"/file/{Path(merged_path).name}",
            "lrc_url": f"/file/{lrc_path.name}",
        }
    except Exception as exc:
        raise_api_error(exc)


@app.post("/api/auto-narrate")
def auto_narrate(req: AutoNarrateRequest):
    try:
        output_name = sanitize_output_name(req.output_name, "auto_narration")
        segments = analyze_text_with_llm(req.text, req.llm)
        role_profiles = {k: v.model_dump() for k, v in (req.role_profiles or {}).items()}
        pipeline = get_pipeline(req.inference_device)

        merged_path = pipeline.synthesize_segments(
            segments=segments,
            base_name=output_name,
            silence_ms=req.silence_ms,
            role_profiles=role_profiles,
        )
        wav_paths = build_segment_wav_paths(output_name, len(segments))
        lrc_path = OUTPUT_DIR / f"{output_name}_merged.lrc"
        build_lrc(wav_paths, build_lyric_lines(segments), lrc_path, silence_ms=req.silence_ms)
        return {
            "ok": True,
            "segments": segments,
            "file": merged_path,
            "audio_url": f"/file/{Path(merged_path).name}",
            "lrc_url": f"/file/{lrc_path.name}",
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
        REF_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = build_unique_upload_filename(req.filename)
        target = REF_AUDIO_DIR / safe_name

        raw = req.content_base64
        if "," in raw:
            raw = raw.split(",", 1)[1]

        content = base64.b64decode(raw)
        target.write_bytes(content)

        return {
            "ok": True,
            "saved_path": str(target.resolve()),
            "audio_url": f"/file/{target.relative_to(Path('outputs')).as_posix()}",
            "file_name": target.name,
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
