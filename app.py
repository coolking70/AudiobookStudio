import base64
import json
import mimetypes
import re
import traceback
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from audio_utils import build_lrc, join_wavs
from llm_client import OpenAICompatibleClient
from pipeline import OmniVoicePipeline
from role_analyzer import analyze_text_with_llm, sanitize_segments
from schemas import AutoNarrateRequest, ConnectivityTestRequest, ImportSegmentsRequest, MergeRequest, NarrateRequest, ParseRequest, TTSRequest, UploadRefAudioRequest


app = FastAPI(title="OmniVoice Reader Studio")
pipeline = OmniVoicePipeline()

Path("outputs").mkdir(parents=True, exist_ok=True)
Path("static").mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

OUTPUT_DIR = Path("outputs")
REF_AUDIO_DIR = OUTPUT_DIR / "ref_audio"


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


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = Path("static/index.html")
    if not html_path.exists():
        return "<h1>index.html 不存在</h1>"
    return html_path.read_text(encoding="utf-8")


@app.post("/api/tts")
def tts(req: TTSRequest):
    try:
        output_name = sanitize_output_name(req.output_name, "tts_result")
        output_path = OUTPUT_DIR / f"{output_name}.wav"

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


@app.post("/api/narrate")
def narrate(req: NarrateRequest):
    try:
        output_name = sanitize_output_name(req.output_name, "narration")
        segments = [seg.model_dump() for seg in req.segments]
        role_profiles = {k: v.model_dump() for k, v in (req.role_profiles or {}).items()}

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
    return {
        "ok": True,
        "app": app.title,
        "model_loaded": pipeline.model is not None,
        "outputs_dir": str(OUTPUT_DIR.resolve()),
    }


@app.get("/file/{name:path}")
def get_file(name: str):
    path = Path("outputs") / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=path.name)
