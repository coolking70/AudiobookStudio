import base64
import asyncio
import json
import mimetypes
import os
import platform
import re
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from threading import Lock
from urllib.parse import urljoin
from uuid import uuid4

import httpx
try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None
try:
    import pythoncom
    import win32com.client
except ImportError:
    pythoncom = None
    win32com = None
try:
    from winsdk.windows.media.speechsynthesis import SpeechSynthesizer as WinSpeechSynthesizer
    from winsdk.windows.storage.streams import DataReader, InputStreamOptions
except ImportError:
    WinSpeechSynthesizer = None
    DataReader = None
    InputStreamOptions = None
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from audio_utils import build_lrc, join_wavs_auto
from llm_client import OpenAICompatibleClient
from local_llm import get_local_llm_runner, get_local_llm_status, unload_all_local_llm_runners
from output_layout import (
    build_segment_output_paths,
    build_temp_archive_path,
    ensure_output_dir,
    get_temp_archive_dir,
    resolve_named_output_path,
)
from pipeline import ASR_MODEL_NAME, HF_CACHE_DIRS, MODEL_NAME, VOXCPM_MODEL_NAME, OmniVoicePipeline, VoxCPMPipeline, get_asr_runtime_dependency_status, get_runtime_dependency_status, get_voxcpm_runtime_dependency_status
from role_analyzer import (
    CHARACTER_ALIAS_RESOLUTION_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    SEGMENT_PLAN_PROMPT,
    SPEAKER_VERIFICATION_PROMPT,
    TEXT_OPTIMIZATION_PROMPT,
    analyze_chunks_with_llm,
    analyze_text_with_llm,
    detect_chapter_structure_with_info,
    intelligent_segment_seed_chunk,
    intelligent_segment_text,
    intelligent_segment_text_with_info,
    optimize_and_analyze_chunks_with_llm,
    optimize_chunks_for_tts,
    preprocess_text_for_chapter_detection,
    resolve_character_aliases,
    sanitize_segments,
    segment_and_optimize_seed_chunk,
    segment_and_optimize_text_with_info,
    verify_speakers_pass,
)
from schemas import (
    AnalyzeChunksRequest,
    AutoNarrateRequest,
    BatchImportVoiceLibraryRequest,
    CheckFilesRequest,
    ChunkOptimizeRequest,
    ConnectivityTestRequest,
    ImportSegmentsRequest,
    ListLLMModelsRequest,
    MergeAliasesRequest,
    MergeRequest,
    ModelLoadRequest,
    NarrateRequest,
    ParseRequest,
    ParseV2ReviewOneRequest,
    ParseV2Request,
    ParseV2ReviewRequest,
    ScanLocalModelsRequest,
    SegmentPlanRequest,
    TTSRequest,
    TranscribeRefAudioRequest,
    UploadRefAudioRequest,
    VerifySpeakersRequest,
)


app = FastAPI(title="OmniVoice Reader Studio")
pipeline_cache: dict[tuple[str, str, str], OmniVoicePipeline] = {}
voxcpm_pipeline_cache: dict[tuple[str, str], VoxCPMPipeline] = {}


@app.get("/.well-known/appspecific/com.chrome.devtools.json")
def chrome_devtools_probe():
    return Response(status_code=204)


def dump_request_debug(name: str, payload: dict[str, object]) -> None:
    try:
        out_dir = get_temp_archive_dir("llm_debug")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        target = out_dir / f"{stamp}_{name}.json"
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
pipeline_lock = Lock()

ensure_output_dir()
Path("static").mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

OUTPUT_DIR = ensure_output_dir()
REF_AUDIO_DIR = OUTPUT_DIR / "ref_audio"
AI_HELPER_CONFIG_CANDIDATES = [
    Path(os.getenv("AUDIOBOOKSTUDIO_LLM_DEFAULTS", "")).expanduser() if os.getenv("AUDIOBOOKSTUDIO_LLM_DEFAULTS") else None,
    Path(__file__).resolve().parent / "ai-api.md",
    Path(__file__).resolve().parent / "config" / "ai-api.md",
    Path(__file__).resolve().parent.parent / "tts_text_parse" / "ai-helper" / "ai-api.md",
]

MICROSOFT_TTS_SUPPORTED_CULTURES = {"zh-CN", "en-US", "ja-JP"}
MIMO_TTS_DEFAULT_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
MIMO_TTS_MODEL = "mimo-v2.5-tts"
MIMO_TTS_VOICE_DESIGN_MODEL = "mimo-v2.5-tts-voicedesign"
MIMO_TTS_VOICE_CLONE_MODEL = "mimo-v2.5-tts-voiceclone"
MIMO_TTS_BUILTIN_VOICES = {"mimo_default", "冰糖", "茉莉", "苏打", "白桦", "Mia", "Chloe", "Milo", "Dean"}
VOXCPM_BRIDGE_SCRIPT = Path(__file__).resolve().parent / "voxcpm_bridge.py"


def augment_process_path_from_python_env() -> None:
    python_path = Path(sys.executable).resolve()
    env_root = python_path.parent.parent if python_path.parent.name.lower() in {"scripts", "bin"} else python_path.parent
    candidates = [
        env_root,
        env_root / "Scripts",
        env_root / "bin",
        env_root / "Library" / "bin",
        env_root / "DLLs",
    ]
    current_path = os.environ.get("PATH", "")
    parts = [part for part in current_path.split(os.pathsep) if part]
    normalized_parts = {part.lower() for part in parts}
    prepend = []
    for candidate in candidates:
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str.lower() not in normalized_parts:
                prepend.append(candidate_str)
    if prepend:
        os.environ["PATH"] = os.pathsep.join(prepend + parts)

    try:
        import imageio_ffmpeg

        ffmpeg_exe = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
        ffmpeg_dir = str(ffmpeg_exe.parent)
        current_parts = os.environ.get("PATH", "").split(os.pathsep)
        if ffmpeg_dir and ffmpeg_dir.lower() not in {part.lower() for part in current_parts if part}:
            os.environ["PATH"] = os.pathsep.join([ffmpeg_dir] + [part for part in current_parts if part])
    except Exception:
        pass


augment_process_path_from_python_env()


def _normalize_voice_name(name: str | None) -> str:
    normalized = str(name or "").strip().lower()
    normalized = normalized.replace(" desktop", "")
    return re.sub(r"\s+", " ", normalized)


def _matches_sapi_voice_description(description: str, requested: str) -> bool:
    actual = str(description or "").strip()
    target = str(requested or "").strip()
    if not actual or not target:
        return False
    return actual == target or actual.startswith(f"{target} -")


def build_microsoft_voice_style(gender: str | None, locale: str | None) -> str:
    tags = []
    gender_normalized = str(gender or "").strip().lower()
    locale_normalized = str(locale or "").strip()
    if gender_normalized == "female":
        tags.append("female")
    elif gender_normalized == "male":
        tags.append("male")
    if locale_normalized == "zh-CN":
        tags.append("chinese accent")
    elif locale_normalized == "ja-JP":
        tags.append("japanese accent")
    elif locale_normalized == "en-US":
        tags.append("english accent")
    tags.append("moderate pitch")
    deduped: list[str] = []
    for item in tags:
        if item not in deduped:
            deduped.append(item)
    return ", ".join(deduped)


def _run_powershell_json(command: str) -> object:
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    stdout = (completed.stdout or "").strip()
    if not stdout:
        return []
    return json.loads(stdout)


def get_voxcpm_bridge_python_candidates() -> list[Path]:
    candidates: list[Path] = []
    explicit = str(os.getenv("VOXCPM_PYTHON_EXE") or "").strip()
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend(
        [
            Path(r"I:\conda_envs\voxcpm\python.exe"),
            Path(r"I:\ProgramData\miniconda3\envs\voxcpm\python.exe"),
            Path(sys.executable).resolve(),
        ]
    )
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate).lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(candidate)
    return deduped


def find_voxcpm_bridge_python() -> Path | None:
    for candidate in get_voxcpm_bridge_python_candidates():
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def get_voxcpm_bridge_status() -> dict[str, object]:
    python_exe = find_voxcpm_bridge_python()
    return {
        "available": bool(python_exe and VOXCPM_BRIDGE_SCRIPT.exists()),
        "python_executable": str(python_exe) if python_exe else None,
        "script_path": str(VOXCPM_BRIDGE_SCRIPT),
    }


def run_voxcpm_bridge(action: str, payload: dict[str, object] | None = None) -> dict[str, object]:
    status = get_voxcpm_bridge_status()
    python_exe = status.get("python_executable")
    if not status.get("available") or not python_exe:
        raise RuntimeError(
            "当前主服务环境未安装 VoxCPM，且未找到可用的 VoxCPM 专用 Python 环境。"
            " 请确认 I:\\conda_envs\\voxcpm\\python.exe 存在，或设置环境变量 VOXCPM_PYTHON_EXE。"
        )

    completed = subprocess.run(
        [str(python_exe), str(VOXCPM_BRIDGE_SCRIPT), action],
        input=json.dumps(payload or {}, ensure_ascii=False),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(Path(__file__).resolve().parent),
        check=False,
    )
    stdout = str(completed.stdout or "").strip()
    stderr = str(completed.stderr or "").strip()
    if not stdout:
        raise RuntimeError(f"VoxCPM 专用环境未返回结果。{stderr or '请检查专用环境日志。'}")

    try:
        data = json.loads(stdout)
    except Exception as exc:
        raise RuntimeError(f"VoxCPM 专用环境返回了无法解析的内容：{stdout[:400]}") from exc

    if completed.returncode != 0 or not data.get("ok"):
        error = data.get("error") or stderr or "未知错误"
        raise RuntimeError(f"VoxCPM 专用环境执行失败：{error}")
    return data


def get_microsoft_tts_catalog() -> dict[str, object]:
    if platform.system() != "Windows":
        return {
            "desktop_voices": [],
            "browser_like_voices": [],
            "importable_voices": [],
            "available_languages": [],
        }

    desktop_command = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.GetInstalledVoices() | ForEach-Object { "
        "$v = $_.VoiceInfo; "
        "[PSCustomObject]@{ "
        "name = $v.Name; "
        "display_name = $v.Name; "
        "locale = $v.Culture.Name; "
        "gender = $v.Gender.ToString(); "
        "description = $v.Description "
        "} "
        "} | ConvertTo-Json -Depth 4"
    )
    onecore_command = (
        "Get-ChildItem 'HKLM:\\SOFTWARE\\Microsoft\\Speech_OneCore\\Voices\\Tokens' -ErrorAction SilentlyContinue | "
        "ForEach-Object { "
        "$p = Get-ItemProperty $_.PSPath; "
        "$gender = if ($_.PSChildName -match '([A-Za-z]+)M$') { "
        "switch ($Matches[1]) { 'David' {'Male'} 'Mark' {'Male'} 'Kangkang' {'Male'} 'Ichiro' {'Male'} default {'Female'} } "
        "} else { '' }; "
        "[PSCustomObject]@{ "
        "name = $p.'(default)'; "
        "display_name = ($p.'(default)' -replace ' - .+$',''); "
        "locale = if ($_.PSChildName -match '_([a-z]{2}[A-Z]{2})_') { ($Matches[1] -replace '([a-z]{2})([A-Z]{2})','$1-$2') } else { '' }; "
        "gender = $gender; "
        "token = $_.PSChildName "
        "} "
        "} | ConvertTo-Json -Depth 4"
    )

    desktop_raw = _run_powershell_json(desktop_command)
    onecore_raw = _run_powershell_json(onecore_command)
    desktop_voices = desktop_raw if isinstance(desktop_raw, list) else ([desktop_raw] if desktop_raw else [])
    browser_like_voices = onecore_raw if isinstance(onecore_raw, list) else ([onecore_raw] if onecore_raw else [])

    browser_like_map = {
        _normalize_voice_name(item.get("display_name") or item.get("name")): item
        for item in browser_like_voices
        if item.get("display_name") or item.get("name")
    }

    importable_source = browser_like_voices if WinSpeechSynthesizer is not None else desktop_voices
    importable_voices = []
    seen_importable: set[tuple[str, str]] = set()
    for item in importable_source:
        locale = str(item.get("locale") or "").strip()
        if locale not in MICROSOFT_TTS_SUPPORTED_CULTURES:
            continue
        display_name = item.get("display_name") or item.get("name")
        normalized_name = _normalize_voice_name(display_name)
        dedupe_key = (normalized_name, locale.lower())
        if dedupe_key in seen_importable:
            continue
        seen_importable.add(dedupe_key)
        browser_match = browser_like_map.get(normalized_name)
        importable_voices.append(
            {
                "name": display_name,
                "system_voice_name": display_name,
                "locale": locale,
                "gender": item.get("gender"),
                "style": build_microsoft_voice_style(item.get("gender"), locale),
                "preview_available": bool(browser_match) or WinSpeechSynthesizer is not None,
                "generation_available": True,
                "browser_display_name": browser_match.get("display_name") if browser_match else None,
            }
        )

    languages = sorted(
        {
            str(item.get("locale") or "").strip()
            for item in [*desktop_voices, *browser_like_voices]
            if str(item.get("locale") or "").strip()
        }
    )
    return {
        "desktop_voices": desktop_voices,
        "browser_like_voices": browser_like_voices,
        "importable_voices": importable_voices,
        "available_languages": languages,
    }


def _infer_microsoft_tts_rate(style: str | None) -> int:
    normalized = str(style or "").lower()
    rate = 0
    if "child" in normalized or "teenager" in normalized or "young adult" in normalized:
        rate += 1
    if "elderly" in normalized:
        rate -= 1
    return max(-3, min(3, rate))


def synthesize_microsoft_tts(text: str, output_path: Path, voice_name: str, style: str | None = None) -> str:
    if platform.system() != "Windows":
        raise RuntimeError("微软系统 TTS 目前仅支持 Windows。")
    safe_voice_name = str(voice_name or "").strip()
    if not safe_voice_name:
        raise RuntimeError("微软 TTS 缺少 voice_name。")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = build_temp_archive_path(f"ms_tmp_{uuid4().hex[:10]}.wav", category="microsoft_tts")
    if WinSpeechSynthesizer is not None and DataReader is not None and InputStreamOptions is not None:
        async def _synthesize_with_winrt() -> None:
            synth = WinSpeechSynthesizer()
            selected_voice = None
            normalized_target = _normalize_voice_name(safe_voice_name)
            for voice in WinSpeechSynthesizer.all_voices:
                display_name = str(getattr(voice, "display_name", "") or "").strip()
                if _normalize_voice_name(display_name) == normalized_target:
                    selected_voice = voice
                    break
            if selected_voice is None:
                raise RuntimeError(f"找不到微软 WinRT 语音：{safe_voice_name}")
            synth.voice = selected_voice
            stream = await synth.synthesize_text_to_stream_async(text)
            input_stream = stream.get_input_stream_at(0)
            reader = DataReader(input_stream)
            reader.input_stream_options = InputStreamOptions.READ_AHEAD
            await reader.load_async(stream.size)
            buffer = reader.read_buffer(stream.size)
            temp_output_path.write_bytes(bytes(buffer))

        asyncio.run(_synthesize_with_winrt())
    elif pythoncom is not None and win32com is not None:
        rate = _infer_microsoft_tts_rate(style)
        pythoncom.CoInitialize()
        stream = None
        try:
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            stream = win32com.client.Dispatch("SAPI.SpFileStream")
            selected = False
            for token in speaker.GetVoices():
                description = str(token.GetDescription() or "").strip()
                if _matches_sapi_voice_description(description, safe_voice_name):
                    speaker.Voice = token
                    selected = True
                    break
            if not selected:
                raise RuntimeError(f"找不到微软系统语音：{safe_voice_name}")
            speaker.Rate = rate
            if temp_output_path.exists():
                temp_output_path.unlink()
            stream.Open(str(temp_output_path), 3)
            speaker.AudioOutputStream = stream
            speaker.Speak(text)
        finally:
            try:
                if stream is not None:
                    stream.Close()
            except Exception:
                pass
            pythoncom.CoUninitialize()
    else:
        raise RuntimeError("当前环境缺少 winsdk 或 pywin32，无法调用微软系统 TTS。")
    if not temp_output_path.exists():
        raise RuntimeError("微软 TTS 未生成输出音频文件。")
    if temp_output_path.stat().st_size <= 46:
        raise RuntimeError("微软 TTS 生成了空音频文件，请检查语音名称和文本内容。")
    if output_path.exists():
        output_path.unlink()
    temp_output_path.replace(output_path)
    return str(output_path)


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
    return build_segment_output_paths(base_name, segment_count)


def build_output_file_url(path: str | Path | None) -> str | None:
    if not path:
        return None
    target = Path(path)
    try:
        relative = target.resolve().relative_to(OUTPUT_DIR.resolve())
    except Exception:
        relative = target.relative_to(OUTPUT_DIR)
    return f"/file/{relative.as_posix()}"


def build_merge_response_payload(
    *,
    merge_result: dict,
    wav_paths: list[Path],
    lyric_lines: list[str] | None,
    output_name: str,
    silence_ms: int,
) -> dict:
    parts_payload = []
    split_applied = bool(merge_result.get("split_applied"))

    for part in merge_result.get("parts", []):
        start_index = int(part["start_index"])
        end_index = int(part["end_index"])
        lrc_url = None
        if lyric_lines:
            part_lrc_path = Path(part["path"]).with_suffix(".lrc")
            build_lrc(
                wav_paths[start_index - 1 : end_index],
                lyric_lines[start_index - 1 : end_index],
                part_lrc_path,
                silence_ms=silence_ms,
            )
            lrc_url = build_output_file_url(part_lrc_path)
        parts_payload.append(
            {
                "index": int(part["index"]),
                "file": part["path"],
                "audio_url": build_output_file_url(part["path"]),
                "lrc_url": lrc_url,
                "segment_count": int(part["segment_count"]),
                "start_index": start_index,
                "end_index": end_index,
                "duration_seconds": float(part["duration_seconds"]),
                "size_bytes": int(part["size_bytes"]),
            }
        )

    primary_part = parts_payload[0] if parts_payload else None
    payload = {
        "ok": True,
        "segments": len(wav_paths),
        "file": merge_result["primary_path"],
        "audio_url": primary_part["audio_url"] if primary_part else None,
        "lrc_url": primary_part["lrc_url"] if primary_part else None,
        "parts": parts_payload,
        "split_applied": split_applied,
        "manifest_file": merge_result.get("manifest_path"),
        "manifest_url": build_output_file_url(merge_result.get("manifest_path")),
    }
    if split_applied:
        payload["message"] = f"音频体积较大，已自动拆分为 {len(parts_payload)} 个分卷。播放器当前加载第 1 卷。"
    return payload


def raise_api_error(exc: Exception) -> None:
    traceback.print_exc()
    raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


def find_ai_helper_config_path() -> Path | None:
    for candidate in AI_HELPER_CONFIG_CANDIDATES:
        if candidate and candidate.exists():
            return candidate
    return None


def load_ai_helper_llm_defaults() -> dict[str, object]:
    config_path = find_ai_helper_config_path()
    if config_path is None:
        return {
            "base_url": "",
            "api_key": "",
            "model": "",
            "temperature": 0.3,
            "max_tokens": 8000,
            "analysis_mode": "balanced",
            "compatibility_mode": "chat_compat",
            "system_prompt": None,
            "source": None,
            "available_models": [],
            "configured": False,
        }

    text = config_path.read_text(encoding="utf-8")
    api_key_match = re.search(r"API Key[：:]\s*(\S+)", text)
    base_url_match = re.search(r"Base URL[：:]\s*(\S+)", text)
    models_match = re.search(r"可用model[：:]\s*(.+)", text)

    if not api_key_match or not base_url_match:
        raise RuntimeError(f"无法从 {config_path} 解析 API Key 或 Base URL")

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
        "source": str(config_path),
        "available_models": available_models,
        "configured": True,
    }


def normalize_remote_base_url(base_url: str | None) -> str:
    normalized = str(base_url or "").strip().rstrip("/")
    if not normalized:
        raise ValueError("远程 API Base URL 不能为空")
    return normalized


def normalize_openai_compat_base_url(base_url: str | None) -> str:
    normalized = str(base_url or "").strip().rstrip("/")
    if not normalized:
        raise ValueError("Base URL 不能为空")
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def normalize_mimo_base_url(base_url: str | None) -> str:
    normalized = str(base_url or MIMO_TTS_DEFAULT_BASE_URL).strip().rstrip("/")
    if not normalized:
        return MIMO_TTS_DEFAULT_BASE_URL
    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


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


def _read_audio_as_data_url(ref_audio: str) -> str:
    source = str(ref_audio or "").strip()
    if not source:
        raise RuntimeError("MiMo VoiceClone 缺少参考音频。")
    if source.startswith("data:"):
        if ";base64," not in source:
            raise RuntimeError("MiMo VoiceClone 参考音频 data URL 必须包含 base64 数据。")
        return source
    if source.startswith("/file/"):
        source = str((OUTPUT_DIR / source.removeprefix("/file/")).resolve())
    if re.match(r"^https?://", source):
        timeout = httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0)
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            resp = client.get(source)
            resp.raise_for_status()
            content = resp.content
            content_type = resp.headers.get("content-type") or "audio/wav"
    else:
        path = Path(source).expanduser()
        if not path.exists():
            raise RuntimeError(f"MiMo VoiceClone 参考音频不存在：{source}")
        content = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "audio/wav"
    normalized_type = content_type.split(";", 1)[0].strip().lower()
    if normalized_type in {"audio/mp3", "audio/mpeg"}:
        normalized_type = "audio/mpeg"
    elif normalized_type in {"audio/wav", "audio/x-wav", "audio/wave"}:
        normalized_type = "audio/wav"
    else:
        raise RuntimeError("MiMo VoiceClone 当前仅支持 mp3 或 wav 参考音频。")
    encoded = base64.b64encode(content).decode("ascii")
    if len(encoded) > 10 * 1024 * 1024:
        raise RuntimeError("MiMo VoiceClone 参考音频 base64 超过 10MB，请换用更短的 mp3/wav。")
    return f"data:{normalized_type};base64,{encoded}"


def resolve_mimo_tts_model(backend: dict | None, voice_engine: str, voice_name: str | None, ref_audio: str | None) -> str:
    configured = str((backend or {}).get("mimo_model") or "auto").strip()
    if configured and configured.lower() != "auto":
        return configured
    if str(voice_engine or "").strip().lower() == "mimo" and ref_audio:
        return MIMO_TTS_VOICE_CLONE_MODEL
    if str(voice_engine or "").strip().lower() == "mimo" and not voice_name:
        return MIMO_TTS_VOICE_DESIGN_MODEL
    return MIMO_TTS_MODEL


def validate_mimo_tts_backend(backend: dict | None) -> dict:
    backend = backend or {}
    api_key = str(backend.get("mimo_api_key") or backend.get("remote_api_key") or "").strip()
    if not api_key:
        raise RuntimeError("MiMo TTS 缺少 API Key。请在模型配置 -> 语音生成模型中填写。")
    model = str(backend.get("mimo_model") or "auto").strip() or "auto"
    if model != "auto" and model not in {MIMO_TTS_MODEL, MIMO_TTS_VOICE_DESIGN_MODEL, MIMO_TTS_VOICE_CLONE_MODEL}:
        raise RuntimeError(f"不支持的 MiMo TTS 模型：{model}")
    voice = str(backend.get("mimo_voice") or "mimo_default").strip() or "mimo_default"
    if voice not in MIMO_TTS_BUILTIN_VOICES:
        raise RuntimeError(f"不支持的 MiMo 内置音色：{voice}")
    return {
        "base_url": normalize_mimo_base_url(backend.get("mimo_base_url") or backend.get("remote_base_url")),
        "model": model,
        "voice": voice,
    }


def synthesize_mimo_tts(
    *,
    backend: dict | None,
    text: str,
    output_path: Path,
    instruct: str | None = None,
    ref_audio: str | None = None,
    voice_name: str | None = None,
    voice_engine: str | None = None,
) -> dict:
    backend = backend or {}
    api_key = str(backend.get("mimo_api_key") or backend.get("remote_api_key") or "").strip()
    if not api_key:
        raise RuntimeError("MiMo TTS 缺少 API Key。请在模型配置 -> 语音生成模型中填写。")
    base_url = normalize_mimo_base_url(backend.get("mimo_base_url") or backend.get("remote_base_url"))
    model = resolve_mimo_tts_model(backend, voice_engine or "", voice_name, ref_audio)
    assistant_text = str(text or "").strip()
    if not assistant_text:
        raise RuntimeError("MiMo TTS 缺少待合成文本。")

    style_text = str(instruct or "").strip()
    messages = []
    if model == MIMO_TTS_VOICE_DESIGN_MODEL:
        messages.append({
            "role": "user",
            "content": style_text or "A natural, clear Mandarin audiobook voice with stable pacing and expressive but restrained emotion.",
        })
    else:
        messages.append({"role": "user", "content": style_text})
    messages.append({"role": "assistant", "content": assistant_text})

    audio: dict[str, str] = {"format": "wav"}
    if model == MIMO_TTS_VOICE_CLONE_MODEL:
        audio["voice"] = _read_audio_as_data_url(str(ref_audio or ""))
    elif model == MIMO_TTS_MODEL:
        audio["voice"] = str(voice_name or backend.get("mimo_voice") or "mimo_default").strip() or "mimo_default"

    payload = {
        "model": model,
        "messages": messages,
        "audio": audio,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "api-key": api_key,
    }
    timeout = httpx.Timeout(connect=30.0, read=300.0, write=60.0, pool=30.0)
    with httpx.Client(timeout=timeout, trust_env=False) as client:
        resp = client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
        try:
            data = resp.json()
        except Exception as exc:
            raise RuntimeError(f"MiMo TTS 返回非 JSON 响应：HTTP {resp.status_code} {resp.text[:300]}") from exc
        if not resp.is_success:
            raise RuntimeError(data.get("error") or data.get("detail") or data)

    try:
        message = data["choices"][0]["message"]
        audio_data = message["audio"]["data"]
    except Exception as exc:
        raise RuntimeError(f"MiMo TTS 响应中缺少 audio.data：{json.dumps(data, ensure_ascii=False)[:500]}") from exc
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(base64.b64decode(audio_data))
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("MiMo TTS 未生成有效音频文件。")
    return {
        "engine": "mimo",
        "model": model,
        "voice": audio.get("voice") if model == MIMO_TTS_MODEL else "",
        "base_url": base_url,
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


def get_pipeline(
    device: str | None = None,
    model_name: str | None = None,
    asr_model_name: str | None = None,
) -> OmniVoicePipeline:
    resolved_device = normalize_inference_device(device)
    resolved_model_name = OmniVoicePipeline.resolve_model_name_or_path(model_name or MODEL_NAME)
    resolved_asr_model_name = OmniVoicePipeline.resolve_asr_model_name_or_path(asr_model_name or ASR_MODEL_NAME)
    cache_key = (resolved_device, resolved_model_name, resolved_asr_model_name)
    with pipeline_lock:
        pipeline = pipeline_cache.get(cache_key)
        if pipeline is None:
            pipeline = OmniVoicePipeline(
                model_name=resolved_model_name,
                asr_model_name=resolved_asr_model_name,
                device=resolved_device,
            )
            pipeline_cache[cache_key] = pipeline
        return pipeline


def unload_all_audio_pipelines() -> None:
    with pipeline_lock:
        pipelines = [*pipeline_cache.values(), *voxcpm_pipeline_cache.values()]
    for pipeline in pipelines:
        pipeline.unload()


def get_voxcpm_pipeline(
    device: str | None = None,
    model_name: str | None = None,
) -> VoxCPMPipeline:
    resolved_device = normalize_inference_device(device)
    resolved_model_name = VoxCPMPipeline.resolve_model_name_or_path(model_name or VOXCPM_MODEL_NAME)
    cache_key = (resolved_device, resolved_model_name)
    with pipeline_lock:
        pipeline = voxcpm_pipeline_cache.get(cache_key)
        if pipeline is None:
            pipeline = VoxCPMPipeline(
                model_name=resolved_model_name,
                device=resolved_device,
            )
            voxcpm_pipeline_cache[cache_key] = pipeline
        return pipeline


def resolve_audio_backend(req_backend, fallback_device: str | None = None) -> tuple[str, str | None, str | None, str | None]:
    backend = req_backend or {}
    mode = str(backend.get("mode") or "local").strip().lower()
    remote_base_url = backend.get("remote_base_url")
    remote_api_key = backend.get("remote_api_key")
    inference_device = backend.get("inference_device") or fallback_device
    return mode, remote_base_url, remote_api_key, inference_device


def resolve_local_audio_model_path(req_backend) -> str | None:
    backend = req_backend or {}
    value = str(backend.get("local_audio_model_path") or "").strip()
    return value or None


def resolve_local_audio_engine(req_backend) -> str:
    backend = req_backend or {}
    value = str(backend.get("local_audio_engine") or "").strip().lower()
    if value in {"", "omnivoice"}:
        return "omnivoice"
    if value == "voxcpm":
        return "voxcpm"
    raise ValueError(f"不支持的本地语音引擎: {backend.get('local_audio_engine')}")


def resolve_local_asr_model_path(req_backend) -> str | None:
    backend = req_backend or {}
    value = str(backend.get("local_asr_model_path") or "").strip()
    return value or None


def build_local_path_preset_catalog() -> dict[str, dict[str, object]]:
    home = Path.home()
    workspace_root = Path(__file__).resolve().parent.parent
    workspace_models = workspace_root / "tts_text_parse" / "models"
    hf_roots = list(HF_CACHE_DIRS or [])

    def first_existing(paths: list[Path]) -> Path | None:
        for path in paths:
            if path.exists():
                return path
        return None

    def build_entry(label: str, paths: list[Path], note: str) -> dict[str, object]:
        selected = first_existing(paths) or (paths[0] if paths else None)
        resolved = str(selected.expanduser()) if selected else ""
        return {
            "label": label,
            "path": resolved,
            "exists": bool(selected and selected.exists()),
            "is_dir": bool(selected and selected.is_dir()),
            "note": note,
            "candidates": [str(path.expanduser()) for path in paths],
        }

    omnivoice_candidates = [root / "models--k2-fsa--OmniVoice" for root in hf_roots]
    whisper_candidates = [root / "models--openai--whisper-large-v3-turbo" for root in hf_roots]
    voxcpm_candidates = [root / "models--openbmb--VoxCPM2" for root in hf_roots]

    return {
        "llm_root_lmstudio": build_entry(
            "LM Studio 模型目录",
            [
                home / ".lmstudio" / "models",
                home / ".cache" / "lm-studio" / "models",
            ],
            "优先查找当前用户目录下的 LM Studio 模型缓存。",
        ),
        "llm_root_workspace_models": build_entry(
            "工作区模型目录",
            [workspace_models],
            "优先查找当前工作区内的本地模型目录。",
        ),
        "audio_omnivoice_cache": build_entry(
            "OmniVoice 缓存目录",
            omnivoice_candidates or [home / ".cache" / "huggingface" / "hub" / "models--k2-fsa--OmniVoice"],
            "会按当前环境的 Hugging Face 缓存根目录自动检测 OmniVoice 模型仓库。",
        ),
        "audio_voxcpm_cache": build_entry(
            "VoxCPM 缓存目录",
            voxcpm_candidates or [home / ".cache" / "huggingface" / "hub" / "models--openbmb--VoxCPM2"],
            "会按当前环境的 Hugging Face 缓存根目录自动检测 VoxCPM2 模型仓库。",
        ),
        "audio_whisper_cache": build_entry(
            "Whisper 缓存目录",
            whisper_candidates or [home / ".cache" / "huggingface" / "hub" / "models--openai--whisper-large-v3-turbo"],
            "会按当前环境的 Hugging Face 缓存根目录自动检测 Whisper 模型仓库。",
        ),
    }


def detect_local_llm_engine(path: Path) -> tuple[str, bool, str | None]:
    expanded = path.expanduser()
    if expanded.is_file() and expanded.suffix.lower() == ".gguf":
        return "gguf", True, None

    if expanded.is_dir():
        config_path = expanded / "config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:
                config = {}
            quantization = config.get("quantization") or config.get("quantization_config") or {}
            if isinstance(quantization, dict) and quantization.get("bits") and not quantization.get("quant_method"):
                return "mlx", True, None
            return "transformers", True, None

        gguf_files = [item for item in sorted(expanded.glob("*.gguf")) if "mmproj" not in item.name.lower()]
        if gguf_files:
            if len(gguf_files) == 1:
                return "gguf", True, None
            return "gguf", False, "目录中存在多个 GGUF 文件，请直接选择其中一个主模型文件。"

    return "unknown", False, "未识别到可直接加载的模型目录。"


def scan_local_llm_models(root_path: str) -> list[dict[str, object]]:
    root = Path(str(root_path or "")).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"目录不存在：{root}")
    if not root.is_dir():
        raise NotADirectoryError(f"请提供目录路径：{root}")

    candidates: dict[str, dict[str, object]] = {}
    for current in [root, *root.rglob("*")]:
        try:
            relative_parts = len(current.relative_to(root).parts)
        except Exception:
            relative_parts = 0
        if relative_parts > 4:
            continue
        if not current.is_dir():
            continue

        has_config = (current / "config.json").exists()
        gguf_files = [item for item in sorted(current.glob("*.gguf")) if "mmproj" not in item.name.lower()]
        if not has_config and not gguf_files:
            continue

        target_path = current if has_config else (gguf_files[0] if len(gguf_files) == 1 else current)
        engine, supported, note = detect_local_llm_engine(target_path)
        key = str(target_path.resolve())
        candidates[key] = {
            "name": target_path.name,
            "path": key,
            "engine": engine,
            "supported": supported,
            "note": note,
        }

    return sorted(
        candidates.values(),
        key=lambda item: (
            0 if item.get("supported") else 1,
            str(item.get("name", "")).lower(),
        ),
    )


def resolve_local_model_size_gb(model_path: str) -> float | None:
    path = Path(str(model_path or "")).expanduser()
    if not path.exists():
        return None
    if path.is_file():
        return round(path.stat().st_size / (1024 ** 3), 2)

    total_bytes = 0
    for pattern in ("*.gguf", "*.safetensors", "*.bin"):
        for item in path.glob(pattern):
            try:
                total_bytes += item.stat().st_size
            except Exception:
                pass
    if not total_bytes:
        return None
    return round(total_bytes / (1024 ** 3), 2)


def build_llm_hardware_summary(system: dict) -> str:
    platform_info = system.get("platform", {})
    memory = system.get("memory", {})
    cpu = system.get("cpu", {})
    gpu = system.get("gpu", {})

    parts = []
    machine = str(platform_info.get("machine") or "").strip()
    system_name = str(platform_info.get("system") or "").strip()
    if system_name or machine:
        parts.append(" / ".join(item for item in [system_name, machine] if item))

    total_mem_gb = memory.get("total_gb")
    available_mem_gb = memory.get("available_gb")
    if total_mem_gb is not None:
        mem_text = f"内存 {total_mem_gb} GB"
        if available_mem_gb is not None:
            mem_text += f"（可用约 {available_mem_gb} GB）"
        parts.append(mem_text)

    logical_count = cpu.get("logical_count")
    cpu_usage = cpu.get("usage_percent")
    if logical_count:
        cpu_text = f"CPU 线程 {logical_count}"
        if cpu_usage is not None:
            cpu_text += f"（当前占用 {cpu_usage}%）"
        parts.append(cpu_text)

    if gpu.get("cuda", {}).get("available"):
        cuda = gpu["cuda"]
        gpu_text = "CUDA 可用"
        if cuda.get("total_gb") is not None:
            gpu_text += f"（总 {cuda.get('total_gb')} GB"
            if cuda.get("free_gb") is not None:
                gpu_text += f"，空闲 {cuda.get('free_gb')} GB"
            gpu_text += "）"
        parts.append(gpu_text)
    elif gpu.get("mps", {}).get("available"):
        mps = gpu["mps"]
        gpu_text = "MPS 可用"
        if mps.get("recommended_max_gb") is not None:
            gpu_text += f"（建议上限约 {mps.get('recommended_max_gb')} GB）"
        parts.append(gpu_text)
    else:
        parts.append("仅 CPU 推理")

    return "；".join(part for part in parts if part)


def normalize_llm_load_profile(profile: str | None) -> str:
    normalized = str(profile or "").strip().lower()
    if normalized in {"conservative", "balanced", "aggressive"}:
        return normalized
    return "balanced"


def recommend_direct_llm_load_params(llm_cfg) -> dict[str, object]:
    model_path = str(llm_cfg.local_model_path or llm_cfg.model or "").strip()
    if not model_path:
        raise ValueError("请先提供本地模型路径。")

    engine = str(llm_cfg.local_engine or detect_local_llm_engine(Path(model_path))[0] or "auto").lower()
    profile = normalize_llm_load_profile(getattr(llm_cfg, "local_load_profile", None))
    system = get_system_status()
    gpu = system.get("gpu", {})
    memory = system.get("memory", {})
    cpu = system.get("cpu", {})
    total_mem_gb = float(memory.get("total_gb") or 16)
    available_mem_gb = float(memory.get("available_gb") or total_mem_gb)
    logical_count = int(cpu.get("logical_count") or os.cpu_count() or 8)
    model_size_gb = resolve_local_model_size_gb(model_path)
    mps_recommended_gb = float(gpu.get("mps", {}).get("recommended_max_gb") or 0)
    cuda_free_gb = float(gpu.get("cuda", {}).get("free_gb") or 0)

    preferred_device = "cpu"
    if gpu.get("cuda", {}).get("available"):
        preferred_device = "cuda"
    elif gpu.get("mps", {}).get("available"):
        preferred_device = "mps"

    memory_budget_gb = available_mem_gb
    if preferred_device == "cuda" and cuda_free_gb > 0:
        memory_budget_gb = min(memory_budget_gb, cuda_free_gb)
    elif preferred_device == "mps" and mps_recommended_gb > 0:
        memory_budget_gb = min(memory_budget_gb, mps_recommended_gb)

    recommended = {
        "engine": engine,
        "device": preferred_device,
        "ctx_tokens": 8192,
        "gpu_layers": None,
        "threads": max(2, min(8, logical_count // 2 or 1)),
        "batch_size": 512,
        "workers": 1,
        "profile": profile,
        "note": "",
        "model_size_gb": model_size_gb,
        "hardware_summary": build_llm_hardware_summary(system),
        "reasoning": [],
    }

    reasoning: list[str] = []
    if model_size_gb is not None:
        reasoning.append(f"检测到模型体积约 {model_size_gb} GB。")
    reasoning.append(
        f"当前优先设备为 {preferred_device.upper() if preferred_device != 'cpu' else 'CPU'}，可用于推理的保守内存预算约 {round(memory_budget_gb, 2)} GB。"
    )
    profile_label = {"conservative": "保守", "balanced": "平衡", "aggressive": "激进"}[profile]
    reasoning.append(f"已按“{profile_label}”档位生成参数。")

    if engine == "gguf":
        if preferred_device == "cpu":
            recommended["gpu_layers"] = 0
            recommended["batch_size"] = 256
            recommended["ctx_tokens"] = 4096 if (model_size_gb or 0) > 8 else 8192
            recommended["threads"] = max(2, min(10, logical_count - 1))
            reasoning.append("GGUF 在 CPU 模式下优先降低 batch，避免首轮加载和推理峰值过高。")
        else:
            if model_size_gb is not None and memory_budget_gb < model_size_gb * 1.8:
                recommended["gpu_layers"] = 24 if preferred_device == "mps" else 35
                recommended["batch_size"] = 128 if preferred_device == "mps" and (model_size_gb or 0) >= 12 else 256
                recommended["ctx_tokens"] = 4096
                reasoning.append("当前显存或统一内存偏紧，先用部分 GPU layers 和较小上下文更稳。")
            else:
                recommended["gpu_layers"] = -1
                if preferred_device == "mps" and (model_size_gb or 0) >= 12:
                    recommended["batch_size"] = 128
                else:
                    recommended["batch_size"] = 512 if total_mem_gb < 48 else 1024
                recommended["ctx_tokens"] = 8192 if total_mem_gb < 48 else 16384
                reasoning.append("当前硬件适合优先尝试全量 GPU layers。")
    elif engine == "mlx":
        recommended["gpu_layers"] = None
        recommended["batch_size"] = 512
        recommended["ctx_tokens"] = 8192 if total_mem_gb < 48 else 16384
        recommended["threads"] = max(2, min(6, logical_count // 2 or 1))
        reasoning.append("MLX 通常由框架自动处理分层，重点控制上下文长度和并发更有效。")
    elif engine == "transformers":
        recommended["gpu_layers"] = None
        recommended["batch_size"] = 256
        recommended["ctx_tokens"] = 4096 if total_mem_gb < 32 else 8192
        recommended["threads"] = max(2, min(6, logical_count // 2 or 1))
        reasoning.append("Transformers 直载更吃内存，默认优先保证稳定启动。")

    if profile == "conservative":
        recommended["ctx_tokens"] = max(2048, int(recommended["ctx_tokens"]) // 2)
        recommended["batch_size"] = max(128, int(recommended["batch_size"]) // 2)
        recommended["workers"] = 1
        if engine == "gguf" and recommended["gpu_layers"] == -1:
            recommended["gpu_layers"] = 40 if preferred_device == "cuda" else 16
        reasoning.append("保守档位会主动压缩上下文和 batch，优先减少 OOM 风险。")
    elif profile == "aggressive":
        recommended["ctx_tokens"] = min(32768, int(recommended["ctx_tokens"]) * 2)
        recommended["batch_size"] = min(2048, int(recommended["batch_size"]) * 2)
        if engine == "gguf" and preferred_device != "cpu":
            recommended["gpu_layers"] = -1
        if (model_size_gb or 0) <= 4 and memory_budget_gb >= 24:
            recommended["workers"] = 2
        reasoning.append("激进档位会提高吞吐和上下文长度，更适合在内存余量充足时尝试。")
    else:
        reasoning.append("平衡档位优先兼顾稳定性和速度，适合作为首轮默认值。")

    if engine == "transformers" and profile == "aggressive" and memory_budget_gb < 20:
        recommended["ctx_tokens"] = min(int(recommended["ctx_tokens"]), 8192)
        recommended["batch_size"] = min(int(recommended["batch_size"]), 512)
        reasoning.append("Transformers 在当前内存预算下不建议过于激进，已自动收回一档。")

    if engine == "gguf" and preferred_device == "mps" and memory_budget_gb < 10:
        recommended["gpu_layers"] = 0 if profile == "conservative" else 16
        recommended["batch_size"] = min(int(recommended["batch_size"]), 256)
        reasoning.append("MPS 可用预算较小，已进一步降低 GPU layers 和 batch。")

    if engine == "gguf" and preferred_device == "mps" and (model_size_gb or 0) >= 12:
        recommended["ctx_tokens"] = min(int(recommended["ctx_tokens"]), 4096 if profile != "aggressive" else 8192)
        recommended["batch_size"] = min(int(recommended["batch_size"]), 128 if profile != "aggressive" else 192)
        recommended["workers"] = 1
        if recommended["gpu_layers"] == -1 and profile != "aggressive":
            recommended["gpu_layers"] = 24
        reasoning.append("大体积 GGUF 在 Apple Silicon 上更容易触发解码失败，已额外压低 batch 并关闭并发。")

    recommended["note"] = " ".join(reasoning[-2:]).strip()
    recommended["reasoning"] = reasoning

    return recommended


def is_mps_oom_error(exc: Exception) -> bool:
    message = str(exc or "").lower()
    return "mps backend out of memory" in message or ("out of memory" in message and "mps" in message)


def synthesize_local_tts_with_fallback(
    *,
    inference_device: str | None,
    model_name: str | None,
    asr_model_name: str | None,
    text: str,
    output_path: Path,
    instruct: str | None = None,
    ref_audio: str | None = None,
    ref_text: str | None = None,
) -> dict:
    resolved_device = normalize_inference_device(inference_device)
    pipeline = get_pipeline(resolved_device, model_name=model_name, asr_model_name=asr_model_name)
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
        cpu_pipeline = get_pipeline("cpu", model_name=model_name, asr_model_name=asr_model_name)
        cpu_pipeline.synthesize_segment(
            text=text,
            output_path=output_path,
            instruct=instruct,
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
        return {"device": "cpu", "fallback_device": "cpu", "original_device": resolved_device}


def synthesize_local_voxcpm_tts_with_fallback(
    *,
    inference_device: str | None,
    model_name: str | None,
    text: str,
    output_path: Path,
    instruct: str | None = None,
    ref_audio: str | None = None,
    ref_text: str | None = None,
    cfg_value: float | None = None,
    inference_timesteps: int | None = None,
) -> dict:
    resolved_device = normalize_inference_device(inference_device)
    pipeline = get_voxcpm_pipeline(resolved_device, model_name=model_name)
    try:
        pipeline.synthesize_segment(
            text=text,
            output_path=output_path,
            instruct=instruct,
            ref_audio=ref_audio,
            ref_text=ref_text,
            cfg_value=cfg_value if cfg_value is not None else 2.0,
            inference_timesteps=inference_timesteps if inference_timesteps is not None else 10,
        )
        return {"device": resolved_device, "fallback_device": None}
    except Exception as exc:
        if resolved_device != "mps" or not is_mps_oom_error(exc):
            raise
        pipeline.unload()
        cpu_pipeline = get_voxcpm_pipeline("cpu", model_name=model_name)
        cpu_pipeline.synthesize_segment(
            text=text,
            output_path=output_path,
            instruct=instruct,
            ref_audio=ref_audio,
            ref_text=ref_text,
            cfg_value=cfg_value if cfg_value is not None else 2.0,
            inference_timesteps=inference_timesteps if inference_timesteps is not None else 10,
        )
        return {"device": "cpu", "fallback_device": "cpu", "original_device": resolved_device}


def synthesize_local_narration_with_fallback(
    *,
    inference_device: str | None,
    model_name: str | None,
    asr_model_name: str | None,
    segments: list[dict],
    output_name: str,
    silence_ms: int,
    role_profiles: dict,
) -> dict:
    resolved_device = normalize_inference_device(inference_device)
    pipeline = get_pipeline(resolved_device, model_name=model_name, asr_model_name=asr_model_name)
    try:
        merge_result = pipeline.synthesize_segments(
            segments=segments,
            base_name=output_name,
            silence_ms=silence_ms,
            role_profiles=role_profiles,
        )
        return {"merge_result": merge_result, "device": resolved_device, "fallback_device": None}
    except Exception as exc:
        if resolved_device != "mps" or not is_mps_oom_error(exc):
            raise
        pipeline.unload()
        cpu_pipeline = get_pipeline("cpu", model_name=model_name, asr_model_name=asr_model_name)
        merge_result = cpu_pipeline.synthesize_segments(
            segments=segments,
            base_name=output_name,
            silence_ms=silence_ms,
            role_profiles=role_profiles,
        )
        return {
            "merge_result": merge_result,
            "device": "cpu",
            "fallback_device": "cpu",
            "original_device": resolved_device,
        }


def get_audio_model_status() -> dict:
    with pipeline_lock:
        pipelines = list(pipeline_cache.values())
        voxcpm_pipelines = list(voxcpm_pipeline_cache.values())
        loaded_devices = sorted({pipeline.device for pipeline in pipelines if pipeline.model is not None})
        cached_devices = sorted({pipeline.device for pipeline in pipelines})
        asr_loaded_devices = sorted({pipeline.device for pipeline in pipelines if pipeline.is_asr_loaded()})
        asr_model_names = sorted({pipeline.asr_model_name for pipeline in pipeline_cache.values() if pipeline.is_asr_loaded()})
        audio_model_names = sorted({pipeline.model_name for pipeline in pipeline_cache.values() if pipeline.model is not None})
        voxcpm_loaded_devices = sorted({pipeline.device for pipeline in voxcpm_pipelines if pipeline.model is not None})
        voxcpm_cached_devices = sorted({pipeline.device for pipeline in voxcpm_pipelines})
        voxcpm_model_names = sorted({pipeline.model_name for pipeline in voxcpm_pipelines if pipeline.model is not None})
    return {
        "runtime": get_runtime_dependency_status(),
        "voxcpm_runtime": get_voxcpm_runtime_dependency_status(),
        "voxcpm_bridge": get_voxcpm_bridge_status(),
        "asr_runtime": get_asr_runtime_dependency_status(),
        "loaded_devices": loaded_devices,
        "cached_devices": cached_devices,
        "asr_loaded_devices": asr_loaded_devices,
        "audio_model_names": audio_model_names,
        "asr_model_names": asr_model_names,
        "voxcpm_loaded_devices": voxcpm_loaded_devices,
        "voxcpm_cached_devices": voxcpm_cached_devices,
        "voxcpm_model_names": voxcpm_model_names,
        "detected_device": OmniVoicePipeline.detect_device(),
    }


def get_gpu_status() -> dict:
    if torch is None:
        return {
            "cuda": {"available": False, "reason": "torch 未安装"},
            "mps": {"available": False, "reason": "torch 未安装"},
        }

    gpu = {
        "cuda": {
            "available": torch.cuda.is_available(),
        },
        "mps": {
            "available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        },
    }
    if gpu["cuda"]["available"]:
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            gpu["cuda"].update(
                {
                    "free_gb": round(free_bytes / (1024 ** 3), 2),
                    "total_gb": round(total_bytes / (1024 ** 3), 2),
                    "used_gb": round((total_bytes - free_bytes) / (1024 ** 3), 2),
                }
            )
        except Exception as exc:
            gpu["cuda"]["error"] = str(exc)
    if gpu["mps"]["available"] and hasattr(torch, "mps"):
        try:
            gpu["mps"].update(
                {
                    "current_allocated_gb": round(torch.mps.current_allocated_memory() / (1024 ** 3), 3),
                    "driver_allocated_gb": round(torch.mps.driver_allocated_memory() / (1024 ** 3), 3),
                    "recommended_max_gb": round(torch.mps.recommended_max_memory() / (1024 ** 3), 3),
                }
            )
        except Exception as exc:
            gpu["mps"]["error"] = str(exc)
    return gpu


def get_system_status() -> dict:
    if psutil is None:
        return {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
            "cpu": {
                "logical_count": os.cpu_count(),
                "usage_percent": None,
                "load_avg": [round(x, 2) for x in os.getloadavg()] if hasattr(os, "getloadavg") else None,
                "temperature_c": None,
                "temperature_source": None,
                "reason": "psutil 未安装",
            },
            "memory": {
                "total_gb": None,
                "used_gb": None,
                "available_gb": None,
                "usage_percent": None,
                "process_rss_gb": None,
                "reason": "psutil 未安装",
            },
            "gpu": get_gpu_status(),
        }

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
        "gpu": get_gpu_status(),
    }


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = Path("static/index.html")
    if not html_path.exists():
        return "<h1>index.html 不存在</h1>"
    return Response(
        content=html_path.read_text(encoding="utf-8"),
        media_type="text/html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/local-path-presets")
def list_local_path_presets():
    try:
        return {
            "ok": True,
            "presets": build_local_path_preset_catalog(),
        }
    except Exception as exc:
        raise_api_error(exc)


@app.get("/api/llm-defaults")
def llm_defaults():
    try:
        return {
            "ok": True,
            "llm": load_ai_helper_llm_defaults(),
        }
    except Exception as exc:
        raise_api_error(exc)


@app.get("/api/default-prompts")
def default_prompts():
    from role_analyzer import CHAPTER_RULE_DETECTION_PROMPT, OPTIMIZE_ANALYZE_PROMPT, SEGMENT_OPTIMIZE_PROMPT
    return {
        "ok": True,
        "prompts": {
            "chapter_prompt": CHAPTER_RULE_DETECTION_PROMPT.strip(),
            "system_prompt": DEFAULT_SYSTEM_PROMPT.strip(),
            "segment_prompt": SEGMENT_PLAN_PROMPT.strip(),
            "optimize_prompt": TEXT_OPTIMIZATION_PROMPT.strip(),
            "segment_optimize_prompt": SEGMENT_OPTIMIZE_PROMPT.strip(),
            "optimize_analyze_prompt": OPTIMIZE_ANALYZE_PROMPT.strip(),
            "speaker_verification_prompt": SPEAKER_VERIFICATION_PROMPT.strip(),
            "alias_resolution_prompt": CHARACTER_ALIAS_RESOLUTION_PROMPT.strip(),
        },
    }


@app.post("/api/verify-speakers")
def verify_speakers_endpoint(req: VerifySpeakersRequest):
    try:
        segments = [dict(s) for s in req.segments]
        result = verify_speakers_pass(segments, req.llm)
        return {"ok": True, "segments": result}
    except Exception as exc:
        raise_api_error(exc)


@app.post("/api/merge-aliases")
def merge_aliases_endpoint(req: MergeAliasesRequest):
    try:
        segments = [dict(s) for s in req.segments]
        registry = resolve_character_aliases(segments, req.llm)
        registry.apply_to_segments(segments)
        return {
            "ok": True,
            "segments": segments,
            "alias_map": registry.alias_dict(),
            "canonical_names": registry.canonical_names(),
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
        voice_engine = str(req.voice_engine or "").strip().lower()
        if voice_engine == "microsoft":
            output_path = resolve_named_output_path(output_name, ".wav", temp_category="preview")
            file_path = synthesize_microsoft_tts(
                text=req.text,
                output_path=output_path,
                voice_name=req.voice_name or "",
                style=req.instruct,
            )
            return {
                "ok": True,
                "file": file_path,
                "audio_url": build_output_file_url(output_path),
                "engine": "microsoft",
                "voice_name": req.voice_name,
                "voice_locale": req.voice_locale,
            }

        backend_data = req.backend.model_dump() if req.backend else None
        mode, remote_base_url, remote_api_key, inference_device = resolve_audio_backend(
            backend_data,
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
                    "voice_engine": req.voice_engine,
                    "cfg_value": req.cfg_value,
                    "inference_timesteps": req.inference_timesteps,
                    "output_name": output_name,
                    "inference_device": inference_device,
                },
                remote_api_key,
            )
            remote_resp["audio_url"] = absolutize_remote_file_url(remote_base_url, remote_resp.get("audio_url"))
            return remote_resp

        if mode == "mimo":
            output_path = resolve_named_output_path(output_name, ".wav", temp_category="preview")
            result = synthesize_mimo_tts(
                backend=backend_data,
                text=req.text,
                output_path=output_path,
                instruct=req.instruct,
                ref_audio=req.ref_audio,
                voice_name=req.voice_name,
                voice_engine=req.voice_engine,
            )
            return {
                "ok": True,
                "file": str(output_path),
                "audio_url": build_output_file_url(output_path),
                **result,
            }

        output_path = resolve_named_output_path(output_name, ".wav", temp_category="preview")
        if voice_engine == "voxcpm":
            if get_voxcpm_runtime_dependency_status().get("ready"):
                result = synthesize_local_voxcpm_tts_with_fallback(
                    inference_device=inference_device,
                    model_name=resolve_local_audio_model_path(backend_data),
                    text=req.text,
                    output_path=output_path,
                    instruct=req.instruct,
                    ref_audio=req.ref_audio,
                    ref_text=req.ref_text,
                    cfg_value=req.cfg_value,
                    inference_timesteps=req.inference_timesteps,
                )
            else:
                bridge_result = run_voxcpm_bridge(
                    "tts",
                    {
                        "device": normalize_inference_device(inference_device),
                        "model_name": resolve_local_audio_model_path(backend_data) or VOXCPM_MODEL_NAME,
                        "text": req.text,
                        "output_path": str(output_path.resolve()),
                        "instruct": req.instruct,
                        "ref_audio": req.ref_audio,
                        "ref_text": req.ref_text,
                        "cfg_value": req.cfg_value,
                        "inference_timesteps": req.inference_timesteps,
                    },
                )
                result = {
                    "device": bridge_result.get("device"),
                    "fallback_device": None,
                    "original_device": None,
                    "bridge_python_executable": bridge_result.get("python_executable"),
                }
            return {
                "ok": True,
                "file": str(output_path),
                "audio_url": build_output_file_url(output_path),
                "engine": "voxcpm",
                "device": result.get("device"),
                "fallback_device": result.get("fallback_device"),
                "original_device": result.get("original_device"),
                "bridge_python_executable": result.get("bridge_python_executable"),
            }

        result = synthesize_local_tts_with_fallback(
            inference_device=inference_device,
            model_name=resolve_local_audio_model_path(backend_data),
            asr_model_name=resolve_local_asr_model_path(backend_data),
            text=req.text,
            output_path=output_path,
            instruct=req.instruct,
            ref_audio=req.ref_audio,
            ref_text=req.ref_text,
        )

        return {
            "ok": True,
            "file": str(output_path),
            "audio_url": build_output_file_url(output_path),
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


def _load_book_voice_parser():
    import sys as _sys
    from pathlib import Path as _Path

    # 兼容两种放置方式：项目根目录/BookVoiceParser 或 上一级目录/BookVoiceParser
    _app_dir = _Path(__file__).resolve().parent
    _bvp_candidates = [
        _app_dir / "BookVoiceParser",
        _app_dir.parent / "BookVoiceParser",
    ]
    _bvp_root = next((candidate for candidate in _bvp_candidates if candidate.exists()), _bvp_candidates[0])
    if str(_bvp_root) not in _sys.path:
        _sys.path.insert(0, str(_bvp_root))

    from book_voice_parser import LLMRouterConfig, parse_novel, route_to_llm
    from book_voice_parser.schema import AttributionType, SegmentEx
    from book_voice_parser.spc_ranker import OpenAICompatibleSPCRanker

    return parse_novel, route_to_llm, LLMRouterConfig, OpenAICompatibleSPCRanker, SegmentEx, AttributionType


def _build_bvp_llm_config(llm):
    _, _, LLMRouterConfig, _, _, _ = _load_book_voice_parser()
    return LLMRouterConfig(
        base_url=str(llm.base_url),
        model=str(llm.model),
        api_key=str(llm.api_key) if llm.api_key else "local",
        max_tokens=max(int(llm.max_tokens or 1024), 1024),
        temperature=float(llm.temperature or 0.0),
    )


def _confidence_label(c: float | None) -> str:
    if c is None:
        return ""
    if c >= 0.85:
        return "high"
    if c >= 0.70:
        return "medium"
    return "low"


def _bvp_segment_to_dict(seg) -> dict[str, object]:
    return {
        "speaker": seg.speaker,
        "text": seg.text,
        "emotion": seg.emotion or "neutral",
        "style": seg.style or "",
        "ref_audio": seg.ref_audio,
        "ref_text": seg.ref_text,
        "voice_engine": seg.voice_engine or "",
        "voice_name": seg.voice_name or "",
        "voice_locale": seg.voice_locale or "",
        "cfg_value": seg.cfg_value or "",
        "inference_timesteps": seg.inference_timesteps or "",
        # BookVoiceParser 扩展字段。部分字段前端不直接展示，但 LLM 复核需要。
        "quote_id": seg.quote_id,
        "confidence": seg.confidence,
        "evidence": seg.evidence,
        "addressee": seg.addressee,
        "attribution_type": str(seg.attribution_type.value if hasattr(seg.attribution_type, "value") else seg.attribution_type or ""),
        "candidates": seg.candidates,
        "candidate_sources": seg.candidate_sources,
        "scene_characters": seg.scene_characters,
        "context_before": seg.context_before,
        "context_after": seg.context_after,
        # 前端 _confidence / _evidence 内部字段
        "_confidence": _confidence_label(seg.confidence),
        "_evidence": seg.evidence or "",
        "_needs_review": seg.speaker in {"未知", "UNKNOWN"} or (seg.confidence is not None and seg.confidence < 0.5),
    }


def _dict_to_bvp_segment(data: dict[str, object]):
    _, _, _, _, SegmentEx, AttributionType = _load_book_voice_parser()
    attribution_type = data.get("attribution_type") or None
    if attribution_type:
        try:
            attribution_type = AttributionType(str(attribution_type))
        except Exception:
            attribution_type = str(attribution_type)

    def _number_or_none(value):
        return None if value in {"", None} else value

    return SegmentEx(
        speaker=str(data.get("speaker") or "旁白"),
        text=str(data.get("text") or ""),
        emotion=str(data.get("emotion") or "neutral"),
        style=data.get("style") or None,
        ref_audio=data.get("ref_audio") or None,
        ref_text=data.get("ref_text") or None,
        voice_engine=data.get("voice_engine") or None,
        voice_name=data.get("voice_name") or None,
        voice_locale=data.get("voice_locale") or None,
        cfg_value=_number_or_none(data.get("cfg_value")),
        inference_timesteps=_number_or_none(data.get("inference_timesteps")),
        addressee=data.get("addressee") or None,
        confidence=float(data.get("confidence") if data.get("confidence") is not None else 1.0),
        evidence=data.get("evidence") or None,
        attribution_type=attribution_type,
        quote_id=data.get("quote_id") or None,
        candidates=list(data.get("candidates") or []),
        candidate_sources=dict(data.get("candidate_sources") or {}),
        scene_characters=list(data.get("scene_characters") or []),
        context_before=str(data.get("context_before") or ""),
        context_after=str(data.get("context_after") or ""),
    )


@app.post("/api/parse_v2")
def parse_text_v2(req: ParseV2Request):
    """
    结构化解析：BookVoiceParser 规则层 + 可选 LLM 复核。
    返回与 /api/parse 兼容的 segments 列表，并附带 confidence / evidence 等扩展字段。
    """
    try:
        parse_novel, route_to_llm, _, OpenAICompatibleSPCRanker, _, _ = _load_book_voice_parser()

        # 构建 SPC ranker（如果需要 LLM 内联推断）
        implicit_ranker = None
        if req.implicit_strategy == "llm_spc" and req.llm:
            implicit_ranker = OpenAICompatibleSPCRanker(req.llm)

        result = parse_novel(
            req.text,
            role_hints=req.role_hints,
            return_result=True,
            review_threshold=req.review_threshold,
            implicit_strategy=req.implicit_strategy,
            implicit_ranker=implicit_ranker,
            include_narration=True,
        )

        segments = result.segments

        # 可选：对低置信度片段调用 LLM 后处理复核
        llm_stats = {}
        if req.use_llm_review and req.llm:
            segments, llm_stats = route_to_llm(
                segments,
                _build_bvp_llm_config(req.llm),
                threshold=req.review_threshold,
            )

        return {
            "ok": True,
            "segments": [_bvp_segment_to_dict(seg) for seg in segments],
            "stats": {
                **result.stats,
                "llm_review": llm_stats,
                "review_items_count": len(result.review_items),
            },
        }
    except Exception as exc:
        raise_api_error(exc)


@app.post("/api/parse_v2/review")
def review_parse_v2_segments(req: ParseV2ReviewRequest):
    """对 BookVoiceParser 低置信度片段执行独立 LLM 复核。"""
    try:
        _, route_to_llm, _, _, _, _ = _load_book_voice_parser()
        segments = [_dict_to_bvp_segment(item) for item in req.segments]
        reviewed, llm_stats = route_to_llm(
            segments,
            _build_bvp_llm_config(req.llm),
            threshold=req.review_threshold,
        )
        return {
            "ok": True,
            "segments": [_bvp_segment_to_dict(seg) for seg in reviewed],
            "stats": {
                "llm_review": llm_stats,
                "review_items_count": sum(1 for seg in reviewed if seg.confidence < req.review_threshold),
            },
        }
    except Exception as exc:
        raise_api_error(exc)


@app.post("/api/parse_v2/review-one")
def review_parse_v2_segment(req: ParseV2ReviewOneRequest):
    """逐条复核 BookVoiceParser 低置信度片段，便于前端展示真实进度和断点状态。"""
    try:
        if req.index < 0 or req.index >= len(req.segments):
            raise HTTPException(status_code=400, detail=f"review index out of range: {req.index}")

        _, route_to_llm, _, _, _, _ = _load_book_voice_parser()
        segments = [_dict_to_bvp_segment(item) for item in req.segments]
        original = segments[req.index].model_copy()

        # route_to_llm 会遍历列表并复核所有低于阈值的片段。这里把非目标片段
        # 临时抬到阈值以上，使每次 HTTP 请求只复核一个目标片段，同时保留目标
        # 之前的已确定说话人作为 recent_speakers 上下文。
        routed_segments = [seg.model_copy() for seg in segments]
        for idx, seg in enumerate(routed_segments):
            if idx != req.index and seg.confidence < req.review_threshold:
                seg.confidence = req.review_threshold

        reviewed, llm_stats = route_to_llm(
            routed_segments,
            _build_bvp_llm_config(req.llm),
            threshold=req.review_threshold,
        )
        updated = reviewed[req.index]
        failed = bool((llm_stats or {}).get("failed"))
        changed = updated.speaker != original.speaker
        confirmed = not failed and not changed and updated.confidence > original.confidence
        return {
            "ok": True,
            "index": req.index,
            "quote_id": updated.quote_id,
            "segment": _bvp_segment_to_dict(updated),
            "stats": {
                "llm_review": llm_stats,
                "changed": changed,
                "confirmed": confirmed,
                "failed": failed,
                "before": {
                    "speaker": original.speaker,
                    "confidence": original.confidence,
                    "evidence": original.evidence,
                },
                "after": {
                    "speaker": updated.speaker,
                    "confidence": updated.confidence,
                    "evidence": updated.evidence,
                },
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise_api_error(exc)


@app.post("/api/chapter-detect")
def chapter_detect(req: SegmentPlanRequest):
    try:
        chunks, info = detect_chapter_structure_with_info(req.text, req.llm)
        return {
            "ok": True,
            "chunks": chunks,
            "info": info,
        }
    except Exception as exc:
        raise_api_error(exc)


@app.post("/api/text-preprocess")
def text_preprocess(req: ParseV2Request):
    """执行章节识别共用的文本预处理，但不做章节识别。"""
    try:
        processed, info = preprocess_text_for_chapter_detection(req.text)
        return {
            "ok": True,
            "text": processed,
            "info": info,
        }
    except Exception as exc:
        raise_api_error(exc)


@app.post("/api/segment-plan")
def segment_plan(req: SegmentPlanRequest):
    try:
        chunks, info = intelligent_segment_text_with_info(req.text, req.llm)
        return {
            "ok": True,
            "chunks": chunks,
            "info": info,
        }
    except Exception as exc:
        raise_api_error(exc)


@app.post("/api/segment-plan-chunks")
def segment_plan_chunks(req: ChunkOptimizeRequest):
    try:
        dump_request_debug("segment_plan_request", {
            "output_mode": getattr(req.llm, "output_mode", None),
            "analysis_combo": getattr(req.llm, "analysis_combo", None),
            "local_runtime": getattr(req.llm, "local_runtime", None),
            "model": getattr(req.llm, "model", None),
            "max_tokens": getattr(req.llm, "max_tokens", None),
            "chunk_titles": [item.title for item in req.chunks],
        })
        source_chunks = [item.model_dump() for item in req.chunks]
        planned = []
        for chunk in source_chunks:
            planned.extend(intelligent_segment_seed_chunk(chunk, req.llm))
        return {
            "ok": True,
            "chunks": planned,
        }
    except Exception as exc:
        mode = getattr(req.llm, "output_mode", None)
        combo = getattr(req.llm, "analysis_combo", None)
        runtime = getattr(req.llm, "local_runtime", None)
        enriched = RuntimeError(
            f"{exc} [segment-plan-chunks output_mode={mode!r} analysis_combo={combo!r} local_runtime={runtime!r}]"
        )
        raise_api_error(enriched)


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


@app.post("/api/segment-optimize")
def segment_optimize(req: SegmentPlanRequest):
    try:
        chunks, info = segment_and_optimize_text_with_info(req.text, req.llm)
        return {
            "ok": True,
            "chunks": chunks,
            "info": info,
        }
    except Exception as exc:
        raise_api_error(exc)


@app.post("/api/segment-optimize-chunks")
def segment_optimize_chunks(req: ChunkOptimizeRequest):
    try:
        source_chunks = [item.model_dump() for item in req.chunks]
        optimized = []
        for chunk in source_chunks:
            optimized.extend(segment_and_optimize_seed_chunk(chunk, req.llm))
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


@app.post("/api/optimize-analyze-chunks")
def optimize_analyze_chunks(req: AnalyzeChunksRequest):
    try:
        chunks = [item.model_dump() for item in req.chunks]
        segments = optimize_and_analyze_chunks_with_llm(chunks, req.llm)
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
        backend_data = req.backend.model_dump() if req.backend else None
        mode, remote_base_url, remote_api_key, inference_device = resolve_audio_backend(
            backend_data,
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
            model_name=resolve_local_audio_model_path(backend_data),
            asr_model_name=resolve_local_asr_model_path(backend_data),
            segments=segments,
            output_name=output_name,
            silence_ms=req.silence_ms,
            role_profiles=role_profiles,
        )
        merge_result = result["merge_result"]
        wav_paths = [Path(path) for path in merge_result.get("wav_paths", build_segment_wav_paths(output_name, len(segments)))]
        response = build_merge_response_payload(
            merge_result=merge_result,
            wav_paths=wav_paths,
            lyric_lines=build_lyric_lines(segments),
            output_name=output_name,
            silence_ms=req.silence_ms,
        )
        response["device"] = result.get("device")
        response["fallback_device"] = result.get("fallback_device")
        response["original_device"] = result.get("original_device")
        return response
    except Exception as exc:
        raise_api_error(exc)


@app.post("/api/auto-narrate")
def auto_narrate(req: AutoNarrateRequest):
    try:
        output_name = sanitize_output_name(req.output_name, "auto_narration")
        segments = analyze_text_with_llm(req.text, req.llm)
        role_profiles = {k: v.model_dump() for k, v in (req.role_profiles or {}).items()}
        backend_data = req.backend.model_dump() if req.backend else None
        mode, remote_base_url, remote_api_key, inference_device = resolve_audio_backend(
            backend_data,
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
            model_name=resolve_local_audio_model_path(backend_data),
            asr_model_name=resolve_local_asr_model_path(backend_data),
            segments=segments,
            output_name=output_name,
            silence_ms=req.silence_ms,
            role_profiles=role_profiles,
        )
        merge_result = result["merge_result"]
        wav_paths = [Path(path) for path in merge_result.get("wav_paths", build_segment_wav_paths(output_name, len(segments)))]
        response = build_merge_response_payload(
            merge_result=merge_result,
            wav_paths=wav_paths,
            lyric_lines=build_lyric_lines(segments),
            output_name=output_name,
            silence_ms=req.silence_ms,
        )
        response["segments"] = segments
        response["device"] = result.get("device")
        response["fallback_device"] = result.get("fallback_device")
        response["original_device"] = result.get("original_device")
        return response
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
            ],
            max_tokens=64,
            purpose="连通性测试",
        )
        return {
            "ok": True,
            "message": content,
        }
    except Exception as exc:
        raise_api_error(exc)


@app.post("/api/model/load-llm")
def load_llm_model(req: ConnectivityTestRequest):
    try:
        if (req.llm.local_runtime or "").strip().lower() != "direct":
            client = OpenAICompatibleClient(req.llm)
            content = client.chat_text(
                [
                    {"role": "system", "content": "Reply in one short sentence."},
                    {"role": "user", "content": "Say hello briefly."},
                ],
                max_tokens=64,
                purpose="连通性测试",
            )
            return {
                "ok": True,
                "mode": "openai_compat",
                "message": content,
                "model_name": req.llm.local_model_path or req.llm.model,
            }

        unload_all_local_llm_runners()
        runner = get_local_llm_runner(
            model_path=req.llm.local_model_path or req.llm.model,
            engine=req.llm.local_engine,
            device=req.llm.local_device,
            ctx_tokens=req.llm.local_ctx_tokens,
            gpu_layers=req.llm.local_gpu_layers,
            threads=req.llm.local_threads,
            batch_size=req.llm.local_batch_size,
        )
        loaded = runner.load()
        return {
            "ok": True,
            "mode": "direct",
            **loaded,
        }
    except Exception as exc:
        raise_api_error(exc)


@app.post("/api/llm/recommend-load-config")
def recommend_llm_load_config(req: ConnectivityTestRequest):
    try:
        return {
            "ok": True,
            "recommendation": recommend_direct_llm_load_params(req.llm),
        }
    except Exception as exc:
        raise_api_error(exc)


@app.post("/api/model/unload-llm")
def unload_llm_model():
    try:
        unload_all_local_llm_runners()
        return {
            "ok": True,
            "status": get_local_llm_status(),
        }
    except Exception as exc:
        raise_api_error(exc)


@app.post("/api/llm/list-models")
def list_llm_models(req: ListLLMModelsRequest):
    try:
        base_url = normalize_openai_compat_base_url(req.base_url)
        models_url = f"{base_url}/models"
        headers: dict[str, str] = {}
        if req.api_key:
            headers["Authorization"] = f"Bearer {req.api_key}"
        timeout = httpx.Timeout(connect=10.0, read=15.0, write=10.0, pool=10.0)
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            resp = client.get(models_url, headers=headers)
            raw_text = resp.text
            try:
                data = resp.json()
            except Exception:
                raise RuntimeError(
                    f"服务商 /models 接口返回了非 JSON 内容（HTTP {resp.status_code}）："
                    f"{raw_text[:200]}"
                )
            if not resp.is_success:
                raise RuntimeError(data.get("error", {}).get("message", "") or data.get("detail") or str(data))
        raw_models = data.get("data") or data.get("models") or []
        if isinstance(raw_models, list):
            models = [
                {"id": str(m.get("id", "")), "owned_by": str(m.get("owned_by", ""))}
                for m in raw_models
                if isinstance(m, dict) and m.get("id")
            ]
        else:
            models = []
        return {
            "ok": True,
            "models": models,
        }
    except Exception as exc:
        raise_api_error(exc)


@app.post("/api/llm/scan-local-models")
def scan_local_models(req: ScanLocalModelsRequest):
    try:
        models = scan_local_llm_models(req.root_path)
        return {
            "ok": True,
            "models": models,
        }
    except Exception as exc:
        raise_api_error(exc)


@app.get("/api/microsoft-voices")
def microsoft_voices():
    try:
        return {
            "ok": True,
            "voices": get_microsoft_tts_catalog(),
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
        backend = req.backend.model_dump() if req.backend else {}
        mode = str(backend.get("mode") or "local").strip().lower()
        if req.transcribe_ref_text and mode == "remote":
            raise RuntimeError("当前仅支持本地音频识别模型自动转写 ref_text，远程音频后端暂不支持该流程。")

        inference_device = backend.get("inference_device")
        pipeline = None
        if req.transcribe_ref_text:
            pipeline = get_pipeline(
                inference_device,
                model_name=resolve_local_audio_model_path(backend),
                asr_model_name=resolve_local_asr_model_path(backend),
            )
            if not pipeline.is_asr_loaded():
                raise RuntimeError("音频识别模型尚未加载。请先在模型配置页加载音频识别模型后再进行批量导入。")

        imported = []
        for item in req.files:
            target = save_ref_audio_from_base64(item.filename, item.content_base64)
            ref_text = ""
            if req.transcribe_ref_text and pipeline is not None:
                ref_text = pipeline.transcribe_audio_file(target)
            imported.append(
                {
                    "name": Path(item.filename).stem,
                    "style": "",
                    "ref_text": ref_text,
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


@app.post("/api/transcribe-ref-audio")
def transcribe_ref_audio(req: TranscribeRefAudioRequest):
    try:
        backend = req.backend.model_dump() if req.backend else {}
        mode = str(backend.get("mode") or "local").strip().lower()
        if mode == "remote":
            raise RuntimeError("当前仅支持本地音频识别模型自动转写 ref_text，远程音频后端暂不支持该流程。")

        ref_audio = str(req.ref_audio or "").strip()
        if not ref_audio:
            raise RuntimeError("请先提供参考音频路径。")

        inference_device = backend.get("inference_device")
        pipeline = get_pipeline(
            inference_device,
            model_name=resolve_local_audio_model_path(backend),
            asr_model_name=resolve_local_asr_model_path(backend),
        )
        if not pipeline.is_asr_loaded():
            raise RuntimeError("音频识别模型尚未加载。请先在模型配置页加载音频识别模型。")

        ref_text = pipeline.transcribe_audio_file(ref_audio)
        return {
            "ok": True,
            "ref_text": ref_text,
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
        if mode == "mimo":
            data = validate_mimo_tts_backend(backend)
            unload_all_audio_pipelines()
            return {
                "ok": True,
                "mode": "mimo",
                "engine": "mimo",
                "model_name": data["model"],
                "voice": data["voice"],
                "base_url": data["base_url"],
                "message": "MiMo TTS v2.5 为远程合成服务，本地无需加载语音生成模型。",
            }

        unload_all_audio_pipelines()
        inference_device = backend.get("inference_device")
        local_audio_engine = resolve_local_audio_engine(backend)
        if local_audio_engine == "voxcpm":
            if get_voxcpm_runtime_dependency_status().get("ready"):
                pipeline = get_voxcpm_pipeline(
                    inference_device,
                    model_name=resolve_local_audio_model_path(backend),
                )
                pipeline.load()
                return {
                    "ok": True,
                    "mode": "local",
                    "engine": "voxcpm",
                    "device": pipeline.device,
                    "model_name": pipeline.model_name,
                }

            bridge_result = run_voxcpm_bridge(
                "load",
                {
                    "device": normalize_inference_device(inference_device),
                    "model_name": resolve_local_audio_model_path(backend) or VOXCPM_MODEL_NAME,
                },
            )
            return {
                "ok": True,
                "mode": "local",
                "engine": "voxcpm",
                "device": bridge_result.get("device"),
                "model_name": bridge_result.get("model_name"),
                "bridge_python_executable": bridge_result.get("python_executable"),
                "message": "当前主服务环境未直装 VoxCPM，已通过专用环境完成可用性检查。实际生成时也会自动走该专用环境。",
            }

        pipeline = get_pipeline(
            inference_device,
            model_name=resolve_local_audio_model_path(backend),
            asr_model_name=resolve_local_asr_model_path(backend),
        )
        pipeline.load()
        return {
            "ok": True,
            "mode": "local",
            "engine": "omnivoice",
            "device": pipeline.device,
            "model_name": pipeline.model_name,
        }
    except Exception as exc:
        raise_api_error(exc)


@app.post("/api/model/load-asr")
def load_asr_model(req: ModelLoadRequest):
    try:
        backend = req.backend.model_dump() if req.backend else {}
        mode = str(backend.get("mode") or "local").strip().lower()
        if mode == "remote":
            raise RuntimeError("当前只支持本地加载音频识别模型，远程音频后端暂不支持。")

        inference_device = backend.get("inference_device")
        pipeline = get_pipeline(
            inference_device,
            model_name=resolve_local_audio_model_path(backend),
            asr_model_name=resolve_local_asr_model_path(backend),
        )
        pipeline.load_asr_model()
        return {
            "ok": True,
            "mode": "local",
            "device": pipeline.device,
            "asr_model_name": pipeline.asr_model_name,
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
        if mode == "mimo":
            return {"ok": True, "mode": "mimo", "engine": "mimo", "message": "MiMo TTS v2.5 为远程合成服务，当前无需在本地卸载。"}

        inference_device = normalize_inference_device(backend.get("inference_device"))
        unload_all_audio_pipelines()
        return {
            "ok": True,
            "mode": "local",
            "engine": resolve_local_audio_engine(backend),
            "device": inference_device,
            "status": get_audio_model_status(),
        }
    except Exception as exc:
        raise_api_error(exc)


@app.post("/api/model/unload-asr")
def unload_asr_model(req: ModelLoadRequest):
    try:
        backend = req.backend.model_dump() if req.backend else {}
        mode = str(backend.get("mode") or "local").strip().lower()
        if mode == "remote":
            return {"ok": True, "mode": "remote", "message": "远程模式不在本地卸载音频识别模型。"}

        inference_device = normalize_inference_device(backend.get("inference_device"))
        pipeline = get_pipeline(
            inference_device,
            model_name=resolve_local_audio_model_path(backend),
            asr_model_name=resolve_local_asr_model_path(backend),
        )
        pipeline.unload_asr_model()
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
            "llm": get_local_llm_status(),
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
                "llm": get_local_llm_status(),
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
            requested = Path(file_name)
            wav_path = OUTPUT_DIR / requested
            if not wav_path.exists():
                wav_path = OUTPUT_DIR / requested.name
            if not wav_path.exists():
                raise FileNotFoundError(f"找不到分段音频: {requested}")
            wav_paths.append(wav_path)

        output_path = resolve_named_output_path(f"{output_name}_merged", ".wav", temp_category="preview_merge")
        merge_result = join_wavs_auto(wav_paths, output_path, silence_ms=req.silence_ms)
        lyric_lines = None
        if req.lyrics:
            lyric_lines = []
            for line in req.lyrics:
                speaker = (line.speaker or "").strip()
                text = line.text.strip()
                lyric_lines.append(f"{speaker}：{text}" if speaker and speaker != "旁白" else text)
        return build_merge_response_payload(
            merge_result=merge_result,
            wav_paths=wav_paths,
            lyric_lines=lyric_lines,
            output_name=output_name,
            silence_ms=req.silence_ms,
        )
    except Exception as exc:
        raise_api_error(exc)


@app.post("/api/check-files")
def check_files(req: CheckFilesRequest):
    """Check which of the given file paths exist and are non-empty on disk.

    Accepts paths exactly as returned in resp.file from /api/tts (relative to
    the server working directory, e.g. "outputs/narration_001.wav").
    Returns {"exists": {path: bool}} — true only when the file exists and
    has size > 0 so that truncated files from a failed synthesis are treated
    as missing.
    """
    result: dict[str, bool] = {}
    for raw_path in req.files[:2000]:  # safety cap
        try:
            p = Path(raw_path)
            result[raw_path] = p.is_file() and p.stat().st_size > 100
        except Exception:
            result[raw_path] = False
    return {"exists": result}


@app.get("/api/health")
def health():
    loaded_devices = sorted({pipeline.device for pipeline in pipeline_cache.values() if pipeline.model is not None})
    return {
        "ok": True,
        "app": app.title,
        "llm_loaded": get_local_llm_status().get("loaded", False),
        "model_loaded": bool(loaded_devices),
        "loaded_devices": loaded_devices,
        "default_device": OmniVoicePipeline.detect_device(),
        "audio_runtime": get_runtime_dependency_status(),
        "outputs_dir": str(OUTPUT_DIR.resolve()),
    }


@app.get("/file/{name:path}")
def get_file(name: str):
    path = Path("outputs") / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=path.name)
