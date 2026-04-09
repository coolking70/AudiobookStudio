from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, Iterable, Optional

import soundfile as sf

try:
    import torch
except ImportError:
    torch = None

try:
    import torchaudio
except ImportError:
    torchaudio = None

try:
    from omnivoice import OmniVoice, OmniVoiceGenerationConfig
except ImportError:
    OmniVoice = None
    OmniVoiceGenerationConfig = None

try:
    from omnivoice.utils.audio import load_audio as omnivoice_load_audio
except ImportError:
    omnivoice_load_audio = None

try:
    from transformers import pipeline as hf_pipeline
except ImportError:
    hf_pipeline = None

from audio_utils import ensure_dir, join_wavs


MODEL_NAME = "k2-fsa/OmniVoice"
ASR_MODEL_NAME = "openai/whisper-large-v3-turbo"
OUTPUT_DIR = Path("outputs")


def _iter_hf_cache_roots() -> list[Path]:
    candidates: list[Path] = []
    env_keys = ("HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE", "HF_HOME", "TRANSFORMERS_CACHE")
    for key in env_keys:
        value = os.getenv(key, "").strip()
        if not value:
            continue
        root = Path(value).expanduser()
        if key == "HF_HOME":
            root = root / "hub"
        candidates.append(root)

    candidates.extend(
        [
            Path(r"I:\hf_cache"),
            Path(r"I:\hf_cache\hub"),
            Path.home() / ".cache" / "huggingface" / "hub",
        ]
    )

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        normalized = str(path.resolve()) if path.exists() else str(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(path)
    return deduped


HF_CACHE_DIRS = _iter_hf_cache_roots()

VALID_INSTRUCT_ITEMS = {
    "american accent",
    "australian accent",
    "british accent",
    "canadian accent",
    "child",
    "chinese accent",
    "elderly",
    "female",
    "high pitch",
    "indian accent",
    "japanese accent",
    "korean accent",
    "low pitch",
    "male",
    "middle-aged",
    "moderate pitch",
    "portuguese accent",
    "russian accent",
    "teenager",
    "very high pitch",
    "very low pitch",
    "whisper",
    "young adult",
}


DEFAULT_ROLE_STYLES = {
    "旁白": "female, moderate pitch",
    "张三": "male, young adult, low pitch",
    "李四": "male, middle-aged, low pitch",
}


_ORIGINAL_TORCHAUDIO_LOAD = torchaudio.load if torchaudio is not None else None
_ORIGINAL_TORCHAUDIO_SAVE = torchaudio.save if torchaudio is not None else None


def _torch_available() -> bool:
    return torch is not None


def _cuda_available() -> bool:
    return bool(_torch_available() and torch.cuda.is_available())


def _mps_available() -> bool:
    return bool(_torch_available() and hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


def _patch_torchaudio_fallbacks() -> None:
    if torchaudio is None or _ORIGINAL_TORCHAUDIO_LOAD is None or _ORIGINAL_TORCHAUDIO_SAVE is None:
        return
    if getattr(torchaudio, "_omnivoice_reader_patched", False):
        return

    def _load_with_soundfile_fallback(path, *args, **kwargs):
        try:
            return _ORIGINAL_TORCHAUDIO_LOAD(path, *args, **kwargs)
        except ImportError as exc:
            if "torchcodec" not in str(exc).lower():
                raise
        if torch is None:
            raise RuntimeError("缺少 torch，无法加载参考音频。")
        data, sample_rate = sf.read(path, always_2d=True, dtype="float32")
        waveform = torch.from_numpy(data.T.copy())
        return waveform, sample_rate

    def _save_with_soundfile_fallback(path, src, sample_rate, *args, **kwargs):
        try:
            return _ORIGINAL_TORCHAUDIO_SAVE(path, src, sample_rate, *args, **kwargs)
        except ImportError as exc:
            if "torchcodec" not in str(exc).lower():
                raise

        tensor = src.detach().cpu()
        if tensor.ndim == 2:
            data = tensor.transpose(0, 1).numpy()
        else:
            data = tensor.numpy()
        sf.write(path, data, sample_rate)

    torchaudio.load = _load_with_soundfile_fallback
    torchaudio.save = _save_with_soundfile_fallback
    torchaudio._omnivoice_reader_patched = True


def ensure_runtime_dependencies() -> None:
    status = get_runtime_dependency_status()
    missing = status["missing"]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            f"当前环境缺少本地音频生成依赖：{joined}。"
            " 可以先仅使用文本分析功能，或补装对应依赖后再启用 OmniVoice。"
        )
    _patch_torchaudio_fallbacks()


def get_runtime_dependency_status() -> dict[str, object]:
    missing = []
    if torch is None:
        missing.append("torch")
    if torchaudio is None:
        missing.append("torchaudio")
    if OmniVoice is None or OmniVoiceGenerationConfig is None:
        missing.append("omnivoice")
    return {
        "ready": not missing,
        "missing": missing,
    }


def get_asr_runtime_dependency_status() -> dict[str, object]:
    missing = []
    if torch is None:
        missing.append("torch")
    if hf_pipeline is None:
        missing.append("transformers")
    return {
        "ready": not missing,
        "missing": missing,
    }


def allow_asr_autoload() -> bool:
    return os.getenv("OMNIVOICE_ALLOW_ASR_AUTOLOAD", "").strip().lower() in {"1", "true", "yes", "on"}


class OmniVoicePipeline:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        asr_model_name: str = ASR_MODEL_NAME,
        device: Optional[str] = None,
        dtype: Optional[Any] = None,
    ):
        self.model_name = self.resolve_model_name_or_path(model_name)
        self.asr_model_name = self.resolve_asr_model_name_or_path(asr_model_name)
        self.device = device or self.detect_device()
        self.dtype = dtype or self.detect_dtype(self.device)
        self.model: Optional[Any] = None
        self.asr_pipe: Optional[Any] = None
        self.voice_clone_prompt_cache: dict[tuple[str, str], object] = {}

    @staticmethod
    def resolve_model_name_or_path(model_name: str) -> str:
        explicit = os.getenv("OMNIVOICE_MODEL_PATH", "").strip()
        if explicit:
            explicit_path = Path(explicit).expanduser()
            if explicit_path.exists():
                repo_or_snapshot = OmniVoicePipeline.resolve_cached_repo_dir(explicit_path)
                return str(repo_or_snapshot) if repo_or_snapshot else str(explicit_path)
            return explicit

        explicit_path = Path(str(model_name or "")).expanduser()
        if explicit_path.exists():
            repo_or_snapshot = OmniVoicePipeline.resolve_cached_repo_dir(explicit_path)
            return str(repo_or_snapshot) if repo_or_snapshot else str(explicit_path)

        if model_name != MODEL_NAME:
            return model_name

        for cache_root in HF_CACHE_DIRS:
            repo_dir = cache_root / "models--k2-fsa--OmniVoice"
            resolved = OmniVoicePipeline.resolve_cached_repo_dir(repo_dir)
            if resolved:
                return str(resolved)

        return model_name

    @staticmethod
    def resolve_asr_model_name_or_path(model_name: str) -> str:
        explicit = os.getenv("OMNIVOICE_ASR_MODEL_PATH", "").strip()
        if explicit:
            explicit_path = Path(explicit).expanduser()
            if explicit_path.exists():
                repo_or_snapshot = OmniVoicePipeline.resolve_cached_repo_dir(explicit_path)
                return str(repo_or_snapshot) if repo_or_snapshot else str(explicit_path)
            return explicit

        explicit_path = Path(str(model_name or "")).expanduser()
        if explicit_path.exists():
            repo_or_snapshot = OmniVoicePipeline.resolve_cached_repo_dir(explicit_path)
            return str(repo_or_snapshot) if repo_or_snapshot else str(explicit_path)

        if model_name != ASR_MODEL_NAME:
            return model_name

        for cache_root in HF_CACHE_DIRS:
            repo_dir = cache_root / "models--openai--whisper-large-v3-turbo"
            resolved = OmniVoicePipeline.resolve_cached_repo_dir(repo_dir)
            if resolved:
                return str(resolved)

        return model_name

    @staticmethod
    def resolve_cached_repo_dir(repo_dir: Path) -> Path | None:
        if not repo_dir.exists():
            return None

        if (repo_dir / "model.safetensors").exists() or (repo_dir / "config.json").exists():
            return repo_dir

        refs_main = repo_dir / "refs" / "main"
        if refs_main.exists():
            revision = refs_main.read_text(encoding="utf-8").strip()
            snapshot_dir = repo_dir / "snapshots" / revision
            if snapshot_dir.exists():
                return snapshot_dir

        snapshot_root = repo_dir / "snapshots"
        if snapshot_root.exists():
            for snapshot_dir in sorted(snapshot_root.iterdir(), reverse=True):
                if (snapshot_dir / "model.safetensors").exists() or (snapshot_dir / "config.json").exists():
                    return snapshot_dir

        return None

    @staticmethod
    def detect_device() -> str:
        preferred = os.getenv("OMNIVOICE_DEVICE", "").strip().lower()
        if preferred:
            return preferred
        if _cuda_available():
            return "cuda"
        if _mps_available():
            return "mps"
        return "cpu"

    @staticmethod
    def detect_dtype(device: str) -> Any:
        if torch is None:
            return None
        preferred = os.getenv("OMNIVOICE_DTYPE", "").strip().lower()
        if preferred:
            mapping = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
            }
            if preferred in mapping:
                return mapping[preferred]
        return torch.float16 if str(device).startswith("cuda") else torch.float32

    def load(self) -> Any:
        ensure_runtime_dependencies()
        if self.model is None:
            self.model = OmniVoice.from_pretrained(
                self.model_name,
                device_map=self.device,
                dtype=self.dtype,
            )
        return self.model

    def unload(self) -> None:
        self.model = None
        self.voice_clone_prompt_cache.clear()
        gc.collect()
        if _cuda_available() and str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
        if _mps_available() and str(self.device) == "mps" and hasattr(torch, "mps"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    def load_asr_model(self) -> str:
        if hf_pipeline is None or torch is None:
            raise RuntimeError("当前环境缺少 transformers 或 torch，无法加载音频识别模型。")
        if self.asr_pipe is None:
            asr_dtype = torch.float16 if str(self.device).startswith("cuda") else torch.float32
            self.asr_pipe = hf_pipeline(
                "automatic-speech-recognition",
                model=self.asr_model_name,
                dtype=asr_dtype,
                device_map=self.device,
            )
        return self.asr_model_name

    def unload_asr_model(self) -> None:
        self.asr_pipe = None
        gc.collect()
        if _cuda_available() and str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()

    def is_asr_loaded(self) -> bool:
        return self.asr_pipe is not None

    def transcribe_audio_file(self, audio_path: str | Path) -> str:
        if self.asr_pipe is None:
            raise RuntimeError("音频识别模型尚未加载。请先在模型配置页加载音频识别模型。")
        input_dict = self.build_asr_input(audio_path)
        result = self.asr_pipe(input_dict, return_timestamps=True)
        return str(result.get("text", "")).strip()

    def build_asr_input(self, audio_path: str | Path) -> dict[str, Any]:
        asr_status = get_asr_runtime_dependency_status()
        if not asr_status["ready"]:
            raise RuntimeError(f"当前环境缺少音频识别依赖：{', '.join(asr_status['missing'])}。")
        if self.asr_pipe is None:
            raise RuntimeError("音频识别模型尚未加载。请先在模型配置页加载音频识别模型。")

        target_path = str(audio_path)
        target_sr = getattr(getattr(self.asr_pipe, "feature_extractor", None), "sampling_rate", 16000)

        try:
            if omnivoice_load_audio is not None:
                waveform = omnivoice_load_audio(target_path, target_sr)
                sample_rate = target_sr
            else:
                waveform, sample_rate = torchaudio.load(target_path)
        except Exception as exc:
            try:
                data, sample_rate = sf.read(target_path, always_2d=True, dtype="float32")
            except Exception as sf_exc:
                raise RuntimeError(
                    "当前环境无法直接解码这份音频文件。已优先尝试 OmniVoice 官方音频读取逻辑，"
                    "但仍然失败。请优先使用 wav/flac/ogg，或补充 ffmpeg 后再处理 mp3/m4a。"
                ) from sf_exc
            waveform = torch.from_numpy(data.T.copy())

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sample_rate != target_sr:
            waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
            sample_rate = target_sr

        return {
            "array": waveform.squeeze(0).cpu().numpy(),
            "sampling_rate": sample_rate,
        }

    def normalize_instruct(self, instruct: Optional[str]) -> Optional[str]:
        if not instruct:
            return None
        parts = [x.strip().lower() for x in instruct.split(",") if x.strip()]
        parts = [x for x in parts if x in VALID_INSTRUCT_ITEMS]
        if not parts:
            return None

        dedup = []
        for item in parts:
            if item not in dedup:
                dedup.append(item)
        return ", ".join(dedup)

    def get_role_profile(self, speaker: str, role_profiles: Optional[dict]) -> dict:
        if role_profiles and speaker in role_profiles:
            profile = role_profiles[speaker]
            return {
                "style": profile.get("style"),
                "ref_audio": profile.get("ref_audio"),
                "ref_text": profile.get("ref_text"),
            }

        return {
            "style": DEFAULT_ROLE_STYLES.get(speaker, "male, moderate pitch"),
            "ref_audio": None,
            "ref_text": None,
        }

    def synthesize_segment(
        self,
        text: str,
        output_path: str | Path,
        instruct: Optional[str] = None,
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        speed: float = 1.0,
        num_step: int = 32,
    ) -> str:
        ensure_runtime_dependencies()
        model = self.load()
        generation_config = OmniVoiceGenerationConfig(
            num_step=num_step,
            guidance_scale=2.0,
            denoise=True,
            preprocess_prompt=True,
            postprocess_output=True,
        )
        kwargs = {
            "text": text,
            "speed": speed,
            "generation_config": generation_config,
        }

        if ref_audio:
            normalized_ref_audio = str(ref_audio).strip()
            normalized_ref_text = str(ref_text or "").strip()
            if not normalized_ref_text and not allow_asr_autoload():
                raise RuntimeError(
                    "当前参考音频未填写 ref_text。为避免 OmniVoice 自动加载 Whisper ASR 并触发额外下载，"
                    "当前已默认禁用自动转写。请先填写对应 ref_text，"
                    "或显式设置环境变量 OMNIVOICE_ALLOW_ASR_AUTOLOAD=1 后再允许自动转写。"
                )
            cache_key = (normalized_ref_audio, normalized_ref_text)
            voice_clone_prompt = self.voice_clone_prompt_cache.get(cache_key)
            if voice_clone_prompt is None:
                voice_clone_prompt = model.create_voice_clone_prompt(
                    ref_audio=normalized_ref_audio,
                    ref_text=normalized_ref_text or None,
                )
                self.voice_clone_prompt_cache[cache_key] = voice_clone_prompt
            kwargs["voice_clone_prompt"] = voice_clone_prompt
        else:
            normalized_instruct = self.normalize_instruct(instruct)
            if normalized_instruct:
                kwargs["instruct"] = normalized_instruct

        audio_list = model.generate(**kwargs)

        if not isinstance(audio_list, list) or len(audio_list) == 0:
            raise RuntimeError("OmniVoice 没有返回有效音频")

        audio = audio_list[0]
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        torchaudio.save(str(output_path), audio.cpu(), 24000)
        return str(output_path)

    def synthesize_segments(
        self,
        segments: Iterable[dict],
        base_name: str = "result",
        silence_ms: int = 350,
        role_profiles: Optional[dict] = None,
    ) -> str:
        ensure_runtime_dependencies()
        out_dir = ensure_dir(OUTPUT_DIR)
        wav_paths: list[str] = []

        for idx, seg in enumerate(segments, start=1):
            speaker = seg.get("speaker", "旁白")
            text = seg["text"]

            profile = self.get_role_profile(speaker, role_profiles)

            ref_audio = seg.get("ref_audio") or profile.get("ref_audio")
            ref_text = seg.get("ref_text") or profile.get("ref_text")
            style = None if ref_audio else (seg.get("style") or profile.get("style"))

            wav_path = out_dir / f"{base_name}_{idx:03d}.wav"
            self.synthesize_segment(
                text=text,
                output_path=wav_path,
                instruct=style,
                ref_audio=ref_audio,
                ref_text=ref_text,
            )
            wav_paths.append(str(wav_path))

        final_path = out_dir / f"{base_name}_merged.wav"
        return join_wavs(wav_paths, final_path, silence_ms=silence_ms)
