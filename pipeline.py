from __future__ import annotations

import os
import gc
from pathlib import Path
from typing import Iterable, Optional

import soundfile as sf
import torch
import torchaudio

_ORIGINAL_TORCHAUDIO_LOAD = torchaudio.load
_ORIGINAL_TORCHAUDIO_SAVE = torchaudio.save


def _load_with_soundfile_fallback(path, *args, **kwargs):
    try:
        return _ORIGINAL_TORCHAUDIO_LOAD(path, *args, **kwargs)
    except ImportError as exc:
        if "torchcodec" not in str(exc).lower():
            raise

    data, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    waveform = torch.from_numpy(data.T.copy())
    return waveform, sample_rate


torchaudio.load = _load_with_soundfile_fallback


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


torchaudio.save = _save_with_soundfile_fallback

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from audio_utils import ensure_dir, join_wavs


MODEL_NAME = "k2-fsa/OmniVoice"
OUTPUT_DIR = Path("outputs")
HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"

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


class OmniVoicePipeline:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        self.model_name = self.resolve_model_name_or_path(model_name)
        self.device = device or self.detect_device()
        self.dtype = dtype or self.detect_dtype(self.device)
        self.model: Optional[OmniVoice] = None
        self.voice_clone_prompt_cache: dict[tuple[str, str], object] = {}

    @staticmethod
    def resolve_model_name_or_path(model_name: str) -> str:
        explicit = os.getenv("OMNIVOICE_MODEL_PATH", "").strip()
        if explicit:
            return explicit

        if model_name != MODEL_NAME:
            return model_name

        repo_dir = HF_CACHE_DIR / "models--k2-fsa--OmniVoice"
        refs_main = repo_dir / "refs" / "main"
        if refs_main.exists():
            revision = refs_main.read_text(encoding="utf-8").strip()
            snapshot_dir = repo_dir / "snapshots" / revision
            if snapshot_dir.exists():
                return str(snapshot_dir)

        snapshot_root = repo_dir / "snapshots"
        if snapshot_root.exists():
            for snapshot_dir in sorted(snapshot_root.iterdir(), reverse=True):
                if (snapshot_dir / "model.safetensors").exists():
                    return str(snapshot_dir)

        return model_name

    @staticmethod
    def detect_device() -> str:
        preferred = os.getenv("OMNIVOICE_DEVICE", "").strip()
        if preferred:
            return preferred
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def detect_dtype(device: str) -> torch.dtype:
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

    def load(self) -> OmniVoice:
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
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if str(self.device) == "mps" and hasattr(torch, "mps"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

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
        out_dir = ensure_dir(OUTPUT_DIR)
        wav_paths: list[str] = []

        for idx, seg in enumerate(segments, start=1):
            speaker = seg.get("speaker", "旁白")
            text = seg["text"]

            profile = self.get_role_profile(speaker, role_profiles)

            # 有参考音频时按官方 voice clone 路径走，不再混入 style。
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
