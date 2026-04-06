from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import torch
import torchaudio

from omnivoice import OmniVoice
from audio_utils import ensure_dir, join_wavs


MODEL_NAME = "k2-fsa/OmniVoice"
OUTPUT_DIR = Path("outputs")

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
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.model: Optional[OmniVoice] = None

    def load(self) -> OmniVoice:
        if self.model is None:
            self.model = OmniVoice.from_pretrained(
                self.model_name,
                device_map=self.device,
                dtype=self.dtype,
            )
        return self.model

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

        kwargs = {
            "text": text,
            "speed": speed,
            "num_step": num_step,
        }

        if ref_audio:
            kwargs["ref_audio"] = ref_audio
            if ref_text:
                kwargs["ref_text"] = ref_text
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

            # 分段里显式写的优先级最高
            style = seg.get("style") or profile.get("style")
            ref_audio = seg.get("ref_audio") or profile.get("ref_audio")
            ref_text = seg.get("ref_text") or profile.get("ref_text")

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