from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
import torchaudio


@dataclass
class PreparedReference:
    key: str
    audio_path: str
    seconds: float
    cache: dict[str, Any]


class IndexTTS2CachedAdapter:
    """Experimental multi-reference cache wrapper for IndexTTS2 bench runs.

    IndexTTS2 already caches one reference internally. This adapter keeps several
    prepared reference tensors and restores the right one before each inference.
    """

    def __init__(self, model: Any):
        self.model = model
        self.references: dict[str, PreparedReference] = {}

    @staticmethod
    def _sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def clear_active_reference(self) -> None:
        self.model.cache_spk_cond = None
        self.model.cache_s2mel_style = None
        self.model.cache_s2mel_prompt = None
        self.model.cache_mel = None
        self.model.cache_spk_audio_prompt = None
        self.model.cache_emo_cond = None
        self.model.cache_emo_audio_prompt = None

    def activate_reference(self, key: str) -> None:
        prepared = self.references[key]
        for attr, value in prepared.cache.items():
            setattr(self.model, attr, value)

    @torch.no_grad()
    def prepare_reference(self, key: str, audio_path: str, verbose: bool = False) -> PreparedReference:
        start = time.perf_counter()
        audio, sr = self.model._load_and_cut_audio(audio_path, 15, verbose)
        audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
        audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

        inputs = self.model.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)
        spk_cond_emb = self.model.get_emb(input_features, attention_mask)

        _, s_ref = self.model.semantic_codec.quantize(spk_cond_emb)
        ref_mel = self.model.mel_fn(audio_22k.to(spk_cond_emb.device).float())
        ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)
        feat = torchaudio.compliance.kaldi.fbank(
            audio_16k.to(ref_mel.device),
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000,
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        style = self.model.campplus_model(feat.unsqueeze(0))
        prompt_condition = self.model.s2mel.models["length_regulator"](
            s_ref,
            ylens=ref_target_lengths,
            n_quantizers=3,
            f0=None,
        )[0]

        emo_audio, _ = self.model._load_and_cut_audio(audio_path, 15, verbose, sr=16000)
        emo_inputs = self.model.extract_features(emo_audio, sampling_rate=16000, return_tensors="pt")
        emo_input_features = emo_inputs["input_features"].to(self.model.device)
        emo_attention_mask = emo_inputs["attention_mask"].to(self.model.device)
        emo_cond_emb = self.model.get_emb(emo_input_features, emo_attention_mask)

        self._sync()
        prepared = PreparedReference(
            key=key,
            audio_path=audio_path,
            seconds=time.perf_counter() - start,
            cache={
                "cache_spk_cond": spk_cond_emb,
                "cache_s2mel_style": style,
                "cache_s2mel_prompt": prompt_condition,
                "cache_mel": ref_mel,
                "cache_spk_audio_prompt": audio_path,
                "cache_emo_cond": emo_cond_emb,
                "cache_emo_audio_prompt": audio_path,
            },
        )
        self.references[key] = prepared
        return prepared

    def prepare_many(self, refs: list[dict], key_field: str = "voice_id", audio_field: str = "ref_audio_ascii") -> list[PreparedReference]:
        prepared_refs = []
        for ref in refs:
            prepared_refs.append(self.prepare_reference(str(ref[key_field]), str(ref[audio_field])))
        return prepared_refs

    def infer_with_reference(self, key: str, **kwargs: Any) -> Any:
        self.activate_reference(key)
        return self.model.infer(**kwargs)
