from __future__ import annotations

import os
from pathlib import Path

import torch
import torchaudio
from transformers import AutoModel, AutoProcessor
from transformers.generation.utils import GenerationMixin

from common_voice_library import (
    DEFAULT_OUTPUT_ROOT,
    PHRASES,
    REFERENCES,
    ResultRow,
    timed_generate,
    write_manifest,
)


ENGINE = "MOSS-TTS-Local-Transformer"
MODEL_PATH = Path(r"I:\code\aitts\MOSS-TTS\weights\MOSS-TTS-Local-Transformer")
CODEC_PATH = Path(r"I:\code\aitts\MOSS-TTS\weights\MOSS-Audio-Tokenizer")


def install_transformers_compat() -> None:
    if hasattr(GenerationMixin, "_get_initial_cache_position"):
        return

    def _get_initial_cache_position(self, cur_len, device, model_kwargs):
        model_kwargs["cache_position"] = torch.arange(0, cur_len, device=device, dtype=torch.long)
        return model_kwargs

    GenerationMixin._get_initial_cache_position = _get_initial_cache_position


def safe_name(index: int) -> str:
    return f"ref{index:02d}"


def main() -> int:
    os.environ.setdefault("HF_HOME", r"I:\hf_cache\huggingface_user_cache")
    os.environ.setdefault("HF_HUB_CACHE", r"I:\hf_cache\huggingface_user_cache\hub")
    os.environ.setdefault("MODELSCOPE_CACHE", r"I:\hf_cache\modelscope")
    install_transformers_compat()

    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    attn_implementation = "sdpa" if device == "cuda" else "eager"

    out_dir = DEFAULT_OUTPUT_ROOT / ENGINE
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(
        str(MODEL_PATH),
        codec_path=str(CODEC_PATH),
        trust_remote_code=True,
    )
    processor.audio_tokenizer = processor.audio_tokenizer.to(device)

    model = AutoModel.from_pretrained(
        str(MODEL_PATH),
        trust_remote_code=True,
        attn_implementation=attn_implementation,
        dtype=dtype,
    )
    if not hasattr(model.config, "num_hidden_layers"):
        model.config.num_hidden_layers = model.config.language_config.num_hidden_layers
    model.to(device).eval()

    rows: list[ResultRow] = []
    with torch.no_grad():
        for ref_index, ref in enumerate(REFERENCES, start=1):
            reference = [ref["ref_audio"]]
            for phrase_id, text in PHRASES:
                for take in range(1, 3):
                    output_path = out_dir / f"{safe_name(ref_index)}__{phrase_id}__take{take:02d}.wav"

                    def call() -> None:
                        message = processor.build_user_message(text=text, reference=reference)
                        batch = processor([[message]], mode="generation")
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        outputs = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=64,
                        )
                        decoded = processor.decode(outputs)
                        if not decoded or not decoded[0].audio_codes_list:
                            raise RuntimeError("MOSS returned no decoded audio")
                        audio = decoded[0].audio_codes_list[0]
                        torchaudio.save(str(output_path), audio.unsqueeze(0).cpu(), processor.model_config.sampling_rate)

                    ok, seconds, duration, rtf, error = timed_generate(call, output_path)
                    rows.append(
                        ResultRow(
                            engine=ENGINE,
                            voice_id=ref["voice_id"],
                            speaker=ref["speaker"],
                            phrase_id=phrase_id,
                            text=text,
                            take=take,
                            ok=ok,
                            file=str(output_path) if ok else "",
                            seconds=seconds,
                            duration_seconds=duration,
                            rtf=rtf,
                            error=error,
                        )
                    )
                    print(f"{ENGINE} {ref['voice_id']} {phrase_id} take{take:02d} ok={ok} sec={seconds:.2f}", flush=True)

    write_manifest(rows, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
