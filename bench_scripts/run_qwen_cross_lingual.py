from __future__ import annotations

import os
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from common_cross_lingual import OUTPUT_ROOT, PHRASES, ResultRow, prepare_ascii_refs, timed_generate, write_manifest


ENGINE = "Qwen3-TTS-0.6B-Base"
MODEL_PATH = Path(r"I:\hf_cache\models--Qwen--Qwen3-TTS-12Hz-0.6B-Base\snapshots\5d83992436eae1d760afd27aff78a71d676296fc")


def main() -> int:
    os.environ.setdefault("HF_HOME", r"I:\hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", r"I:\hf_cache")
    out_dir = OUTPUT_ROOT / ENGINE
    refs = prepare_ascii_refs(out_dir)
    model = Qwen3TTSModel.from_pretrained(str(MODEL_PATH), device_map="cuda:0", dtype=torch.bfloat16)

    rows: list[ResultRow] = []
    for ref in refs:
        prompt = model.create_voice_clone_prompt(ref_audio=ref["ref_audio_ascii"], ref_text=ref["ref_text"])
        for phrase_id, text in PHRASES:
            output_path = out_dir / f"ref{ref['ref_index']:02d}__{phrase_id}.wav"

            def call() -> None:
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language="English",
                    voice_clone_prompt=prompt,
                    max_new_tokens=512,
                    do_sample=True,
                    top_p=1.0,
                    temperature=0.8,
                )
                sf.write(output_path, wavs[0], sr)

            ok, seconds, duration, rtf, error = timed_generate(call, output_path)
            rows.append(ResultRow(ENGINE, ref["voice_id"], ref["speaker"], phrase_id, text, 1, ok, str(output_path) if ok else "", seconds, duration, rtf, error))
            write_manifest(rows, out_dir)
            print(f"{ENGINE} {ref['speaker']} {phrase_id} ok={ok} sec={seconds:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
