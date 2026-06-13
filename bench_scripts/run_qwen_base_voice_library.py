from __future__ import annotations

import os
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from common_voice_library import DEFAULT_OUTPUT_ROOT, PHRASES, REFERENCES, ResultRow, timed_generate, write_manifest


ENGINE = "Qwen3-TTS-0.6B-Base"
MODEL_PATH = Path(r"I:\hf_cache\models--Qwen--Qwen3-TTS-12Hz-0.6B-Base\snapshots\5d83992436eae1d760afd27aff78a71d676296fc")


def main() -> int:
    os.environ.setdefault("HF_HOME", r"I:\hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", r"I:\hf_cache")
    out_dir = DEFAULT_OUTPUT_ROOT / ENGINE
    out_dir.mkdir(parents=True, exist_ok=True)

    model = Qwen3TTSModel.from_pretrained(
        str(MODEL_PATH),
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )

    rows: list[ResultRow] = []
    for ref in REFERENCES:
        prompt = model.create_voice_clone_prompt(
            ref_audio=ref["ref_audio"],
            ref_text=ref["ref_text"],
        )
        for phrase_id, text in PHRASES:
            for take in range(1, 3):
                output_path = out_dir / f"{ref['voice_id']}__{phrase_id}__take{take:02d}.wav"

                def call() -> None:
                    wavs, sr = model.generate_voice_clone(
                        text=text,
                        language="Chinese",
                        voice_clone_prompt=prompt,
                        max_new_tokens=256,
                        do_sample=True,
                        top_p=1.0,
                        temperature=0.8,
                    )
                    sf.write(output_path, wavs[0], sr)

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
