from __future__ import annotations

import os
import sys
from pathlib import Path

import soundfile as sf
import torch

from common_japanese_sample_lines import OUTPUT_ROOT, ResultRow, extract_target_texts, prepare_ascii_refs, timed_generate, write_manifest


ENGINE = "OmniVoice-updated"
OMNIVOICE_SRC = Path(r"I:\code\aitts\OmniVoice")
MODEL_PATH = Path(r"I:\hf_cache\models--k2-fsa--OmniVoice\snapshots\29cde0ee295ee673d33e9ab570e7bbbe761c33b3")


def main() -> int:
    os.environ.setdefault("HF_HOME", r"I:\hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", r"I:\hf_cache")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.path.insert(0, str(OMNIVOICE_SRC))
    from omnivoice.models.omnivoice import OmniVoice

    out_dir = OUTPUT_ROOT / ENGINE
    refs = prepare_ascii_refs(out_dir)
    targets = extract_target_texts()
    model = OmniVoice.from_pretrained(str(MODEL_PATH), device_map="cuda:0", dtype=torch.float16)

    rows: list[ResultRow] = []
    for ref in refs:
        for target in targets:
            phrase_id = target["phrase_id"]
            text = target["text"]
            output_path = out_dir / f"ref{ref['ref_index']:02d}__{phrase_id}.wav"

            def call() -> None:
                audios = model.generate(
                    text=text,
                    language="Chinese",
                    ref_audio=ref["ref_audio_ascii"],
                    ref_text=ref.get("ref_text", ""),
                    num_step=32,
                )
                sf.write(str(output_path), audios[0], model.sampling_rate)

            ok, seconds, duration, rtf, error = timed_generate(call, output_path)
            rows.append(ResultRow(ENGINE, ref["voice_id"], ref["speaker"], phrase_id, text, 1, ok, str(output_path) if ok else "", seconds, duration, rtf, error))
            write_manifest(rows, out_dir)
            print(f"{ENGINE} {ref['speaker']} {phrase_id} ok={ok} sec={seconds:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
