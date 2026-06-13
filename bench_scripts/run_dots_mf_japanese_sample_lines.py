from __future__ import annotations

import os
import sys
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, r"I:\code\aitts\dots.tts\src")

from common_japanese_sample_lines import OUTPUT_ROOT, ResultRow, extract_target_texts, prepare_ascii_refs, timed_generate, write_manifest
from dots_tts.runtime import DotsTtsRuntime


ENGINE = "dots.tts-mf"
MODEL_PATH = Path(r"I:\code\aitts\dots.tts\models\dots.tts-mf")


def main() -> int:
    os.environ.setdefault("HF_HOME", r"I:\hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", r"I:\hf_cache")
    out_dir = OUTPUT_ROOT / ENGINE
    refs = prepare_ascii_refs(out_dir)
    targets = extract_target_texts()
    runtime = DotsTtsRuntime.from_pretrained(str(MODEL_PATH), precision="bfloat16", max_generate_length=500)

    rows: list[ResultRow] = []
    for ref in refs:
        for target in targets:
            phrase_id = target["phrase_id"]
            text = target["text"]
            output_path = out_dir / f"ref{ref['ref_index']:02d}__{phrase_id}.wav"

            def call() -> None:
                result = runtime.generate(
                    text=text,
                    prompt_audio_path=ref["ref_audio_ascii"],
                    prompt_text=ref.get("ref_text", ""),
                    language="ZH",
                    num_steps=4,
                    guidance_scale=1.0,
                    normalize_text=False,
                )
                audio = result["audio"].float().cpu().squeeze().numpy()
                sf.write(str(output_path), audio, int(result["sample_rate"]))

            ok, seconds, duration, rtf, error = timed_generate(call, output_path)
            rows.append(ResultRow(ENGINE, ref["voice_id"], ref["speaker"], phrase_id, text, 1, ok, str(output_path) if ok else "", seconds, duration, rtf, error))
            write_manifest(rows, out_dir)
            print(f"{ENGINE} {ref['speaker']} {phrase_id} ok={ok} sec={seconds:.2f}", flush=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
