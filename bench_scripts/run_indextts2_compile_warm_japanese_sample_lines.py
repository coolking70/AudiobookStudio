from __future__ import annotations

import os
import sys
from pathlib import Path

from common_japanese_sample_lines import OUTPUT_ROOT, ResultRow, extract_target_texts, prepare_ascii_refs, timed_generate, write_manifest


ENGINE = "IndexTTS2-compile-beam1-warm"
INDEX_ROOT = Path(r"I:\code\aitts\index-tts")
MODEL_DIR = INDEX_ROOT / "checkpoints"


def main() -> int:
    os.environ.setdefault("HF_HOME", r"I:\hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", r"I:\hf_cache")
    os.environ.setdefault("MODELSCOPE_CACHE", r"I:\hf_cache\modelscope")
    sys.path.insert(0, str(INDEX_ROOT))
    from indextts.infer_v2 import IndexTTS2

    out_dir = OUTPUT_ROOT / ENGINE
    refs = prepare_ascii_refs(out_dir)
    targets = extract_target_texts()
    model = IndexTTS2(
        cfg_path=str(MODEL_DIR / "config.yaml"),
        model_dir=str(MODEL_DIR),
        use_fp16=True,
        use_cuda_kernel=False,
        use_deepspeed=False,
        use_torch_compile=True,
    )

    warmup_path = out_dir / "_warmup.wav"
    warmup_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"{ENGINE} warmup start", flush=True)
    model.infer(
        spk_audio_prompt=refs[0]["ref_audio_ascii"],
        text=targets[0]["text"],
        output_path=str(warmup_path),
        verbose=False,
        max_text_tokens_per_segment=120,
        num_beams=1,
    )
    print(f"{ENGINE} warmup done", flush=True)

    rows: list[ResultRow] = []
    for ref in refs:
        for target in targets:
            phrase_id = target["phrase_id"]
            text = target["text"]
            output_path = out_dir / f"ref{ref['ref_index']:02d}__{phrase_id}.wav"

            def call() -> None:
                model.infer(
                    spk_audio_prompt=ref["ref_audio_ascii"],
                    text=text,
                    output_path=str(output_path),
                    verbose=False,
                    max_text_tokens_per_segment=120,
                    num_beams=1,
                )

            ok, seconds, duration, rtf, error = timed_generate(call, output_path)
            rows.append(ResultRow(ENGINE, ref["voice_id"], ref["speaker"], phrase_id, text, 1, ok, str(output_path) if ok else "", seconds, duration, rtf, error))
            write_manifest(rows, out_dir)
            print(f"{ENGINE} {ref['speaker']} {phrase_id} ok={ok} sec={seconds:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
