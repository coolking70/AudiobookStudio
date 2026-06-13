from __future__ import annotations

import os
import sys
from pathlib import Path

from common_japanese_sample_lines import (
    OUTPUT_ROOT,
    ResultRow,
    extract_target_texts,
    prepare_ascii_refs,
    timed_generate,
    write_manifest,
)


INDEX_ROOT = Path(r"I:\code\aitts\index-tts")
MODEL_DIR = INDEX_ROOT / "checkpoints"


VARIANTS = [
    {
        "engine": "IndexTTS2-full-temp07",
        "infer": {"num_beams": 1, "temperature": 0.7, "top_p": 0.8, "top_k": 30},
    },
    {
        "engine": "IndexTTS2-full-topk20",
        "infer": {"num_beams": 1, "temperature": 0.8, "top_p": 0.8, "top_k": 20},
    },
    {
        "engine": "IndexTTS2-full-reppenalty5",
        "infer": {"num_beams": 1, "temperature": 0.8, "top_p": 0.8, "top_k": 30, "repetition_penalty": 5.0},
    },
]


def main() -> int:
    os.environ.setdefault("HF_HOME", r"I:\hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", r"I:\hf_cache")
    os.environ.setdefault("MODELSCOPE_CACHE", r"I:\hf_cache\modelscope")
    sys.path.insert(0, str(INDEX_ROOT))
    from indextts.infer_v2 import IndexTTS2

    refs = prepare_ascii_refs(OUTPUT_ROOT / "_indextts2_selected_params_refs")
    targets = extract_target_texts()

    model = IndexTTS2(
        cfg_path=str(MODEL_DIR / "config.yaml"),
        model_dir=str(MODEL_DIR),
        use_fp16=True,
        use_cuda_kernel=False,
        use_deepspeed=False,
        use_torch_compile=True,
    )

    warmup_path = OUTPUT_ROOT / "_indextts2_selected_params_warmup.wav"
    print("IndexTTS2 selected params warmup start", flush=True)
    model.infer(
        spk_audio_prompt=refs[0]["ref_audio_ascii"],
        text=targets[0]["text"],
        output_path=str(warmup_path),
        verbose=False,
        max_text_tokens_per_segment=120,
        num_beams=1,
    )
    print("IndexTTS2 selected params warmup done", flush=True)

    for variant in VARIANTS:
        engine = variant["engine"]
        out_dir = OUTPUT_ROOT / engine
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
                        **variant["infer"],
                    )

                ok, seconds, duration, rtf, error = timed_generate(call, output_path)
                rows.append(
                    ResultRow(
                        engine,
                        ref["voice_id"],
                        ref["speaker"],
                        phrase_id,
                        text,
                        1,
                        ok,
                        str(output_path) if ok else "",
                        seconds,
                        duration,
                        rtf,
                        error,
                    )
                )
                write_manifest(rows, out_dir)
                print(f"{engine} {ref['speaker']} {phrase_id} ok={ok} sec={seconds:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
