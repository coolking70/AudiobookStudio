from __future__ import annotations

import os
import sys
from pathlib import Path

from common_japanese_sample_lines import OUTPUT_ROOT, ResultRow, extract_target_texts, prepare_ascii_refs, timed_generate, write_manifest


INDEX_ROOT = Path(r"I:\code\aitts\index-tts")
MODEL_DIR = INDEX_ROOT / "checkpoints"


VARIANTS = {
    "cuda-kernel": {
        "engine": "IndexTTS2-cuda-kernel",
        "init": {"use_cuda_kernel": True, "use_torch_compile": False, "use_accel": False},
        "infer": {},
    },
    "beam1": {
        "engine": "IndexTTS2-beam1",
        "init": {"use_cuda_kernel": False, "use_torch_compile": False, "use_accel": False},
        "infer": {"num_beams": 1},
    },
    "cuda-kernel-beam1": {
        "engine": "IndexTTS2-cuda-kernel-beam1",
        "init": {"use_cuda_kernel": True, "use_torch_compile": False, "use_accel": False},
        "infer": {"num_beams": 1},
    },
    "accel-beam1": {
        "engine": "IndexTTS2-accel-beam1",
        "init": {"use_cuda_kernel": False, "use_torch_compile": False, "use_accel": True},
        "infer": {"num_beams": 1},
    },
    "compile-beam1": {
        "engine": "IndexTTS2-compile-beam1",
        "init": {"use_cuda_kernel": False, "use_torch_compile": True, "use_accel": False},
        "infer": {"num_beams": 1},
    },
}


def main() -> int:
    variant_name = os.environ.get("INDEXTTS2_VARIANT", "beam1")
    if variant_name not in VARIANTS:
        raise RuntimeError(f"Unknown INDEXTTS2_VARIANT={variant_name!r}. Expected one of {sorted(VARIANTS)}")
    variant = VARIANTS[variant_name]

    os.environ.setdefault("HF_HOME", r"I:\hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", r"I:\hf_cache")
    os.environ.setdefault("MODELSCOPE_CACHE", r"I:\hf_cache\modelscope")
    sys.path.insert(0, str(INDEX_ROOT))
    from indextts.infer_v2 import IndexTTS2

    out_dir = OUTPUT_ROOT / variant["engine"]
    refs = prepare_ascii_refs(out_dir)
    targets = extract_target_texts()
    if os.environ.get("AITTS_PROBE_ONLY") == "1":
        refs = refs[:1]
        targets = targets[:1]

    model = IndexTTS2(
        cfg_path=str(MODEL_DIR / "config.yaml"),
        model_dir=str(MODEL_DIR),
        use_fp16=True,
        use_deepspeed=False,
        **variant["init"],
    )

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
            rows.append(ResultRow(variant["engine"], ref["voice_id"], ref["speaker"], phrase_id, text, 1, ok, str(output_path) if ok else "", seconds, duration, rtf, error))
            write_manifest(rows, out_dir)
            print(f"{variant['engine']} {ref['speaker']} {phrase_id} ok={ok} sec={seconds:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
