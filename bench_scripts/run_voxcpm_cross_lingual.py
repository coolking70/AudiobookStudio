from __future__ import annotations

import os
import sys

from common_cross_lingual import OUTPUT_ROOT, PHRASES, ResultRow, prepare_ascii_refs, timed_generate, write_manifest

sys.path.insert(0, str(OUTPUT_ROOT.parents[1]))

from pipeline import VoxCPMPipeline


ENGINE = "VoxCPM2-2.0.3"


def main() -> int:
    os.environ.setdefault("HF_HOME", r"I:\hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", r"I:\hf_cache")
    out_dir = OUTPUT_ROOT / ENGINE
    refs = prepare_ascii_refs(out_dir)
    pipeline = VoxCPMPipeline(device="cuda")

    rows: list[ResultRow] = []
    for ref in refs:
        for phrase_id, text in PHRASES:
            output_path = out_dir / f"ref{ref['ref_index']:02d}__{phrase_id}.wav"

            def call() -> None:
                pipeline.synthesize_segment(
                    text=text,
                    output_path=output_path,
                    ref_audio=ref["ref_audio_ascii"],
                    ref_text=ref["ref_text"],
                    cfg_value=2.0,
                    inference_timesteps=10,
                )

            ok, seconds, duration, rtf, error = timed_generate(call, output_path)
            rows.append(ResultRow(ENGINE, ref["voice_id"], ref["speaker"], phrase_id, text, 1, ok, str(output_path) if ok else "", seconds, duration, rtf, error))
            write_manifest(rows, out_dir)
            print(f"{ENGINE} {ref['speaker']} {phrase_id} ok={ok} sec={seconds:.2f}", flush=True)
    pipeline.unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
