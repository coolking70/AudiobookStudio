from __future__ import annotations

import os
import sys

from common_voice_library import DEFAULT_OUTPUT_ROOT, PHRASES, REFERENCES, ResultRow, timed_generate, write_manifest

sys.path.insert(0, str(DEFAULT_OUTPUT_ROOT.parents[1]))

from pipeline import VoxCPMPipeline


ENGINE = "VoxCPM2-2.0.3"


def main() -> int:
    os.environ.setdefault("HF_HOME", r"I:\hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", r"I:\hf_cache")
    out_dir = DEFAULT_OUTPUT_ROOT / ENGINE
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = VoxCPMPipeline(device="cuda")

    rows: list[ResultRow] = []
    for ref in REFERENCES:
        for phrase_id, text in PHRASES:
            for take in range(1, 3):
                output_path = out_dir / f"{ref['voice_id']}__{phrase_id}__take{take:02d}.wav"

                def call() -> None:
                    pipeline.synthesize_segment(
                        text=text,
                        output_path=output_path,
                        ref_audio=ref["ref_audio"],
                        ref_text=ref["ref_text"],
                        cfg_value=2.0,
                        inference_timesteps=10,
                    )

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
    pipeline.unload()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
