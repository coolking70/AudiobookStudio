from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import soundfile as sf
import torch

from common_voice_library import (
    DEFAULT_OUTPUT_ROOT,
    PHRASES,
    REFERENCES,
    ResultRow,
    timed_generate,
    write_manifest,
)


ENGINE = "OmniVoice-updated"
OMNIVOICE_SRC = Path(r"I:\code\aitts\OmniVoice")
MODEL_PATH = Path(
    r"I:\hf_cache\models--k2-fsa--OmniVoice\snapshots\29cde0ee295ee673d33e9ab570e7bbbe761c33b3"
)


def prepare_environment() -> None:
    os.environ.setdefault("HF_HOME", r"I:\hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", r"I:\hf_cache")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    sys.path.insert(0, str(OMNIVOICE_SRC))


def prepare_ascii_refs(out_dir: Path) -> list[dict]:
    ref_dir = out_dir / "_ascii_refs"
    ref_dir.mkdir(parents=True, exist_ok=True)
    ascii_refs = []
    for idx, ref in enumerate(REFERENCES, start=1):
        src = Path(ref["ref_audio"])
        dst = ref_dir / f"ref{idx:02d}{src.suffix}"
        if not dst.exists():
            shutil.copyfile(src, dst)
        ascii_refs.append({**ref, "ref_audio": str(dst), "ref_index": idx})
    return ascii_refs


def main() -> None:
    prepare_environment()

    from omnivoice.models.omnivoice import OmniVoice

    out_dir = DEFAULT_OUTPUT_ROOT / ENGINE
    out_dir.mkdir(parents=True, exist_ok=True)
    refs = prepare_ascii_refs(out_dir)

    print(f"Loading {ENGINE} from {MODEL_PATH}", flush=True)
    model = OmniVoice.from_pretrained(str(MODEL_PATH), device_map="cuda:0", dtype=torch.float16)
    print(f"Loaded {ENGINE}, sampling_rate={model.sampling_rate}", flush=True)

    rows: list[ResultRow] = []
    for ref in refs:
        for phrase_id, text in PHRASES:
            for take in (1, 2):
                filename = f"ref{ref['ref_index']:02d}__{phrase_id}__take{take:02d}.wav"
                output_path = out_dir / filename

                def call() -> None:
                    audios = model.generate(
                        text=text,
                        language="Chinese",
                        ref_audio=ref["ref_audio"],
                        ref_text=ref["ref_text"],
                        num_step=32,
                    )
                    sf.write(str(output_path), audios[0], model.sampling_rate)

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
                        file=str(output_path),
                        seconds=seconds,
                        duration_seconds=duration,
                        rtf=rtf,
                        error=error,
                    )
                )
                write_manifest(rows, out_dir)
                status = "ok" if ok else "failed"
                print(
                    f"{status}: {ref['speaker']} {phrase_id} take{take} "
                    f"{seconds:.2f}s rtf={rtf:.2f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
