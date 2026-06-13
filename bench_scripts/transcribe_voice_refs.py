from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import pipeline

OUT_ROOT = Path(r"I:\code\aitts\omnivoice-reader\bench_outputs\voice_library_refs_20260604")
MODEL_PATH = Path(r"I:\hf_cache\models--openai--whisper-large-v3-turbo\snapshots\41f01f3fe87f28c78e2fbf8b568835947dd65ed9")


def main() -> int:
    manifest_path = OUT_ROOT / "reference_manifest.json"
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    asr = pipeline(
        "automatic-speech-recognition",
        model=str(MODEL_PATH),
        dtype=torch.float16,
        device_map="cuda:0",
    )
    for row in rows:
        result = asr(row["file"], generate_kwargs={"language": "zh", "task": "transcribe"})
        row["ref_text"] = str(result.get("text", "")).strip()
        print(row["speaker"], row["ref_text"], flush=True)
    manifest_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
