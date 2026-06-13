from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import pipeline

from common_japanese_to_chinese import OUTPUT_ROOT, REF_MANIFEST, discover_japanese_samples


MODEL_PATH = Path(r"I:\hf_cache\models--openai--whisper-large-v3-turbo\snapshots\41f01f3fe87f28c78e2fbf8b568835947dd65ed9")


def main() -> int:
    rows = discover_japanese_samples()
    if not rows:
        raise RuntimeError("No Japanese reference samples found in I:\\code\\aitts\\voice")
    asr = pipeline(
        "automatic-speech-recognition",
        model=str(MODEL_PATH),
        dtype=torch.float16,
        device_map="cuda:0",
    )
    for row in rows:
        result = asr(row["file"], generate_kwargs={"language": "ja", "task": "transcribe"})
        row["ref_text"] = str(result.get("text", "")).strip()
        print(row["speaker"], row["ref_text"], flush=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REF_MANIFEST.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(REF_MANIFEST)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
