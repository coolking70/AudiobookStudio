from __future__ import annotations

import json
import subprocess
from pathlib import Path

VOICE_ROOT = Path(r"I:\code\aitts\voice")
OUT_ROOT = Path(r"I:\code\aitts\omnivoice-reader\bench_outputs\voice_library_refs_20260604")

CANDIDATES = ["余音", "茜茜", "十手卫", "闲云"]


def ffmpeg_convert(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            "24000",
            str(dst),
        ],
        check=True,
    )


def main() -> int:
    rows = []
    for name in CANDIDATES:
        src = VOICE_ROOT / f"{name}.mp3"
        dst = OUT_ROOT / "reference_audio" / f"{name}.wav"
        ffmpeg_convert(src, dst)
        rows.append(
            {
                "voice_id": name,
                "speaker": name,
                "source_file": str(src),
                "file": str(dst),
                "ref_text": "",
            }
        )
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "reference_manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    html = [
        "<!doctype html><meta charset='utf-8'><title>Voice Reference Candidates</title>",
        "<style>body{font-family:Segoe UI,Microsoft YaHei,sans-serif;margin:24px;background:#f7f4ee}"
        ".card{background:#fff;border:1px solid #ddd;border-radius:12px;padding:14px;margin:12px 0}"
        "audio{width:360px}</style>",
        "<h1>Voice Reference Candidates</h1>",
    ]
    for row in rows:
        rel = Path(row["file"]).relative_to(OUT_ROOT).as_posix()
        html.append(f"<div class='card'><h2>{row['speaker']}</h2><p>{row['source_file']}</p><audio controls src='{rel}'></audio></div>")
    (OUT_ROOT / "listen_refs.html").write_text("\n".join(html), encoding="utf-8")
    print(OUT_ROOT / "listen_refs.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
