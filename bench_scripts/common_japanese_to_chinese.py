from __future__ import annotations

import csv
import html
import json
import re
import shutil
import subprocess
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
VOICE_DIR = Path(r"I:\code\aitts\voice")
OUTPUT_ROOT = ROOT / "bench_outputs" / "japanese_ref_to_chinese_20260612"
REF_DIR = OUTPUT_ROOT / "reference_audio"
REF_MANIFEST = OUTPUT_ROOT / "reference_manifest.json"

TARGET_TEXTS = [
    ("zh_short", "你好，很高兴见到你。"),
    ("zh_emotion", "真的吗？这也太让人惊讶了吧！"),
    ("zh_narration", "今天天气很好，我们一起去公园散步吧。"),
]


@dataclass
class ResultRow:
    engine: str
    voice_id: str
    speaker: str
    phrase_id: str
    text: str
    take: int
    ok: bool
    file: str
    seconds: float
    duration_seconds: float
    rtf: float
    error: str = ""


def safe_id(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return value or "voice"


def wav_duration(path: Path) -> float:
    try:
        import soundfile as sf

        return float(sf.info(str(path)).duration or 0.0)
    except Exception:
        pass
    try:
        with wave.open(str(path), "rb") as wf:
            rate = wf.getframerate()
            return wf.getnframes() / float(rate) if rate else 0.0
    except Exception:
        return 0.0


def timed_generate(call, output_path: Path) -> tuple[bool, float, float, float, str]:
    start = time.perf_counter()
    try:
        call()
        seconds = time.perf_counter() - start
        duration = wav_duration(output_path)
        rtf = seconds / duration if duration > 0 else 0.0
        return True, seconds, duration, rtf, ""
    except Exception as exc:
        return False, time.perf_counter() - start, 0.0, 0.0, f"{exc.__class__.__name__}: {exc}"


def write_manifest(rows: Iterable[ResultRow], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows)
    (out_dir / "manifest.json").write_text(
        json.dumps([asdict(row) for row in rows_list], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not rows_list:
        return
    with (out_dir / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows_list[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows_list)


def convert_to_wav(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [ffmpeg, "-y", "-i", str(src), "-ac", "1", "-ar", "24000", str(dst)],
        check=True,
        capture_output=True,
    )


def discover_japanese_samples() -> list[dict]:
    rows = []
    for index, src in enumerate(sorted(VOICE_DIR.glob("*")), start=1):
        if not src.is_file() or src.suffix.lower() not in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}:
            continue
        if not re.search(r"日语|日本|ja|jp|jpn|japanese|日本語", src.name, re.IGNORECASE):
            continue
        speaker = src.stem
        voice_id = f"jp{len(rows) + 1:02d}_{safe_id(speaker)}"
        wav_path = REF_DIR / f"{voice_id}.wav"
        convert_to_wav(src, wav_path)
        rows.append(
            {
                "voice_id": voice_id,
                "speaker": speaker,
                "source_file": str(src),
                "file": str(wav_path),
                "ref_text": "",
            }
        )
    return rows


def load_references() -> list[dict]:
    if REF_MANIFEST.exists():
        return json.loads(REF_MANIFEST.read_text(encoding="utf-8"))
    refs = discover_japanese_samples()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REF_MANIFEST.write_text(json.dumps(refs, ensure_ascii=False, indent=2), encoding="utf-8")
    return refs


def prepare_ascii_refs(out_dir: Path) -> list[dict]:
    ref_dir = out_dir / "_ascii_refs"
    ref_dir.mkdir(parents=True, exist_ok=True)
    refs = []
    for index, ref in enumerate(load_references(), start=1):
        src = Path(ref["file"])
        dst = ref_dir / f"ref{index:02d}.wav"
        if not dst.exists():
            shutil.copyfile(src, dst)
        refs.append({**ref, "ref_index": index, "ref_audio_ascii": str(dst)})
    return refs


def build_listen_page(out_root: Path = OUTPUT_ROOT) -> Path:
    refs = load_references()
    rows = []
    for manifest in sorted(out_root.glob("*/manifest.json")):
        try:
            rows.extend(json.loads(manifest.read_text(encoding="utf-8")))
        except Exception:
            continue

    summary = {}
    for row in rows:
        item = summary.setdefault(row["engine"], {"ok": 0, "total": 0, "seconds": 0.0})
        item["total"] += 1
        if row["ok"]:
            item["ok"] += 1
            item["seconds"] += float(row["seconds"])

    def esc(value: object) -> str:
        return html.escape(str(value), quote=True)

    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Japanese Reference to Chinese Speech</title>",
        "<style>body{font-family:Segoe UI,Microsoft YaHei,sans-serif;margin:24px;background:#fbf7f2;color:#201a15}"
        ".card{background:#fff;border:1px solid #eadfd2;border-radius:14px;padding:16px;margin:14px 0}"
        "table{border-collapse:collapse;width:100%;background:#fff}td,th{border-bottom:1px solid #f0e6dc;padding:8px;text-align:left}"
        "audio{width:260px}.bad{color:#a33}.muted{color:#75685c}</style>",
        "<h1>Japanese Reference -> Chinese Speech</h1>",
        "<div class='card'><h2>Japanese References</h2>",
    ]
    for ref in refs:
        rel = Path(ref["file"]).resolve().relative_to(out_root.resolve()).as_posix()
        parts.append(
            f"<p><b>{esc(ref['speaker'])}</b> <span class='muted'>{esc(ref['voice_id'])}</span><br>"
            f"ref_text: {esc(ref.get('ref_text',''))}<br><audio controls src='{esc(rel)}'></audio></p>"
        )
    parts.append("</div>")
    parts.append("<div class='card'><h2>Summary</h2><table><tr><th>Engine</th><th>OK</th><th>Total</th><th>Total Sec</th></tr>")
    for engine, item in sorted(summary.items()):
        parts.append(f"<tr><td>{esc(engine)}</td><td>{item['ok']}</td><td>{item['total']}</td><td>{item['seconds']:.2f}</td></tr>")
    parts.append("</table></div>")
    parts.append("<div class='card'><h2>Samples</h2><table><tr><th>Engine</th><th>Japanese Ref</th><th>Chinese Text</th><th>Audio</th><th>Seconds</th><th>RTF</th><th>Error</th></tr>")
    for row in sorted(rows, key=lambda r: (r["engine"], r["voice_id"], r["phrase_id"])):
        audio = ""
        if row.get("ok") and row.get("file"):
            rel = Path(row["file"]).resolve().relative_to(out_root.resolve()).as_posix()
            audio = f"<audio controls src='{esc(rel)}'></audio>"
        err = f"<span class='bad'>{esc(row.get('error',''))}</span>" if row.get("error") else ""
        parts.append(
            f"<tr><td>{esc(row['engine'])}</td><td>{esc(row['speaker'])}</td><td>{esc(row['text'])}</td>"
            f"<td>{audio}</td><td>{float(row['seconds']):.2f}</td><td>{float(row['rtf']):.2f}</td><td>{err}</td></tr>"
        )
    parts.append("</table></div>")
    page = out_root / "listen.html"
    page.write_text("\n".join(parts), encoding="utf-8")
    return page
