from __future__ import annotations

import csv
import html
import json
import shutil
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "bench_outputs" / "cross_lingual_clone_20260612"
REF_MANIFEST = ROOT / "bench_outputs" / "voice_library_refs_20260604" / "reference_manifest.json"

REFERENCES = []
if REF_MANIFEST.exists():
    for index, row in enumerate(json.loads(REF_MANIFEST.read_text(encoding="utf-8")), start=1):
        REFERENCES.append(
            {
                "ref_index": index,
                "voice_id": str(row["voice_id"]),
                "speaker": str(row["speaker"]),
                "ref_audio": str(row["file"]),
                "ref_text": str(row.get("ref_text") or ""),
            }
        )

PHRASES = [
    ("en_hello", "Hello, this is a cross language voice cloning test."),
    ("en_weather", "The weather is nice today, and I feel very happy."),
    ("en_short", "Okay, let's try it again."),
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
        for row in rows_list:
            writer.writerow(asdict(row))


def prepare_ascii_refs(out_dir: Path) -> list[dict]:
    ref_dir = out_dir / "_ascii_refs"
    ref_dir.mkdir(parents=True, exist_ok=True)
    refs = []
    for ref in REFERENCES:
        src = Path(ref["ref_audio"])
        dst = ref_dir / f"ref{ref['ref_index']:02d}{src.suffix}"
        if not dst.exists():
            shutil.copyfile(src, dst)
        refs.append({**ref, "ref_audio_ascii": str(dst)})
    return refs


def copy_reference_audio(out_root: Path) -> list[dict]:
    ref_dir = out_root / "reference_audio"
    ref_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for ref in REFERENCES:
        src = Path(ref["ref_audio"])
        dst = ref_dir / f"ref{ref['ref_index']:02d}_reference{src.suffix}"
        if not dst.exists():
            shutil.copyfile(src, dst)
        copied.append({**ref, "file": str(dst.relative_to(out_root)).replace("\\", "/")})
    return copied


def build_listen_page(out_root: Path = OUTPUT_ROOT) -> Path:
    references = copy_reference_audio(out_root)
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
        "<title>Cross-Lingual Voice Clone Benchmark</title>",
        "<style>body{font-family:Segoe UI,Microsoft YaHei,sans-serif;margin:24px;background:#f5f7fb;color:#172033}"
        ".card{background:white;border:1px solid #dbe3f0;border-radius:14px;padding:16px;margin:14px 0}"
        "table{border-collapse:collapse;width:100%;background:white}td,th{border-bottom:1px solid #edf1f7;padding:8px;text-align:left}"
        "audio{width:260px}.bad{color:#a33}.muted{color:#667085}</style>",
        "<h1>Cross-Lingual Voice Clone Benchmark</h1>",
        "<div class='card'><h2>Chinese Reference Voices</h2>",
    ]
    for ref in references:
        parts.append(
            f"<p><b>{esc(ref['speaker'])}</b> <span class='muted'>{esc(ref['voice_id'])}</span> "
            f"| {esc(ref['ref_text'])}<br><audio controls src='{esc(ref['file'])}'></audio></p>"
        )
    parts.append("</div>")
    parts.append("<div class='card'><h2>Summary</h2><table><tr><th>Engine</th><th>OK</th><th>Total</th><th>Total Sec</th></tr>")
    for engine, item in sorted(summary.items()):
        parts.append(f"<tr><td>{esc(engine)}</td><td>{item['ok']}</td><td>{item['total']}</td><td>{item['seconds']:.2f}</td></tr>")
    parts.append("</table></div>")
    parts.append("<div class='card'><h2>Samples</h2><table><tr><th>Engine</th><th>Reference</th><th>English Text</th><th>Audio</th><th>Seconds</th><th>RTF</th><th>Error</th></tr>")
    for row in sorted(rows, key=lambda r: (r["engine"], r["voice_id"], r["phrase_id"], r["take"])):
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
    out_root.mkdir(parents=True, exist_ok=True)
    page = out_root / "listen.html"
    page.write_text("\n".join(parts), encoding="utf-8")
    return page
