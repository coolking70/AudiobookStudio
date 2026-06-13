from __future__ import annotations

import csv
import html
import json
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "bench_outputs" / "voice_library_phrases_from_voice_dir_20260604"

_REF_MANIFEST = ROOT / "bench_outputs" / "voice_library_refs_20260604" / "reference_manifest.json"
if _REF_MANIFEST.exists():
    _ref_rows = json.loads(_REF_MANIFEST.read_text(encoding="utf-8"))
    REFERENCES = [
        {
            "voice_id": str(row["voice_id"]),
            "speaker": str(row["speaker"]),
            "ref_audio": str(row["file"]),
            "ref_text": str(row.get("ref_text") or ""),
        }
        for row in _ref_rows
    ]
else:
    REFERENCES = []

PHRASES = [
    ("a", "啊。"),
    ("en", "嗯。"),
    ("haha", "哈哈。"),
    ("hehe", "呵呵。"),
    ("aiya", "哎呀。"),
    ("eh_question", "欸？"),
    ("really", "真的吗？"),
    ("okay", "好吧。"),
]

EXCLUDED_LISTEN_ENGINES = {"Chatterbox-0.1.7"}


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

        info = sf.info(str(path))
        return float(info.duration or 0.0)
    except Exception:
        pass
    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate) if rate else 0.0
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
        seconds = time.perf_counter() - start
        return False, seconds, 0.0, 0.0, f"{exc.__class__.__name__}: {exc}"


def write_manifest(rows: Iterable[ResultRow], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows)
    (out_dir / "manifest.json").write_text(
        json.dumps([asdict(row) for row in rows_list], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (out_dir / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows_list[0]).keys()) if rows_list else [])
        if rows_list:
            writer.writeheader()
            for row in rows_list:
                writer.writerow(asdict(row))


def copy_reference_audio(out_root: Path) -> list[dict]:
    ref_dir = out_root / "reference_audio"
    ref_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for ref in REFERENCES:
        src = Path(ref["ref_audio"])
        dst = ref_dir / f"{ref['voice_id']}_reference{src.suffix}"
        if not dst.exists():
            dst.write_bytes(src.read_bytes())
        copied.append({**ref, "file": str(dst.relative_to(out_root)).replace("\\", "/")})
    (out_root / "reference_manifest.json").write_text(
        json.dumps(copied, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return copied


def build_listen_page(out_root: Path) -> None:
    references = copy_reference_audio(out_root)
    rows = []
    for manifest in sorted(out_root.glob("*/manifest.json")):
        try:
            manifest_rows = json.loads(manifest.read_text(encoding="utf-8"))
            rows.extend(
                row
                for row in manifest_rows
                if row.get("engine") not in EXCLUDED_LISTEN_ENGINES
            )
        except Exception:
            continue

    summary = {}
    for row in rows:
        item = summary.setdefault(row["engine"], {"ok": 0, "total": 0, "seconds": 0.0, "rtf": []})
        item["total"] += 1
        if row["ok"]:
            item["ok"] += 1
            item["seconds"] += float(row["seconds"])
            if row["rtf"]:
                item["rtf"].append(float(row["rtf"]))

    def esc(value: object) -> str:
        return html.escape(str(value), quote=True)

    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Voice Library Phrase Benchmark</title>",
        "<style>body{font-family:Segoe UI,Microsoft YaHei,sans-serif;margin:24px;background:#f7f4ee;color:#1e1b18}"
        "h1{font-size:28px}.card{background:#fff;border:1px solid #e5ded3;border-radius:14px;padding:16px;margin:14px 0}"
        "table{border-collapse:collapse;width:100%;background:#fff}td,th{border-bottom:1px solid #eee;padding:8px;text-align:left}"
        "audio{width:240px}.bad{color:#a33}.muted{color:#766}</style>",
        "<h1>Voice Library Phrase Benchmark</h1>",
        "<div class='card'><h2>Reference Voices</h2>",
    ]
    for ref in references:
        parts.append(
            f"<p><b>{esc(ref['speaker'])}</b> <span class='muted'>{esc(ref['voice_id'])}</span> "
            f"| {esc(ref['ref_text'])}<br><audio controls src='{esc(ref['file'])}'></audio></p>"
        )
    parts.append("</div>")
    parts.append("<div class='card'><h2>Summary</h2><table><tr><th>Engine</th><th>OK</th><th>Total</th><th>Total Sec</th><th>Avg RTF</th></tr>")
    for engine, item in sorted(summary.items()):
        avg_rtf = sum(item["rtf"]) / len(item["rtf"]) if item["rtf"] else 0
        parts.append(
            f"<tr><td>{esc(engine)}</td><td>{item['ok']}</td><td>{item['total']}</td>"
            f"<td>{item['seconds']:.2f}</td><td>{avg_rtf:.2f}</td></tr>"
        )
    parts.append("</table></div>")
    parts.append("<div class='card'><h2>Samples</h2><table><tr><th>Engine</th><th>Voice</th><th>Phrase</th><th>Take</th><th>Audio</th><th>Seconds</th><th>RTF</th><th>Error</th></tr>")
    for row in sorted(rows, key=lambda r: (r["engine"], r["voice_id"], r["phrase_id"], r["take"])):
        audio = ""
        if row["ok"] and row["file"]:
            rel = Path(row["file"]).resolve().relative_to(out_root.resolve()).as_posix()
            audio = f"<audio controls src='{esc(rel)}'></audio>"
        err = f"<span class='bad'>{esc(row['error'])}</span>" if row.get("error") else ""
        parts.append(
            f"<tr><td>{esc(row['engine'])}</td><td>{esc(row['speaker'])}</td><td>{esc(row['text'])}</td>"
            f"<td>{row['take']}</td><td>{audio}</td><td>{float(row['seconds']):.2f}</td>"
            f"<td>{float(row['rtf']):.2f}</td><td>{err}</td></tr>"
        )
    parts.append("</table></div>")
    (out_root / "listen.html").write_text("\n".join(parts), encoding="utf-8")
