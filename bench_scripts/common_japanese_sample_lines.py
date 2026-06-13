from __future__ import annotations

import csv
import html
import json
import random
import re
import shutil
import subprocess
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / "docs" / "samples"
VOICE_DIR = Path(r"I:\code\aitts\voice")
BASE_REF_ROOT = ROOT / "bench_outputs" / "japanese_ref_to_chinese_20260612"
BASE_REF_MANIFEST = BASE_REF_ROOT / "reference_manifest.json"
OUTPUT_ROOT = ROOT / "bench_outputs" / "japanese_ref_to_chinese_sample_lines_20260612"
REF_DIR = OUTPUT_ROOT / "reference_audio"
REF_MANIFEST = OUTPUT_ROOT / "reference_manifest.json"
TARGET_MANIFEST = OUTPUT_ROOT / "target_texts.json"
DEFAULT_RECENT_ENGINE_COUNT = 8

UTTERANCE_MARKERS = "啊呀呢吧哦嗯哼哈唉哎嘛啦吗哟呃喔耶诶欸啦哇嘛"


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
    return value or "item"


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


def load_references() -> list[dict]:
    if REF_MANIFEST.exists():
        return json.loads(REF_MANIFEST.read_text(encoding="utf-8"))
    if BASE_REF_MANIFEST.exists():
        base_refs = json.loads(BASE_REF_MANIFEST.read_text(encoding="utf-8"))
    else:
        base_refs = []
        for src in sorted(VOICE_DIR.glob("*")):
            if not src.is_file() or src.suffix.lower() not in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}:
                continue
            if not re.search(r"日语|日本|ja|jp|jpn|japanese|日本語", src.name, re.IGNORECASE):
                continue
            voice_id = f"jp{len(base_refs) + 1:02d}_{safe_id(src.stem)}"
            base_refs.append({"voice_id": voice_id, "speaker": src.stem, "source_file": str(src), "ref_text": ""})

    refs = []
    for index, ref in enumerate(base_refs, start=1):
        source = Path(ref.get("source_file") or ref.get("file", ""))
        if source.exists():
            wav_path = REF_DIR / f"ref{index:02d}_{safe_id(ref['speaker'])}.wav"
            convert_to_wav(source, wav_path)
        else:
            wav_path = Path(ref["file"])
        refs.append({**ref, "file": str(wav_path), "ref_index": index})
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


def _clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>|&[#A-Za-z0-9]+;", "", text)
    text = re.sub(r"^\s*[\[\(（]?\d+[\]\)）]?\s*", "", text)
    text = re.sub(r"^[\s\w·\u4e00-\u9fff]{1,12}[：:]\s*", "", text)
    text = text.strip().strip("「」『』“”\"'")
    return re.sub(r"\s+", "", text)


def _score_candidate(text: str) -> int:
    score = 0
    if 10 <= len(text) <= 34:
        score += 5
    if any(mark in text for mark in "！？……──～"):
        score += 3
    score += min(5, sum(text.count(ch) for ch in UTTERANCE_MARKERS))
    if len(text) < 6 or len(text) > 48:
        score -= 10
    return score


def extract_target_texts(count: int = 6) -> list[dict]:
    if TARGET_MANIFEST.exists():
        return json.loads(TARGET_MANIFEST.read_text(encoding="utf-8"))

    candidates: list[dict] = []
    for path in sorted(SAMPLES_DIR.glob("*_groundtruth.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not data.get("source", {}).get("human_reviewed", False):
            continue
        for seg in data.get("segments", []):
            text = _clean_text(str(seg.get("text", "")))
            if not re.search(r"[\u4e00-\u9fff]", text):
                continue
            if not any(marker in text for marker in UTTERANCE_MARKERS):
                continue
            score = _score_candidate(text)
            if score <= 0:
                continue
            candidates.append(
                {
                    "text": text,
                    "speaker": seg.get("speaker", ""),
                    "source": path.name,
                    "segment_index": seg.get("i"),
                    "score": score,
                }
            )

    seen = set()
    unique = []
    for item in sorted(candidates, key=lambda x: (-x["score"], x["source"], x["segment_index"])):
        if item["text"] in seen:
            continue
        seen.add(item["text"])
        unique.append(item)

    top_pool = unique[:80] if len(unique) > 80 else unique
    rng = random.Random(20260612)
    selected = rng.sample(top_pool, k=min(count, len(top_pool)))
    selected = sorted(selected, key=lambda x: (x["source"], x["segment_index"] if x["segment_index"] is not None else 99999))
    rows = []
    for index, item in enumerate(selected, start=1):
        rows.append({"phrase_id": f"sample{index:02d}", **item})
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    TARGET_MANIFEST.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def build_listen_page(out_root: Path = OUTPUT_ROOT) -> Path:
    refs = load_references()
    targets = extract_target_texts()
    rows = []
    engine_mtimes = {}
    for manifest in sorted(out_root.glob("*/manifest.json")):
        try:
            manifest_rows = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows.extend(manifest_rows)
        mtime = manifest.stat().st_mtime
        for row in manifest_rows:
            engine = row.get("engine")
            if engine:
                engine_mtimes[engine] = max(engine_mtimes.get(engine, 0.0), mtime)

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
        "<title>Japanese Reference to Chinese Sample Lines</title>",
        "<style>body{font-family:Segoe UI,Microsoft YaHei,sans-serif;margin:24px;background:#f7f1e8;color:#231a12}"
        ".card{background:#fffaf3;border:1px solid #e8d8c3;border-radius:14px;padding:16px;margin:14px 0}"
        "table{border-collapse:collapse;width:100%;background:#fff}td,th{border-bottom:1px solid #efe2d1;padding:8px;text-align:left;vertical-align:top}"
        "audio{width:260px}.bad{color:#a33}.muted{color:#75685c}.text{max-width:360px}"
        ".filters{display:flex;gap:12px;flex-wrap:wrap;align-items:end}.filters label{display:grid;gap:4px;font-size:13px;color:#75685c}"
        ".engine-list{display:flex;gap:8px;flex-wrap:wrap;max-width:980px}.engine-list label{display:flex;gap:6px;align-items:center;border:1px solid #e2cfb8;border-radius:999px;background:#fff;padding:6px 10px;color:#231a12}"
        ".engine-list input{accent-color:#8b5e34}select,button{border:1px solid #d7c4aa;border-radius:10px;background:#fffaf3;color:#231a12;padding:8px 10px}"
        "button{cursor:pointer}.pill{display:inline-block;border-radius:999px;background:#f0dfc8;padding:4px 10px;color:#5d4c3b}</style>",
        "<h1>Japanese Reference -> Chinese Sample Lines</h1>",
        "<div class='card'><h2>Selected Chinese Lines</h2><table><tr><th>ID</th><th>Text</th><th>Speaker</th><th>Source</th></tr>",
    ]
    for target in targets:
        parts.append(
            f"<tr><td>{esc(target['phrase_id'])}</td><td class='text'>{esc(target['text'])}</td>"
            f"<td>{esc(target.get('speaker',''))}</td><td>{esc(target.get('source',''))}#{esc(target.get('segment_index',''))}</td></tr>"
        )
    parts.append("</table></div>")
    parts.append("<div class='card'><h2>Japanese References</h2>")
    for ref in refs:
        rel = Path(ref["file"]).resolve().relative_to(out_root.resolve()).as_posix()
        parts.append(
            f"<p><b>{esc(ref['speaker'])}</b> <span class='muted'>{esc(ref['voice_id'])}</span><br>"
            f"ref_text: {esc(ref.get('ref_text',''))}<br><audio controls src='{esc(rel)}'></audio></p>"
        )
    parts.append("</div>")
    engines = sorted({row["engine"] for row in rows})
    recent_engines = {
        engine
        for engine, _ in sorted(engine_mtimes.items(), key=lambda item: (-item[1], item[0]))[:DEFAULT_RECENT_ENGINE_COUNT]
    }
    parts.append(
        "<div class='card'><h2>Summary</h2>"
        "<div class='filters'>"
        f"<label>Engines <span class='muted'>(default: latest {DEFAULT_RECENT_ENGINE_COUNT}; double-click to invert all)</span><div class='engine-list' id='engineFilters'>"
    )
    for engine in engines:
        checked = " checked" if engine in recent_engines else ""
        recent = " data-recent='true'" if engine in recent_engines else " data-recent='false'"
        parts.append(f"<label><input type='checkbox' class='engine-checkbox' value='{esc(engine)}'{checked}{recent}>{esc(engine)}</label>")
    parts.append("</div></label><label>Line<select id='phraseFilter'><option value=''>All lines</option>")
    for target in targets:
        label = f"{target['phrase_id']} - {target['text']}"
        parts.append(f"<option value='{esc(target['phrase_id'])}'>{esc(label)}</option>")
    parts.append(
        "</select></label><button type='button' id='resetFilters'>Reset</button>"
        "<button type='button' id='selectRecentEngines'>Select recent</button>"
        "<button type='button' id='selectAllEngines'>Select all</button>"
        "<button type='button' id='clearEngines'>Clear engines</button>"
        "<span class='pill' id='visibleCount'></span></div>"
        "<table><thead><tr><th>Engine</th><th>OK</th><th>Total</th><th>Total Sec</th></tr></thead><tbody id='summaryBody'>"
    )
    for engine, item in sorted(summary.items()):
        parts.append(f"<tr class='summary-row' data-engine='{esc(engine)}'><td>{esc(engine)}</td><td>{item['ok']}</td><td>{item['total']}</td><td>{item['seconds']:.2f}</td></tr>")
    parts.append("</tbody></table></div>")
    parts.append("<div class='card'><h2>Samples</h2><table><tr><th>Engine</th><th>Take</th><th>Japanese Ref</th><th>Chinese Text</th><th>Audio</th><th>Seconds</th><th>RTF</th><th>Error</th></tr>")
    for row in sorted(rows, key=lambda r: (r["engine"], r["voice_id"], r["phrase_id"], int(r.get("take", 1)))):
        audio = ""
        if row.get("ok") and row.get("file"):
            rel = Path(row["file"]).resolve().relative_to(out_root.resolve()).as_posix()
            audio = f"<audio controls src='{esc(rel)}'></audio>"
        err = f"<span class='bad'>{esc(row.get('error',''))}</span>" if row.get("error") else ""
        parts.append(
            f"<tr class='sample-row' data-engine='{esc(row['engine'])}' data-phrase='{esc(row['phrase_id'])}' data-ok='{str(bool(row.get('ok'))).lower()}' data-seconds='{float(row['seconds']):.6f}'><td>{esc(row['engine'])}</td><td>{esc(row.get('take', 1))}</td><td>{esc(row['speaker'])}</td><td class='text'>{esc(row['text'])}</td>"
            f"<td>{audio}</td><td>{float(row['seconds']):.2f}</td><td>{float(row['rtf']):.2f}</td><td>{err}</td></tr>"
        )
    parts.append("</table></div>")
    parts.append(
        "<script>"
        "const engineCheckboxes=Array.from(document.querySelectorAll('.engine-checkbox'));"
        "const engineFilters=document.getElementById('engineFilters');"
        "const phraseFilter=document.getElementById('phraseFilter');"
        "const visibleCount=document.getElementById('visibleCount');"
        "const summaryBody=document.getElementById('summaryBody');"
        "function htmlEscape(value){return String(value).replace(/[&<>'\"]/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;',\"'\":'&#39;','\"':'&quot;'}[ch]));}"
        "function applyFilters(){"
        "const selected=new Set(engineCheckboxes.filter(box=>box.checked).map(box=>box.value));const phrase=phraseFilter.value;let shown=0,total=0;const stats=new Map();"
        "document.querySelectorAll('.sample-row').forEach(row=>{total++;const ok=selected.has(row.dataset.engine)&&(!phrase||row.dataset.phrase===phrase);row.style.display=ok?'':'none';if(!ok)return;shown++;const name=row.dataset.engine;const item=stats.get(name)||{ok:0,total:0,seconds:0};item.total++;if(row.dataset.ok==='true')item.ok++;item.seconds+=Number(row.dataset.seconds||0);stats.set(name,item);});"
        "summaryBody.innerHTML=Array.from(stats.entries()).sort((a,b)=>a[0].localeCompare(b[0])).map(([name,item])=>`<tr class='summary-row' data-engine='${htmlEscape(name)}'><td>${htmlEscape(name)}</td><td>${item.ok}</td><td>${item.total}</td><td>${item.seconds.toFixed(2)}</td></tr>`).join('');"
        "visibleCount.textContent=`Showing ${shown} / ${total} samples`;"
        "}"
        "engineCheckboxes.forEach(box=>box.addEventListener('change',applyFilters));phraseFilter.addEventListener('change',applyFilters);"
        "engineFilters.addEventListener('dblclick',event=>{event.preventDefault();engineCheckboxes.forEach(box=>box.checked=!box.checked);applyFilters();});"
        "function selectRecent(){engineCheckboxes.forEach(box=>box.checked=box.dataset.recent==='true');}"
        "document.getElementById('selectRecentEngines').addEventListener('click',()=>{selectRecent();applyFilters();});"
        "document.getElementById('selectAllEngines').addEventListener('click',()=>{engineCheckboxes.forEach(box=>box.checked=true);applyFilters();});"
        "document.getElementById('clearEngines').addEventListener('click',()=>{engineCheckboxes.forEach(box=>box.checked=false);applyFilters();});"
        "document.getElementById('resetFilters').addEventListener('click',()=>{selectRecent();phraseFilter.value='';applyFilters();});"
        "applyFilters();"
        "</script>"
    )
    page = out_root / "listen.html"
    page.write_text("\n".join(parts), encoding="utf-8")
    return page
