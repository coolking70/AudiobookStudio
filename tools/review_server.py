"""Manual speaker-review tool (local web app, stdlib only).

Center pane shows the full original text; left column lists every marked quote with an
editable speaker field. Every edit is saved to disk immediately.

Usage:
    python tools/review_server.py            # uses the muli4_seg2 sample by default
    python tools/review_server.py --parse docs/samples/X_parse.json --raw docs/samples/X.txt --out docs/samples/X_review.json
Then open http://127.0.0.1:8765/
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SAMP = REPO / "docs/samples"
sys.path.insert(0, str(REPO / "tools"))

DEFAULT_PARSE = SAMP / "muli4_seg2_parse.json"
DEFAULT_RAW = SAMP / "muli4_seg2_sample.txt"
DEFAULT_OUT = SAMP / "muli4_seg2_review.json"
# Extra always-flag indices (sample-specific overrides). Empty by default; the align()
# heuristic auto-flags low-confidence / block-review-changed / uncertain-speaker segments.
FLAGGED: set[int] = set()


def load_roster() -> list[str]:
    try:
        from evaluate_agnes_bookmark_review import ROLE_HINTS
        return list(ROLE_HINTS.keys())
    except Exception:
        return []


_MATCH_STRIP = set("「」『』") | set(" \t\r\n　")


def _norm_key(t: str) -> str:
    # ignore whitespace and quote-bracket VARIANTS (「」/『』) so nested-quote and
    # bracket-normalized segments (e.g. 「会」 vs 『会』) still locate in the raw text
    return "".join(ch for ch in str(t or "") if ch not in _MATCH_STRIP)


def align(raw: str, segments: list[dict]) -> list[dict]:
    """Attach raw-text [start,end) offsets to each segment, robust to whitespace and
    quote-bracket-variant differences (maps a stripped match back to real offsets)."""
    # build a stripped index of the raw text -> original char positions
    sidx: list[int] = []
    schars: list[str] = []
    for k, ch in enumerate(raw):
        if ch in _MATCH_STRIP:
            continue
        schars.append(ch)
        sidx.append(k)
    sraw = "".join(schars)
    scursor = 0
    out = []
    for i, s in enumerate(segments):
        key = _norm_key(s.get("text", ""))
        start = end = -1
        if key:
            p = sraw.find(key, scursor)
            if p < 0:
                p = sraw.find(key)  # fall back to a global search if cursor overshot
            if p >= 0:
                start = sidx[p]
                end = sidx[p + len(key) - 1] + 1
                scursor = p + len(key)
        evidence = str(s.get("evidence") or "")
        try:
            conf = float(s.get("confidence") or 1.0)
        except (TypeError, ValueError):
            conf = 1.0
        # Auto-flag the worth-checking ones: low confidence, changed by block review,
        # or assigned to a rare/uncertain speaker. (Generic across samples.)
        flagged = (
            (i in FLAGGED)
            or conf < 0.85
            or "块级结构化复核" in evidence
            or s.get("speaker", "") in {"未知临时人物", "未知", "其他"}
        )
        out.append({
            "i": i,
            "speaker": s.get("speaker", ""),
            "orig_speaker": s.get("speaker", ""),
            "text": s.get("text", ""),
            "confidence": s.get("confidence"),
            "attribution_type": s.get("attribution_type"),
            "evidence": evidence[:120],
            "start": start,
            "end": end,
            "flagged": flagged,
        })
    return out


class State:
    def __init__(self, parse_path: Path, raw_path: Path, out_path: Path):
        self.out_path = out_path
        self.parse_name = parse_path.name
        self.raw = raw_path.read_text(encoding="utf-8")
        segs = json.loads(parse_path.read_text(encoding="utf-8"))["segments"]
        self.segments = align(self.raw, segs)
        # roster = known role hints + any speakers the parse actually produced (new chars)
        roster = list(load_roster())
        for seg in self.segments:
            sp = seg.get("speaker", "")
            if sp and sp not in roster and sp not in {"旁白", "未知", "其他"}:
                roster.append(sp)
        self.roster = roster
        self.corrections: dict[str, str] = {}
        if out_path.exists():
            try:
                self.corrections = json.loads(out_path.read_text(encoding="utf-8")).get("corrections", {})
            except Exception:
                self.corrections = {}
        # apply existing corrections to displayed speaker
        for seg in self.segments:
            c = self.corrections.get(str(seg["i"]))
            if c is not None:
                seg["speaker"] = c

    def save_one(self, i: int, speaker: str):
        orig = self.segments[i]["orig_speaker"]
        if speaker == orig:
            self.corrections.pop(str(i), None)
        else:
            self.corrections[str(i)] = speaker
        self.segments[i]["speaker"] = speaker
        self.out_path.write_text(json.dumps({
            "source_parse": self.parse_name,
            "updated": datetime.now().isoformat(timespec="seconds"),
            "corrections": self.corrections,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    def data(self) -> dict:
        return {"text": self.raw, "segments": self.segments, "roster": self.roster,
                "corrected": list(self.corrections.keys())}


HTML = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>说话人人工复核</title>
<style>
:root{--bg:#1e1e22;--panel:#26262c;--line:#34343c;--txt:#e6e6ea;--mut:#9a9aa6;--acc:#4ea1ff;--edit:#ffb454;--flag:#ff6b6b;--ok:#3ecf8e}
*{box-sizing:border-box}
body{margin:0;font:14px/1.6 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif;background:var(--bg);color:var(--txt);height:100vh;display:flex;flex-direction:column}
header{padding:8px 14px;background:var(--panel);border-bottom:1px solid var(--line);display:flex;gap:16px;align-items:center;flex:0 0 auto}
header b{color:var(--acc)} #status{color:var(--ok)} .muted{color:var(--mut)}
header input[type=text]{background:var(--bg);border:1px solid var(--line);color:var(--txt);border-radius:6px;padding:3px 8px;width:90px}
.main{flex:1;display:flex;min-height:0}
#list{flex:0 0 430px;overflow:auto;border-right:1px solid var(--line);padding:6px}
#doc{flex:1;overflow:auto;padding:20px 30px;white-space:pre-wrap;line-height:2.1}
.row{display:grid;grid-template-columns:38px 130px 1fr;gap:6px;align-items:start;padding:5px 4px;border-bottom:1px solid var(--line);cursor:pointer;border-radius:6px}
.row:hover{background:#2d2d35}
.row.active{background:#33384a;outline:1px solid var(--acc)}
.row.edited{background:#3a3320}
.idx{color:var(--mut);text-align:right;font-variant-numeric:tabular-nums}
.idx .flag{color:var(--flag)}
.row input.sp{width:100%;background:var(--bg);border:1px solid var(--line);color:var(--txt);border-radius:5px;padding:2px 6px}
.row.edited input.sp{border-color:var(--edit);color:var(--edit)}
.tx{color:var(--mut);font-size:12.5px;max-height:3.4em;overflow:hidden}
.orig{font-size:11px;color:var(--edit)}
.q{border-bottom:2px solid #3a3a44;border-radius:3px;padding:0 1px;cursor:pointer}
.q.active{background:var(--acc);color:#06121f;border-color:var(--acc)}
.q.edited{background:#5a4622;border-color:var(--edit)}
.q.flag{border-color:var(--flag)}
.q .tag{font-size:10px;color:var(--mut);vertical-align:super}
.q.active .tag{color:#06121f}
.filterbtn{background:var(--bg);border:1px solid var(--line);color:var(--txt);border-radius:6px;padding:3px 8px;cursor:pointer}
.filterbtn.on{background:var(--acc);color:#06121f}
</style></head><body>
<header>
  <b>说话人人工复核</b>
  <span class="muted">总 <span id="total">0</span> 句 · 已改 <span id="nedit">0</span> · 待核 <span id="nflag">0</span></span>
  <button class="filterbtn" id="fFlag">只看待核⚑</button>
  <button class="filterbtn" id="fEdit">只看已改</button>
  跳到 <input type="text" id="jump" placeholder="序号"/>
  <span id="status"></span>
</header>
<div class="main">
  <div id="list"></div>
  <div id="doc"></div>
</div>
<datalist id="roster"></datalist>
<script>
let DATA=null, active=-1;
const esc=s=>s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
async function load(){DATA=await (await fetch('/data')).json();render();}
function render(){
  // roster datalist
  document.getElementById('roster').innerHTML=DATA.roster.map(n=>`<option value="${esc(n)}">`).join('');
  // center doc with quote spans
  const t=DATA.text, segs=DATA.segments.filter(s=>s.start>=0).sort((a,b)=>a.start-b.start);
  let html='',prev=0;
  for(const s of segs){
    if(s.start<prev)continue;
    html+=esc(t.slice(prev,s.start));
    const cls='q'+(s.flagged?' flag':'')+(s.speaker!==s.orig_speaker?' edited':'');
    html+=`<span class="${cls}" id="q${s.i}" data-i="${s.i}" onclick="select(${s.i})">`+esc(t.slice(s.start,s.end))+`<span class="tag" id="tag${s.i}">${esc(short(s.speaker))}</span></span>`;
    prev=s.end;
  }
  html+=esc(t.slice(prev));
  document.getElementById('doc').innerHTML=html;
  // left list
  const list=document.getElementById('list');
  list.innerHTML='';
  for(const s of DATA.segments){
    const row=document.createElement('div');
    row.className='row'+(s.speaker!==s.orig_speaker?' edited':'');
    row.id='row'+s.i;
    row.onclick=e=>{if(e.target.tagName!=='INPUT')select(s.i,true);};
    row.innerHTML=`<div class="idx">${s.flagged?'<span class="flag">⚑</span>':''}${s.i}</div>
      <div><input class="sp" list="roster" value="${esc(s.speaker)}" data-i="${s.i}">
        ${s.speaker!==s.orig_speaker?`<div class="orig">原: ${esc(s.orig_speaker)}</div>`:''}</div>
      <div class="tx">「${esc(s.text)}」</div>`;
    list.appendChild(row);
  }
  list.querySelectorAll('input.sp').forEach(inp=>{
    inp.addEventListener('change',()=>save(+inp.dataset.i,inp.value.trim()));
    inp.addEventListener('focus',()=>select(+inp.dataset.i));
  });
  updateCounts();
}
function short(n){return (n||'').length>4?n.slice(0,4)+'…':n;}
function updateCounts(){
  document.getElementById('total').textContent=DATA.segments.length;
  document.getElementById('nedit').textContent=DATA.segments.filter(s=>s.speaker!==s.orig_speaker).length;
  document.getElementById('nflag').textContent=DATA.segments.filter(s=>s.flagged).length;
}
function select(i,fromList){
  if(active>=0){document.getElementById('q'+active)?.classList.remove('active');document.getElementById('row'+active)?.classList.remove('active');}
  active=i;
  const q=document.getElementById('q'+i),row=document.getElementById('row'+i);
  q?.classList.add('active');row?.classList.add('active');
  q?.scrollIntoView({block:'center',behavior:'smooth'});
  if(fromList)q?.scrollIntoView({block:'center',behavior:'smooth'});
  else row?.scrollIntoView({block:'nearest'});
}
async function save(i,speaker){
  const seg=DATA.segments[i];seg.speaker=speaker;
  await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({i,speaker})});
  // update UI
  const row=document.getElementById('row'+i),q=document.getElementById('q'+i),tag=document.getElementById('tag'+i);
  const changed=speaker!==seg.orig_speaker;
  row.classList.toggle('edited',changed);q?.classList.toggle('edited',changed);
  if(tag)tag.textContent=short(speaker);
  // refresh orig note
  const cell=row.children[1];let note=cell.querySelector('.orig');
  if(changed&&!note){note=document.createElement('div');note.className='orig';cell.appendChild(note);}
  if(changed)note.textContent='原: '+seg.orig_speaker; else if(note)note.remove();
  updateCounts();
  const st=document.getElementById('status');st.textContent='已保存 #'+i+' → '+speaker;
  clearTimeout(window._t);window._t=setTimeout(()=>st.textContent='',1500);
}
// filters
let flt=null;
function applyFilter(kind){
  flt=(flt===kind)?null:kind;
  document.getElementById('fFlag').classList.toggle('on',flt==='flag');
  document.getElementById('fEdit').classList.toggle('on',flt==='edit');
  DATA.segments.forEach(s=>{
    const row=document.getElementById('row'+s.i);
    let show=true;
    if(flt==='flag')show=s.flagged;
    if(flt==='edit')show=s.speaker!==s.orig_speaker;
    row.style.display=show?'':'none';
  });
}
document.getElementById('fFlag').onclick=()=>applyFilter('flag');
document.getElementById('fEdit').onclick=()=>applyFilter('edit');
document.getElementById('jump').addEventListener('keydown',e=>{if(e.key==='Enter'){const i=+e.target.value;if(i>=0&&i<DATA.segments.length)select(i,true);}});
load();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    state: State = None  # set on server

    def _send(self, code, body, ctype="application/json"):
        b = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, HTML, "text/html")
        elif self.path == "/data":
            self._send(200, json.dumps(self.state.data(), ensure_ascii=False))
        else:
            self._send(404, "{}")

    def do_POST(self):
        if self.path == "/save":
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            self.state.save_one(int(payload["i"]), str(payload.get("speaker", "")).strip())
            self._send(200, json.dumps({"ok": True}))
        else:
            self._send(404, "{}")

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parse", type=Path, default=DEFAULT_PARSE)
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    Handler.state = State(args.parse, args.raw, args.out)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"复核工具已启动：http://127.0.0.1:{args.port}/")
    print(f"  原文: {args.raw.name}  解析: {args.parse.name}")
    print(f"  改动实时保存到: {args.out}")
    print("  Ctrl+C 退出")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出，复核结果已保存。")


if __name__ == "__main__":
    main()
