"""Full analysis + block review on the new seg2 sample; dump a reviewable transcript."""
from __future__ import annotations
import json, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SAMP = REPO / "docs/samples"
sys.path.insert(0, str(REPO / "BookVoiceParser"))
sys.path.insert(0, str(REPO / "tools"))
from book_voice_parser import BatchConfig, parse_novel  # noqa
from evaluate_agnes_bookmark_review import ROLE_HINTS  # noqa


def main():
    if not os.getenv("AGNES_API_KEY"):
        raise SystemExit("AGNES_API_KEY not set")
    text = (SAMP / "muli4_seg2_sample.txt").read_text(encoding="utf-8")
    cfg = BatchConfig(base_url="https://apihub.agnes-ai.com/v1", api_key=os.environ["AGNES_API_KEY"],
                      model="agnes-2.0-flash", batch_size=8, max_tokens=5000, temperature=0.0,
                      timeout=180, context_chars=320, output_mode="compact", disable_thinking=True)
    res = parse_novel(text, role_hints=ROLE_HINTS, batch_llm_config=cfg, narrator="甘织玲奈子",
                      return_result=True, include_narration=False, review_threshold=0.7,
                      enable_block_review=True)
    segs = [s.model_dump(mode="json") for s in res.segments]
    (SAMP / "muli4_seg2_parse.json").write_text(json.dumps({"segments": segs, "stats": res.stats}, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = []
    for i, s in enumerate(segs):
        ev = (s.get("evidence") or "")[:50]
        lines.append(f"[{i:>3}] {s.get('speaker',''):<8} ({s.get('confidence')}) {s.get('attribution_type') or ''}")
        lines.append(f"      「{s.get('text','')}」")
        if ev:
            lines.append(f"      ev: {ev}")
    (SAMP / "muli4_seg2_transcript.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"segments: {len(segs)}")
    print("block_review:", json.dumps(res.stats.get("block_review"), ensure_ascii=False))
    sc = {}
    for s in segs:
        sc[s.get("speaker", "")] = sc.get(s.get("speaker", ""), 0) + 1
    print("speaker counts:", json.dumps(sc, ensure_ascii=False))


if __name__ == "__main__":
    main()
