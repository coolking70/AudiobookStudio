"""Step 2：用 BookVoiceParser 完整流水线解析一个 sample.txt → parse.json + transcript.txt。

这是权威样本生产的「模型解析」步，必须用它（而不是裸调 agnes-2.0-flash 单遍直出），否则
parse 缺少 block_review / dense_scene_review_routing / address_term_backcheck 等纠错阶段，
model_speaker 是未经纠错的弱基线，与既有样本不可比，且 verify_sample.py 会 FAIL。

与 seg5 逐参数一致：agnes-2.0-flash，batch_size=8，max_tokens=5000，temperature=0，
context_chars=320，compact 输出，disable_thinking，narrator=甘织玲奈子，启用块级复核。

用法（先 `source .env` 让 AGNES_API_KEY 就位）：
    .venv/bin/python tools/parse_sample.py --seg muli4_seg6
或：
    .venv/bin/python tools/parse_sample.py --raw docs/samples/X_sample.txt --out docs/samples/X_parse.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SAMP = REPO / "docs/samples"
sys.path.insert(0, str(REPO / "BookVoiceParser"))
sys.path.insert(0, str(REPO / "tools"))

AGNES_BASE_URL = "https://apihub.agnes-ai.com/v1"
AGNES_MODEL = "agnes-2.0-flash"


def make_transcript(segments: list[dict]) -> str:
    lines = []
    for i, s in enumerate(segments):
        spk = s.get("speaker", "")
        conf = s.get("confidence")
        atype = s.get("attribution_type") or ""
        lines.append(f"[{i:>3}] {spk:<9}({conf}) {atype}".rstrip())
        lines.append(f"      「{s.get('text', '')}」")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg", help="样本前缀，如 muli4_seg6（自动推导 raw/out/transcript 路径）")
    ap.add_argument("--raw", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--transcript", type=Path)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=5000)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--evidence-mode", default="short", choices=["short", "structured"],
                    help="初判 evidence 输出：short=旧短依据，structured=证据标签+风险标签")
    ap.add_argument("--narrator", default="甘织玲奈子",
                    help='第一人称叙述者全名（主线甘织视角默认）。第三人称/视角切换的外传'
                         '传空字符串 "" 或 none，让流水线不锚定固定叙述者。')
    args = ap.parse_args()
    narrator = args.narrator if args.narrator and args.narrator.lower() != "none" else None

    if args.seg:
        raw = args.raw or SAMP / f"{args.seg}_sample.txt"
        out = args.out or SAMP / f"{args.seg}_parse.json"
        transcript = args.transcript or SAMP / f"{args.seg}_transcript.txt"
    else:
        if not (args.raw and args.out):
            raise SystemExit("请给出 --seg，或同时给出 --raw 和 --out")
        raw, out, transcript = args.raw, args.out, args.transcript

    api_key = os.environ.get("AGNES_API_KEY")
    if not api_key:
        raise SystemExit("未设置 AGNES_API_KEY —— 请先 `source .env`（见 .env.example）")

    from book_voice_parser import BatchConfig, parse_novel
    from evaluate_agnes_bookmark_review import ROLE_HINTS

    text = raw.read_text(encoding="utf-8")
    cfg = BatchConfig(
        base_url=AGNES_BASE_URL, api_key=api_key, model=AGNES_MODEL,
        batch_size=args.batch_size, max_tokens=args.max_tokens, temperature=0.0,
        timeout=args.timeout, context_chars=320, output_mode="compact",
        evidence_mode=args.evidence_mode, disable_thinking=True,
    )
    print(f"解析 {raw.name}（{len(text)} 字）… 叙述者={narrator or '无(第三人称)'} 完整流水线，可能需数分钟", flush=True)
    result = parse_novel(
        text, role_hints=ROLE_HINTS, batch_llm_config=cfg, narrator=narrator,
        return_result=True, include_narration=False, review_threshold=0.7,
        enable_block_review=True,
    )
    segments = [seg.model_dump(mode="json") for seg in result.segments]
    out.write_text(json.dumps({"segments": segments, "stats": result.stats},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    if transcript:
        transcript.write_text(make_transcript(segments), encoding="utf-8")

    stats = result.stats or {}
    ran = [k for k in ("block_review", "dense_scene_review_routing", "address_term_backcheck",
                       "scene_state") if k in stats]
    print(f"✓ {out.name}: {len(segments)} 段；流水线阶段 {ran}")
    if transcript:
        print(f"✓ {transcript.name}")
    print("下一步：tools/review_server.py 复核 → build_groundtruth.py 固化 → verify_sample.py 验收")


if __name__ == "__main__":
    main()
