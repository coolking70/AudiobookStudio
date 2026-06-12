"""测试 mimo-auto 作为异构第二意见的质量。

对 seg5 已知错误段（groundtruth 标注的 corrected_indices）逐一向 mimo-proxy 查询，
统计：
  - 命中：mimo 意见 ≠ 流水线归因 且 mimo 意见 = 正确答案
  - 误标：mimo 意见 ≠ 流水线归因 且 mimo 意见 ≠ 正确答案（或查询了无错误的段）
  - 漏网：mimo 意见 = 流水线归因（未发现错误）

用法：
    # 先在另一个终端启动代理：
    python tools/mimo_proxy.py
    # 然后：
    python tools/test_mimo_hetero.py [--seg muli4_seg5] [--proxy-url http://127.0.0.1:19999]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SAMP = REPO / "docs/samples"
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "BookVoiceParser"))

from evaluate_agnes_bookmark_review import ROLE_HINTS  # noqa: E402
from book_voice_parser.audit import build_locator, make_audit_prompt  # noqa: E402

_A2C: dict[str, str] = {}
for _c, _al in ROLE_HINTS.items():
    _names = _al if isinstance(_al, (list, tuple)) else _al.get("aliases", [])
    for _a in _names:
        _A2C[_a] = _c


def canon(x: str) -> str:
    return _A2C.get(x or "", x or "")


def call_proxy(prompt: str, proxy_url: str) -> dict | None:
    url = proxy_url.rstrip("/") + "/v1/chat/completions"
    payload = {"model": "mimo-auto",
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": 300}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer local"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=120))
        content = r["choices"][0]["message"]["content"]
        m = re.search(r"\{[^{}]*\}", content)
        return json.loads(m.group()) if m else None
    except Exception as e:
        print(f"  [ERROR] proxy call failed: {e}")
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg", default="muli4_seg5")
    ap.add_argument("--proxy-url", default="http://127.0.0.1:19999")
    ap.add_argument("--narrator", default="甘织玲奈子")
    ap.add_argument("--context", type=int, default=800)
    ap.add_argument("--also-test-correct", type=int, default=10,
                    help="额外测 N 个正确段，估算误标率（0 = 跳过）")
    args = ap.parse_args()

    narrator = args.narrator if args.narrator.lower() != "none" else None

    raw = (SAMP / f"{args.seg}_sample.txt").read_text(encoding="utf-8")
    parse = json.loads((SAMP / f"{args.seg}_parse.json").read_text())["segments"]
    gt = json.loads((SAMP / f"{args.seg}_groundtruth.json").read_text())
    error_indices = set(gt["corrected_indices"])

    locate = build_locator(raw)
    roster = "、".join(ROLE_HINTS.keys())

    # ── 测试已知错误段 ────────────────────────────────────────
    print(f"\n=== {args.seg}  已知错误段测试 ({len(error_indices)} 段) ===\n")
    hit = miss = fail = 0
    for i in sorted(error_indices):
        seg = parse[i]
        pipeline_ans = canon(seg.get("speaker", ""))
        # groundtruth 中正确答案（segments 为列表，按索引访问）
        gt_seg = gt["segments"][i]
        correct_ans = canon(gt_seg.get("speaker", pipeline_ans))

        text = seg.get("text", "")
        st, en = locate(text)
        if st < 0:
            print(f"  [{i:3d}] 定位失败，跳过")
            fail += 1
            continue

        prompt = make_audit_prompt(
            roster, narrator,
            raw[max(0, st - args.context):st], text, raw[en:en + args.context])

        print(f"  [{i:3d}] 流水线={pipeline_ans}  正确={correct_ans}  文本={repr(text[:30])}…", end="", flush=True)
        r = call_proxy(prompt, args.proxy_url)
        if not r:
            print(" → [无响应]")
            fail += 1
            continue
        mimo_ans = canon(r.get("speaker", ""))
        agrees_with_pipeline = (mimo_ans == pipeline_ans)
        agrees_with_correct = (mimo_ans == correct_ans)
        tag = "✓命中" if (not agrees_with_pipeline and agrees_with_correct) else \
              "✗误标" if (not agrees_with_pipeline and not agrees_with_correct) else "△漏网"
        print(f" → mimo={mimo_ans}  {tag}  reason={r.get('reason','')}")
        if tag == "✓命中":
            hit += 1
        elif tag == "△漏网":
            miss += 1
        else:
            fail += 1

    total_error = len(error_indices)
    print(f"\n错误段汇总: 命中={hit}/{total_error}  漏网={miss}/{total_error}  失败={fail}/{total_error}")
    print(f"  错误段召回率: {hit/max(1,total_error-fail)*100:.0f}%  (排除失败后)")

    # ── 额外测正确段（估算误标率）───────────────────────────
    if args.also_test_correct > 0:
        narrators = {"旁白", "未知", "未知临时人物", "其他", "UNKNOWN", ""}
        correct_segs = [i for i, s in enumerate(parse)
                        if i not in error_indices
                        and s.get("speaker", "") not in narrators][:args.also_test_correct]
        print(f"\n=== 正确段误标率估算 ({len(correct_segs)} 段) ===\n")
        false_alarms = 0
        fa_total = 0
        for i in correct_segs:
            seg = parse[i]
            pipeline_ans = canon(seg.get("speaker", ""))
            text = seg.get("text", "")
            st, en = locate(text)
            if st < 0:
                continue
            prompt = make_audit_prompt(
                roster, narrator,
                raw[max(0, st - args.context):st], text, raw[en:en + args.context])
            print(f"  [{i:3d}] 正确={pipeline_ans}  文本={repr(text[:30])}…", end="", flush=True)
            r = call_proxy(prompt, args.proxy_url)
            fa_total += 1
            if not r:
                print(" → [无响应]")
                continue
            mimo_ans = canon(r.get("speaker", ""))
            agrees = (mimo_ans == pipeline_ans)
            if not agrees:
                false_alarms += 1
                print(f" → mimo={mimo_ans}  ⚑误标  reason={r.get('reason','')}")
            else:
                print(f" → mimo={mimo_ans}  ✓一致")
        print(f"\n正确段误标率: {false_alarms}/{fa_total} = {false_alarms/max(1,fa_total)*100:.0f}%")


if __name__ == "__main__":
    main()
