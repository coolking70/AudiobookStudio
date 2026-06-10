"""Step 3 前置：机器审计（API 按规则执行），替代人工/AI 手工初筛。

对流水线 parse 的每个具名段做「聚焦重问」（宽原文上下文 ±800 字、标注目标句、
带归属约定），与流水线意见比对；可选再叠加裸单遍 parse 的意见。产出两级标记：

  tier1（⚑必看）  = 重问≠流水线 ∪ 裸≠流水线(若有) ∪ 被块级复核改过
  tier2（·顺带看）= tier1 的 ±1 邻段（话轮翻转错误会连锁，邻段风险高；
                    复核 tier1 时邻段就在上下文里，几乎不增加负担）

实测（seg6+seg7，439 具名段 26 错，2026-06-10）：
  tier1 标 25% 段 → 命中 88% 错误（precision 21%）
  tier1+tier2     → 命中 96% 错误
  对照旧 ⚑ 规则（conf<0.85 等）：标 47% 段 / precision 13% / recall 84%

注意：三方多数票的「建议答案」实测正确率很低（同模型共享盲区），所以本工具只做
分流不做自动改判——最终裁决永远交人工。重问意见仅作为参考提示展示。

用法（先 `source .env`）：
    .venv/bin/python tools/audit_sample.py --seg muli4_seg8
产出 docs/samples/muli4_seg8_audit.json；review_server.py 检测到该文件会自动用
tier1/tier2 替代默认 ⚑ 规则，并显示重问意见。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SAMP = REPO / "docs/samples"
sys.path.insert(0, str(REPO / "tools"))

from evaluate_agnes_bookmark_review import ROLE_HINTS  # noqa: E402

AGNES_URL = "https://apihub.agnes-ai.com/v1/chat/completions"
AGNES_MODEL = "agnes-2.0-flash"
STRIP = set("「」『』《》〈〉 \t\r\n　")

_A2C = {}
for _c, _al in ROLE_HINTS.items():
    _names = _al if isinstance(_al, (list, tuple)) else _al.get("aliases", [])
    for _a in _names:
        _A2C[_a] = _c


def canon(x: str) -> str:
    return _A2C.get(x, x)


def is_crowd(x: str) -> bool:
    x = x or ""
    return x.startswith(("群众·", "厕所女生")) or x in {"未知", "未知临时人物", "旁白", "其他", ""}


def build_locator(raw: str):
    """对话内文字的剥离索引（与 review_server.align 同思路），供按文本定位原文偏移。"""
    sidx, schars, depth = [], [], 0
    for k, ch in enumerate(raw):
        if ch in "「『":
            depth += 1
            continue
        if ch in "」』":
            depth = max(0, depth - 1)
            continue
        if depth == 0 or ch in STRIP:
            continue
        schars.append(ch)
        sidx.append(k)
    sraw = "".join(schars)

    def locate(text: str):
        key = "".join(c for c in text if c not in STRIP)
        p = sraw.find(key)
        return (sidx[p], sidx[p + len(key) - 1] + 1) if p >= 0 else (-1, -1)

    return locate


TOKENHUB_URL = "https://tokenhub.tencentmaas.com/v1/chat/completions"
TOKENHUB_MODEL = "glm-5-turbo"  # 异构第二意见（2026-06-10 验证：纠对81%/漏网捕获6/8/误标8%）


def call_api(prompt: str, api_key: str, retries: int = 6, *, url: str = AGNES_URL,
             model: str = AGNES_MODEL, max_tokens: int = 200, extra: dict | None = None) -> str | None:
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.0}
    if extra:
        payload.update(extra)
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"})
    for t in range(retries):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=150))
            return r["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            if "429" in str(e) or "timed out" in str(e).lower():
                time.sleep(2 ** t + random.random() * 2)
                continue
            return None
    return None


def make_prompt(roster: str, narrator: str | None, ctx_b: str, text: str, ctx_a: str) -> str:
    nar = (f'本书为{narrator}第一人称叙述（叙述中的"我"={narrator}）。' if narrator
           else "本段为第三人称叙述，没有固定的第一人称叙述者。")
    return f"""你是小说说话人标注专家。{nar}
已知角色：{roster}

下面是小说原文片段，其中用【【】】标出了一句对话，请判断这句话是谁说出口的。
归属约定（务必遵守）：
1. 分析前后对话轮换、称呼习惯、引述动词归属。
2. 若是叙述者回忆/引用某人说过的话（如"想起X说的『…』"），说话人=被引用的人X，不是叙述者。
3. 若是叙述者想象/假想某人会说的话，说话人=被想象的人。
4. 若【【】】内只是叙述中引用的概念词/术语/书名（无人说出口），回答"旁白"。

{ctx_b}【【「{text}」】】{ctx_a}

只输出JSON：{{"speaker":"角色全名或旁白","reason":"15字内依据"}}"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg", required=True, help="样本前缀，如 muli4_seg8")
    ap.add_argument("--narrator", default="甘织玲奈子",
                    help='第一人称叙述者；第三人称样本传 none')
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--context", type=int, default=800)
    ap.add_argument("--hetero", action="store_true",
                    help="异构第二轮：glm-5-turbo(tokenhub) 扫描 agnes 一致段，抓同构盲区"
                         "（额度有限，只在固化前的最终审计用）")
    args = ap.parse_args()
    narrator = args.narrator if args.narrator and args.narrator.lower() != "none" else None

    api_key = os.environ.get("AGNES_API_KEY")
    if not api_key:
        raise SystemExit("未设置 AGNES_API_KEY —— 请先 `source .env`")

    raw = (SAMP / f"{args.seg}_sample.txt").read_text(encoding="utf-8")
    parse = json.loads((SAMP / f"{args.seg}_parse.json").read_text(encoding="utf-8"))["segments"]
    bare_p = SAMP / f"{args.seg}_bareflash_parse.json"
    bare = json.loads(bare_p.read_text(encoding="utf-8"))["segments"] if bare_p.exists() else None

    locate = build_locator(raw)
    roster = "、".join(ROLE_HINTS.keys())

    def ask(i: int):
        text = parse[i].get("text", "")
        st, en = locate(text)
        if st < 0:
            return i, None
        prompt = make_prompt(roster, narrator,
                             raw[max(0, st - args.context):st], text, raw[en:en + args.context])
        out = call_api(prompt, api_key)
        if not out:
            return i, None
        m = re.search(r"\{[^{}]*\}", out)
        try:
            return i, json.loads(m.group()) if m else None
        except Exception:  # noqa: BLE001
            return i, None

    targets = [i for i, p in enumerate(parse) if not is_crowd(p.get("speaker", ""))]
    print(f"审计 {args.seg}: {len(parse)} 段，重问 {len(targets)} 个具名段（{args.workers} 并发）…", flush=True)
    reask: dict[int, dict | None] = {}
    with ThreadPoolExecutor(args.workers) as ex:
        for i, r in ex.map(ask, targets):
            reask[i] = r

    # 两级标记
    n = len(parse)
    tier1 = [False] * n
    reasons: dict[int, list[str]] = {}
    for i, p in enumerate(parse):
        if is_crowd(p.get("speaker", "")):
            continue
        rs = []
        r = reask.get(i)
        rk = canon(str(r.get("speaker", ""))) if r else None
        if rk and rk != canon(p["speaker"]):
            rs.append(f"重问意见={rk}")
        if bare and canon(bare[i]["speaker"]) != canon(p["speaker"]):
            rs.append(f"裸单遍意见={canon(bare[i]['speaker'])}")
        if "块级结构化复核" in str(p.get("evidence") or ""):
            rs.append("被块级复核改过")
        if rs:
            tier1[i] = True
            reasons[i] = rs
    # 异构第二轮（--hetero，省额度）：glm-5-turbo 只扫 agnes 各路意见一致的段
    # （tier1 之外的具名段），分歧即补入 tier1——专抓同构模型的集体盲区。
    hetero_used = 0
    if args.hetero:
        th_key = os.environ.get("TOKENHUB_API_KEY")
        if not th_key:
            raise SystemExit("--hetero 需要 TOKENHUB_API_KEY（见 .env）")
        sweep = [i for i in targets if not tier1[i]]
        print(f"异构第二轮: glm-5-turbo 扫描 {len(sweep)} 个一致段…", flush=True)

        def ask_th(i: int):
            text = parse[i].get("text", "")
            st, en = locate(text)
            if st < 0:
                return i, None
            prompt = make_prompt(roster, narrator,
                                 raw[max(0, st - args.context):st], text, raw[en:en + args.context])
            out = call_api(prompt, th_key, url=TOKENHUB_URL, model=TOKENHUB_MODEL,
                           max_tokens=2000, extra={"chat_template_kwargs": {"enable_thinking": False}})
            if not out:
                return i, None
            m = re.search(r"\{[^{}]*\}", out)
            try:
                return i, json.loads(m.group()) if m else None
            except Exception:  # noqa: BLE001
                return i, None

        with ThreadPoolExecutor(args.workers) as ex:
            for i, r in ex.map(ask_th, sweep):
                hetero_used += 1
                hk = canon(str(r.get("speaker", ""))) if r else None
                if hk and hk != canon(parse[i]["speaker"]):
                    tier1[i] = True
                    reasons.setdefault(i, []).append(f"异构意见={hk}")
                    reask.setdefault(i, r)

    tier2 = [False] * n
    for i, f in enumerate(tier1):
        if f:
            for d in (-1, 1):
                if 0 <= i + d < n and not tier1[i + d]:
                    tier2[i + d] = True

    out = {
        "seg": args.seg,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "method": "focused-reask + bare-disagree + block-review-changed"
                  + (" + hetero(glm-5-turbo)一致段扫描" if args.hetero else "")
                  + "; tier2=±1邻段",
        "hetero_calls": hetero_used,
        "narrator": narrator,
        "reask_done": sum(1 for v in reask.values() if v),
        "tier1": [i for i, f in enumerate(tier1) if f],
        "tier2": [i for i, f in enumerate(tier2) if f],
        "segments": {str(i): {
            "reask": (reask.get(i) or {}).get("speaker"),
            "reask_reason": (reask.get(i) or {}).get("reason"),
            "flags": reasons.get(i, []),
        } for i in targets},
    }
    out_p = SAMP / f"{args.seg}_audit.json"
    out_p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    named = len(targets)
    print(f"✓ {out_p.name}: tier1 {len(out['tier1'])}/{named} 段需重点复核，tier2 邻段 {len(out['tier2'])} 段")
    print("下一步：review_server.py 会自动加载该审计文件（⚑=tier1，提示重问意见）")


if __name__ == "__main__":
    main()
