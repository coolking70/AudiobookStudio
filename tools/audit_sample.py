"""Step 3 前置：机器审计（API 按规则执行），替代人工/AI 手工初筛。

对流水线 parse 的每个具名段做「聚焦重问」（宽原文上下文 ±800 字、标注目标句、
带归属约定），与流水线意见比对；可选再叠加裸单遍 parse 的意见。产出两级标记：

  tier1（⚑必看）  = 重问≠流水线 ∪ 裸≠流水线(若有) ∪ 被块级复核改过
                    【场景感知例外】简单场景（±4窗口内≤2个具名说话人）且仅"被块级复核
                    改过"触发时，降为 tier2——简单场景基础错误率仅 3.3%，该信号此处
                    精度太低（多数是块复核把初判错误成功纠正后的误标）。
  tier2（·顺带看）= tier1 的 ±1 邻段 ∪ 简单场景仅块复核触发段（话轮翻转错误会
                    连锁，邻段风险高；复核 tier1 时邻段就在上下文里，几乎不增加负担）

tier1 内部按信号强度打优先级（P1 最高）：
  P1  密集场景（≥3人）+ 强信号（裸单遍/异构意见分歧）
  P2  强信号，非密集  |  密集 + 多信号交叉
  P3  密集 + 重问分歧  |  多信号交叉，非密集
  P4  重问分歧，非密集
  P5  密集 + 仅块复核改过

实测（seg6+seg7，439 具名段 26 错，2026-06-10）：
  tier1 标 25% 段 → 命中 88% 错误（precision 21%）
  tier1+tier2     → 命中 96% 错误
  对照旧 ⚑ 规则（conf<0.85 等）：标 47% 段 / precision 13% / recall 84%
  场景感知改造预计：tier1 标记量再降 ~35%，recall 损失 <5%（简单场景低错误率）

注意：三方多数票的「建议答案」实测正确率很低（同模型共享盲区），所以本工具只做
分流不做自动改判——最终裁决永远交人工。重问意见仅作为参考提示展示。

用法（先 `source .env`）：
    .venv/bin/python tools/audit_sample.py --seg muli4_seg8
产出 docs/samples/muli4_seg8_audit.json；review_server.py 检测到该文件会自动用
tier1/tier2 替代默认 ⚑ 规则，并显示重问意见及优先级。
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
sys.path.insert(0, str(REPO / "BookVoiceParser"))

from evaluate_agnes_bookmark_review import ROLE_HINTS  # noqa: E402
from book_voice_parser.audit import build_locator, make_audit_prompt as make_prompt  # noqa: E402  共享核心

AGNES_URL = "https://apihub.agnes-ai.com/v1/chat/completions"
AGNES_MODEL = "agnes-2.0-flash"
TOKENHUB_URL = "https://tokenhub.tencentmaas.com/v1/chat/completions"
TOKENHUB_MODEL = "glm-5-turbo"  # 异构第二意见（验证：纠对81%/漏网捕获6/8/误标8%）
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


def scene_density(parse: list, i: int, window: int = 4) -> int:
    """Count unique named speakers in ±window around index i."""
    speakers: set[str] = set()
    for j in range(max(0, i - window), min(len(parse), i + window + 1)):
        sp = canon(parse[j].get("speaker", ""))
        if not is_crowd(sp):
            speakers.add(sp)
    return len(speakers)


def signal_priority(signals: list[str], is_dense: bool) -> int:
    """
    Compute tier1 display priority (lower = review first).
      P1  密集 + 强信号（裸单遍/异构）
      P2  强信号非密集  |  密集+多信号
      P3  密集+重问  |  多信号非密集
      P4  重问非密集
      P5  密集+仅块复核
    """
    has_strong = any("裸单遍意见" in s or "异构意见" in s or "单遍意见" in s for s in signals)
    has_reask = any("重问意见" in s for s in signals)
    has_block = "被块级复核改过" in signals
    multi = sum([has_strong, has_reask, has_block]) >= 2

    if has_strong and is_dense:
        return 1
    if has_strong or (multi and is_dense):
        return 2
    if (has_reask and is_dense) or (multi and not is_dense):
        return 3
    if has_reask:
        return 4
    return 5  # block-only in dense scene


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
    ap.add_argument("--singlepass", action="store_true",
                    help="单遍直出第二意见：整文阅读+角色表注入，一次性归因；与流水线分歧入 tier1"
                         "（强信号，suggest-only，绝不改写 parse）。专抓分批窄窗丢失的跨句轮换/在场错误")
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

    # 两级标记（场景感知：简单场景仅块复核触发 → 降为 tier2）
    n = len(parse)
    tier1 = [False] * n
    reasons: dict[int, list[str]] = {}
    priorities: dict[int, int] = {}
    downgraded: dict[int, list[str]] = {}  # 简单场景仅块复核 → 直接入 tier2
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
            density = scene_density(parse, i)
            is_dense = density >= 3
            if not is_dense and rs == ["被块级复核改过"]:
                # 简单场景仅块复核：基础错误率 3.3%，信号精度太低 → 降为 tier2
                downgraded[i] = rs
            else:
                tier1[i] = True
                reasons[i] = rs
                priorities[i] = signal_priority(rs, is_dense)
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
                    downgraded.pop(i, None)  # 异构信号优先级高，取消之前的降级
                    rs_h = reasons.setdefault(i, [])
                    rs_h.append(f"异构意见={hk}")
                    reask.setdefault(i, r)
                    density = scene_density(parse, i)
                    priorities[i] = signal_priority(rs_h, density >= 3)

    # 单遍直出第二意见（--singlepass）：整文阅读 + 角色表注入，一次性归因，
    # 与流水线分歧即入 tier1（强信号）。suggest-only：只读 parse、绝不改写其 speaker。
    sp_used = 0
    if args.singlepass:
        from book_voice_parser.single_pass_attributor import SinglePassAttributor
        from book_voice_parser.batch_llm_attributor import BatchConfig
        from book_voice_parser.schema import QuoteSpan
        role_aliases = {
            c: (al if isinstance(al, (list, tuple)) else al.get("aliases", []))
            for c, al in ROLE_HINTS.items()
        }
        quotes, qid2idx = [], {}
        for i in targets:
            qid = str(parse[i].get("quote_id") or f"L{i}")
            quotes.append(QuoteSpan(quote_id=qid, text=parse[i].get("text", ""),
                                    start=0, end=len(parse[i].get("text", "")),
                                    context_before="", context_after=""))
            qid2idx[qid] = i
        cfg = BatchConfig(base_url=AGNES_URL.rsplit("/chat/completions", 1)[0],
                          api_key=api_key, model=AGNES_MODEL, timeout=300)
        sp = SinglePassAttributor(cfg, full_text=raw, chunk_size=200)
        print(f"单遍第二意见: 整文阅读归因 {len(quotes)} 个具名段…", flush=True)
        sp_res = sp.attribute(quotes, role_hints=list(ROLE_HINTS.keys()),
                              narrator=narrator, role_aliases=role_aliases)
        for qid, a in sp_res.items():
            i = qid2idx.get(qid)
            if i is None:
                continue
            sp_used += 1
            sk = canon(str(getattr(a, "speaker", "") or ""))
            if sk and not is_crowd(sk) and sk != canon(parse[i]["speaker"]):
                tier1[i] = True
                downgraded.pop(i, None)  # 强信号优先级高，取消之前的降级
                rs_s = reasons.setdefault(i, [])
                if not any("单遍意见" in s for s in rs_s):
                    rs_s.append(f"单遍意见={sk}")
                reask.setdefault(i, {"speaker": sk, "reason": "单遍直出"})
                priorities[i] = signal_priority(rs_s, scene_density(parse, i) >= 3)

    tier2 = [False] * n
    for i, f in enumerate(tier1):
        if f:
            for d in (-1, 1):
                if 0 <= i + d < n and not tier1[i + d]:
                    tier2[i + d] = True
    # 降级的简单场景仅块复核段直接入 tier2（不生成它自己的邻段扩散）
    for i in downgraded:
        if not tier1[i]:
            tier2[i] = True

    # tier1 按优先级排序（同优先级内按索引稳定排序）
    tier1_sorted = sorted(
        (i for i, f in enumerate(tier1) if f),
        key=lambda i: (priorities.get(i, 99), i),
    )

    # 降级统计
    n_downgraded = len(downgraded)
    n_tier1_orig = len(tier1_sorted) + n_downgraded  # 如果没有场景感知会有多少 tier1

    out = {
        "seg": args.seg,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "method": "focused-reask + bare-disagree + block-review-changed(scene-aware)"
                  + (" + hetero(glm-5-turbo)一致段扫描" if args.hetero else "")
                  + (" + singlepass(整文+角色表)第二意见" if args.singlepass else "")
                  + "; tier2=±1邻段+简单场景降级段",
        "hetero_calls": hetero_used,
        "singlepass_segments": sp_used,
        "narrator": narrator,
        "reask_done": sum(1 for v in reask.values() if v),
        "tier1": tier1_sorted,  # 按优先级排序，P1 在前
        "tier2": [i for i, f in enumerate(tier2) if f],
        "stats": {
            "tier1_count": len(tier1_sorted),
            "tier2_count": sum(tier2),
            "downgraded_to_tier2": n_downgraded,
            "tier1_saved_vs_naive": n_downgraded,
        },
        "segments": {str(i): {
            "reask": (reask.get(i) or {}).get("speaker"),
            "reask_reason": (reask.get(i) or {}).get("reason"),
            "flags": reasons.get(i) or downgraded.get(i, []),
            "priority": priorities.get(i),
            "downgraded": i in downgraded,
        } for i in targets},
    }
    out_p = SAMP / f"{args.seg}_audit.json"
    out_p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    named = len(targets)
    saved_pct = f"（场景感知降级 {n_downgraded} 段，节省 {n_downgraded/named*100:.1f}%）" if n_downgraded else ""
    print(f"✓ {out_p.name}: tier1 {len(tier1_sorted)}/{named} 段需重点复核{saved_pct}，tier2 {sum(tier2)} 段")
    print("  优先级分布: " + "  ".join(
        f"P{p}×{sum(1 for v in priorities.values() if v==p)}"
        for p in sorted(set(priorities.values()))
    ))
    print("下一步：review_server.py 会自动加载该审计文件（⚑=tier1 按优先级排序，·=tier2）")


if __name__ == "__main__":
    main()
