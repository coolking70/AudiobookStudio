"""机器审计核心（两级分流）：对已归因的段做聚焦重问，标出值得人工复核的段。

方法与指标见 docs/flagging_audit_analysis_2026-06-10.md：
  tier1 ⚑（必看） = 重问意见≠当前归因 ∪ 被块级复核改过 ∪ 异构意见≠当前归因(可选)
  tier2 ·（顺带看）= tier1 的 ±1 邻段（话轮翻转错误连锁传染）
样本外实测：tier1 标 ~41% 段（旧置信度启发式 ~86%）、precision 28%、tier1+2 覆盖 93%。
审计只分流不裁决——重问意见仅作提示，最终判断交人工。

本模块是 tools/audit_sample.py（样本文件工作流）与 app.py /api/audit_segments
（生产前端）共用的核心，无文件依赖：输入原文 + 段列表 + LLM 配置，输出分级结果。
"""
from __future__ import annotations

import json
import random
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

_STRIP = set("「」『』《》〈〉 \t\r\n　")
_NARRATION_SPEAKERS = {"旁白", "未知", "未知临时人物", "其他", "UNKNOWN", ""}


def _scene_density(segments: list[dict], i: int, alias_map: dict[str, str], window: int = 4) -> int:
    """Count unique named speakers in ±window around index i."""
    canon = lambda x: alias_map.get(x or "", x or "")  # noqa: E731
    speakers: set[str] = set()
    for j in range(max(0, i - window), min(len(segments), i + window + 1)):
        sp = canon(str(segments[j].get("speaker", "")))
        if sp not in _NARRATION_SPEAKERS:
            speakers.add(sp)
    return len(speakers)


def _signal_priority(flags: list[str], is_dense: bool) -> int:
    """Tier-1 display priority: 1 = review first.
      P1  dense + strong (bare/hetero)
      P2  strong alone  |  dense + multi-signal
      P3  dense + reask  |  multi-signal alone
      P4  reask alone
      P5  dense + block-review only
    """
    has_strong = any("裸单遍意见" in f or "异构意见" in f for f in flags)
    has_reask = any("重问意见" in f for f in flags)
    has_block = "被块级复核改过" in flags
    multi = sum([has_strong, has_reask, has_block]) >= 2
    if has_strong and is_dense:
        return 1
    if has_strong or (multi and is_dense):
        return 2
    if (has_reask and is_dense) or (multi and not is_dense):
        return 3
    if has_reask:
        return 4
    return 5  # block-only in dense


def build_locator(raw: str) -> Callable[[str], tuple[int, int]]:
    """对话内文字剥离索引：只在引号内匹配，避免短引文错配到叙述里的同字子串。"""
    sidx: list[int] = []
    schars: list[str] = []
    depth = 0
    for k, ch in enumerate(raw):
        if ch in "「『":
            depth += 1
            continue
        if ch in "」』":
            depth = max(0, depth - 1)
            continue
        if depth == 0 or ch in _STRIP:
            continue
        schars.append(ch)
        sidx.append(k)
    sraw = "".join(schars)

    def locate(text: str) -> tuple[int, int]:
        key = "".join(c for c in text if c not in _STRIP)
        p = sraw.find(key)
        return (sidx[p], sidx[p + len(key) - 1] + 1) if p >= 0 else (-1, -1)

    return locate


def make_audit_prompt(roster: str, narrator: str | None, ctx_b: str, text: str, ctx_a: str) -> str:
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


def _call_llm(prompt: str, llm: dict, retries: int = 5) -> str | None:
    base = str(llm.get("base_url", "")).rstrip("/")
    url = base if base.endswith("/chat/completions") else base + "/chat/completions"
    payload: dict[str, Any] = {
        "model": llm.get("model"),
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(llm.get("max_tokens", 2000)),  # 思考型模型需≥2000，否则空响应
        "temperature": 0.0,
    }
    if llm.get("disable_thinking", True):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={
        "Authorization": f"Bearer {llm.get('api_key') or 'local'}",
        "Content-Type": "application/json"})
    for t in range(retries):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=int(llm.get("timeout", 150))))
            return r["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "429" in msg or "timed out" in msg.lower():
                time.sleep(2 ** t + random.random())
                continue
            if "400" in msg and "chat_template_kwargs" in json.dumps(payload):
                payload.pop("chat_template_kwargs", None)  # 不支持该参数的服务降级重试
                req.data = json.dumps(payload).encode()
                continue
            return None
    return None


def _parse_answer(out: str | None) -> dict | None:
    if not out:
        return None
    m = re.search(r"\{[^{}]*\}", out)
    try:
        return json.loads(m.group()) if m else None
    except Exception:  # noqa: BLE001
        return None


def audit_segments(
    raw_text: str,
    segments: list[dict],
    llm: dict,
    *,
    narrator: str | None = None,
    roster: list[str] | None = None,
    alias_map: dict[str, str] | None = None,
    hetero_llm: dict | None = None,
    workers: int = 2,
    context_chars: int = 800,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict:
    """对段列表做机器审计。

    segments: [{speaker, text, evidence?}, ...]（与解析输出同序；speaker 为旁白/未知的
              段跳过重问）。llm/hetero_llm: {base_url, api_key, model[, max_tokens,
              timeout, disable_thinking]}。
    返回 {"tier1": [idx], "tier2": [idx], "details": {idx: {"reask","reask_reason","flags"}}}。
    """
    a2c = dict(alias_map or {})
    canon = lambda x: a2c.get(x or "", x or "")  # noqa: E731
    locate = build_locator(raw_text)
    roster_str = "、".join(roster) if roster else "、".join(
        sorted({canon(s.get("speaker", "")) for s in segments
                if s.get("speaker") and s["speaker"] not in _NARRATION_SPEAKERS}))

    targets = [i for i, s in enumerate(segments)
               if str(s.get("speaker", "")) not in _NARRATION_SPEAKERS and s.get("text")]
    n = len(segments)
    details: dict[int, dict] = {}
    tier1 = [False] * n

    def ask(i: int, cfg: dict):
        seg = segments[i]
        st, en = locate(seg["text"])
        if st >= 0:
            ctx_b = raw_text[max(0, st - context_chars):st]
            ctx_a = raw_text[en:en + context_chars]
        else:
            # raw_text 缺失（如快照导入场景）：降级用段内嵌的上下文字段
            ctx_b = str(seg.get("context_before") or "")[-context_chars:]
            ctx_a = str(seg.get("context_after")  or "")[:context_chars]
            if not ctx_b and not ctx_a:
                return i, None  # 完全无上下文才跳过
        prompt = make_audit_prompt(roster_str, narrator, ctx_b, seg["text"], ctx_a)
        return i, _parse_answer(_call_llm(prompt, cfg))

    done = 0
    total = len(targets) + (len(targets) if hetero_llm else 0)  # 上界，异构实际更少
    priorities: dict[int, int] = {}
    downgraded: set[int] = set()  # 简单场景仅块复核 → 直接入 tier2
    with ThreadPoolExecutor(max(1, workers)) as ex:
        for i, r in ex.map(lambda i: ask(i, llm), targets):
            done += 1
            if on_progress:
                on_progress(done, total)
            flags = []
            rk = canon(str(r.get("speaker", ""))) if r else None
            if rk and rk != canon(segments[i].get("speaker", "")):
                flags.append(f"重问意见={rk}")
            if "块级结构化复核" in str(segments[i].get("evidence") or ""):
                flags.append("被块级复核改过")
            if flags:
                density = _scene_density(segments, i, a2c)
                is_dense = density >= 3
                if not is_dense and flags == ["被块级复核改过"]:
                    # 简单场景（≤2人）仅块复核触发：基础错误率 3.3%，精度太低 → 降为 tier2
                    downgraded.add(i)
                else:
                    tier1[i] = True
                    priorities[i] = _signal_priority(flags, is_dense)
            details[i] = {"reask": (r or {}).get("speaker"),
                          "reask_reason": (r or {}).get("reason"), "flags": flags,
                          "priority": priorities.get(i)}

    if hetero_llm:
        sweep = [i for i in targets if not tier1[i]]
        with ThreadPoolExecutor(max(1, workers)) as ex:
            for i, r in ex.map(lambda i: ask(i, hetero_llm), sweep):
                done += 1
                if on_progress:
                    on_progress(done, total)
                hk = canon(str(r.get("speaker", ""))) if r else None
                if hk and hk != canon(segments[i].get("speaker", "")):
                    tier1[i] = True
                    downgraded.discard(i)  # 异构信号优先级高，取消降级
                    details[i]["flags"].append(f"异构意见={hk}")
                    details[i]["hetero"] = r.get("speaker")
                    density = _scene_density(segments, i, a2c)
                    p = _signal_priority(details[i]["flags"], density >= 3)
                    priorities[i] = p
                    details[i]["priority"] = p

    tier2 = [False] * n
    for i, f in enumerate(tier1):
        if f:
            for d in (-1, 1):
                if 0 <= i + d < n and not tier1[i + d]:
                    tier2[i + d] = True
    # 降级段直接入 tier2（不扩散邻段）
    for i in downgraded:
        if not tier1[i]:
            tier2[i] = True

    # tier1 按优先级排序（同级内按索引保持稳定）
    tier1_sorted = sorted(
        (i for i, f in enumerate(tier1) if f),
        key=lambda i: (priorities.get(i, 99), i),
    )

    return {
        "tier1": tier1_sorted,
        "tier2": [i for i, f in enumerate(tier2) if f],
        "details": {i: d for i, d in details.items() if d["flags"] or d.get("reask")},
        "audited": len(targets),
        "stats": {
            "downgraded_to_tier2": len(downgraded),
        },
    }
