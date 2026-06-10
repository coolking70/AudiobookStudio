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
        st, en = locate(segments[i]["text"])
        if st < 0:
            return i, None
        prompt = make_audit_prompt(
            roster_str, narrator,
            raw_text[max(0, st - context_chars):st], segments[i]["text"],
            raw_text[en:en + context_chars])
        return i, _parse_answer(_call_llm(prompt, cfg))

    done = 0
    total = len(targets) + (len(targets) if hetero_llm else 0)  # 上界，异构实际更少
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
                tier1[i] = True
            details[i] = {"reask": (r or {}).get("speaker"),
                          "reask_reason": (r or {}).get("reason"), "flags": flags}

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
                    details[i]["flags"].append(f"异构意见={hk}")
                    details[i]["hetero"] = r.get("speaker")

    tier2 = [False] * n
    for i, f in enumerate(tier1):
        if f:
            for d in (-1, 1):
                if 0 <= i + d < n and not tier1[i + d]:
                    tier2[i + d] = True

    return {
        "tier1": [i for i, f in enumerate(tier1) if f],
        "tier2": [i for i, f in enumerate(tier2) if f],
        "details": {i: d for i, d in details.items() if d["flags"] or d.get("reask")},
        "audited": len(targets),
    }
