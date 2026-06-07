"""block_review.py
------------------
对话块级结构化复核（第二趟）。

动机：主 BatchLLM 归因是"逐句"判断，缺少对一整段对话的【轮次结构】建模，在
日式双人快速对话、同名/同类配角同框等场景容易"错一格"。本模块对每一段连续对话
块（同场景、相邻台词间旁白间隙小）整体送给 LLM，按真人读法重判：

  1. 先读旁白确定在场人物与【登场顺序】（"A与B走过来"→先出现者先开口）；
  2. 用登场顺序锚定首发，按【续说 / 两人交替 / 第三人加入】推进；
  3. 用【称呼】【语气/语域】【语义邻接】（提问→回答、提议→拒绝）校正。

应用时做【场景感知的显式锚点保护】：在 3+ 人同框的块里，保留 baseline 的显式归属
（如"X 说道"），只在干净 2 人块里让结构化结果覆盖——避免在多人场景错锚扩散。

设计原则：失败安全（任一块复核出错只跳过该块，绝不影响主流程返回）。
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import httpx

from .alias_registry import AliasRegistry
from .schema import AttributionType, QuoteSpan, SegmentEx

logger = logging.getLogger(__name__)

# 对话块分组间隙（与 parser._DIALOGUE_GAP_THRESHOLD 对齐）
GAP_THRESHOLD = 150
MIN_BLOCK = 4          # 仅对 >=4 句的连续对话块做结构化复核（短交流不易出错）
SUB_PER = 14           # 大块按 14 句分窗
SUB_LOOKBACK = 3       # 分窗回看 3 句提供连续性
LEADIN = 400           # 窗口向前多取的旁白字数（捕捉"A与B走过来"式引入）
TRAIL = 100
PROTECT_SCENE_SPEAKERS = 3   # 块内具名说话人 >= 此数时，保护 baseline 显式锚点

SKIP_NAMES = {"旁白", "未知", "其他", "众人", "大家", "二人", "三人", "所有人"}

_SYSTEM = "你是中文小说说话人归属专家。像真人读小说那样，结合上下文与对话轮次判断每句对话由谁说出。"

_PROMPT = """下面是一段连续的小说原文（旁白与对话混排），每句对话前有【{prefix}NNNN】标记。

角色表（只用其中的规范名，不要用别名）：
{roster}
叙述者（第一人称“我”）：{narrator}

请按真人阅读方式判断每句对话的说话人：
1. 先读旁白，确定这一段【在场人物】及其【登场顺序】——留意“A与B / A和B / A、B 走了过来”这类并列引入，先出现者通常先开口。
2. 用登场顺序【锚定第一句】说话人，定下交替基准。
3. 逐句判断它是【与上句同一人续说】【两人交替】还是【有新人加入 / 某人退出】；没有其他线索时，默认两人按登场顺序交替。
4. 用以下证据【校正】：
   - 称呼：对话里出现的称呼/亲属称谓指向被称呼者，发话者通常不是被喊的那位（如有人喊“X、姊姊、哥哥”）。
   - 语气/语域：不同角色说话风格不同（强势 / 怯懦 / 毒舌 / 客气 / 内心戏多）。
   - 语义邻接：提问→回答、提议→拒绝或接受、递出→接住，多为两人交替；一个人连续做一件事（如逐张点评照片）可能是续说。
5. 第一人称“我”说出口的话归叙述者 {narrator}。
6. 无名路人归“其他”；若某【{prefix}】其实不是说出口的台词则归“旁白”。

【本段原文】：
{block}

只输出一个 JSON 对象：键为本段每个编号，值为规范名。例如 {{"{prefix}0012":"角色甲","{prefix}0013":"角色乙"}}。不要任何解释。"""


def _is_explicit(seg: SegmentEx) -> bool:
    at = seg.attribution_type
    at = at.value if isinstance(at, AttributionType) else at
    return isinstance(at, str) and at.startswith("explicit")


def _group_blocks(quotes: list[QuoteSpan]) -> list[list[int]]:
    if not quotes:
        return []
    blocks: list[list[int]] = []
    cur = [0]
    for i in range(1, len(quotes)):
        pe = quotes[i - 1].raw_end if quotes[i - 1].raw_end is not None else quotes[i - 1].end
        cs = quotes[i].raw_start if quotes[i].raw_start is not None else quotes[i].start
        if (cs - pe) <= GAP_THRESHOLD:
            cur.append(i)
        else:
            blocks.append(cur)
            cur = [i]
    blocks.append(cur)
    return blocks


def _subchunks(idxs: list[int]):
    """Yield (context_idxs, assign_idxs): assign only new quotes, but render a window
    that includes a few lookback quotes for continuity."""
    i = 0
    while i < len(idxs):
        ctx = idxs[max(0, i - SUB_LOOKBACK): i + SUB_PER]
        new = idxs[i: i + SUB_PER]
        yield ctx, new
        i += SUB_PER


def _render_window(cleaned: str, quotes: list[QuoteSpan], ctx_idxs: list[int]) -> str:
    i0, i1 = ctx_idxs[0], ctx_idxs[-1]
    ws = max(0, quotes[i0].start - LEADIN)
    we = min(len(cleaned), quotes[i1].end + TRAIL)
    win = cleaned[ws:we]
    for pos, qid in sorted(((quotes[k].start, quotes[k].quote_id) for k in ctx_idxs), reverse=True):
        rel = pos - ws
        if 0 <= rel <= len(win):
            win = win[:rel] + f"【{qid}】" + win[rel:]
    return win


def _roster_text(role_hints: list[str], aliases: AliasRegistry) -> str:
    # group aliases under canonical for the prompt
    canon_to_alias: dict[str, list[str]] = {name: [] for name in role_hints}
    for alias, canon in getattr(aliases, "alias_map", {}).items():
        if canon in canon_to_alias and alias != canon:
            canon_to_alias[canon].append(alias)
    lines = []
    for name in role_hints:
        al = canon_to_alias.get(name) or []
        lines.append(name + (f"（别名：{'、'.join(al)}）" if al else ""))
    return "\n".join(lines)


def _call_llm(base_url: str, api_key: str, model: str, prompt: str, timeout: int) -> str:
    base = (base_url or "").strip().rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")].rstrip("/")
    url = f"{base}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 1500,
    }
    headers = {"Authorization": f"Bearer {api_key or 'local'}", "Content-Type": "application/json"}
    attempt = 0
    with httpx.Client(timeout=timeout, trust_env=False) as cli:
        while True:
            r = cli.post(url, headers=headers, json=payload)
            if r.status_code == 429 and attempt < 6:
                time.sleep(min(60.0, 5.0 * (2 ** attempt)))
                attempt += 1
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]


def _parse_json(content: str) -> dict:
    content = re.sub(r"```(json)?|```", "", str(content or "")).strip()
    m = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not m:
        return {}
    try:
        out = json.loads(m.group(0))
        return out if isinstance(out, dict) else {}
    except json.JSONDecodeError:
        return {}


def apply_block_review(
    segments: list[SegmentEx],
    quotes: list[QuoteSpan],
    cleaned: str,
    batch_config: Any,
    *,
    narrator: str | None,
    role_hints: list[str],
    aliases: AliasRegistry,
    quote_prefix: str = "q",
) -> tuple[list[SegmentEx], dict[str, Any]]:
    """Run structured block-level re-attribution. Returns (segments, stats).

    Mutates `segments` in place (segment[i] corresponds to quotes[i])."""
    stats: dict[str, Any] = {
        "mode": "block_review", "blocks": 0, "subcalls": 0, "reviewed": 0,
        "corrected": 0, "confirmed": 0, "protected": 0, "failed": 0, "skipped_blocks": 0,
    }
    if not segments or not quotes or len(segments) != len(quotes):
        return segments, stats

    base_url = str(getattr(batch_config, "base_url", "") or "")
    api_key = str(getattr(batch_config, "api_key", "") or "")
    model = str(getattr(batch_config, "model", "") or "")
    timeout = int(getattr(batch_config, "timeout", 180) or 180)
    if not base_url or not model:
        return segments, stats

    narrator = (narrator or "").strip() or "叙述者"
    valid_names = set(role_hints) | {narrator}
    roster = _roster_text(role_hints, aliases)

    for block in _group_blocks(quotes):
        if len(block) < MIN_BLOCK:
            stats["skipped_blocks"] += 1
            continue
        stats["blocks"] += 1
        distinct = {
            aliases.canonicalize(segments[k].speaker)
            for k in block if segments[k].speaker not in SKIP_NAMES
        }
        protect_explicit = len(distinct) >= PROTECT_SCENE_SPEAKERS

        for ctx_idxs, new_idxs in _subchunks(block):
            window = _render_window(cleaned, quotes, ctx_idxs)
            prompt = _PROMPT.format(prefix=quote_prefix, roster=roster, narrator=narrator, block=window)
            try:
                out = _parse_json(_call_llm(base_url, api_key, model, prompt, timeout))
            except Exception as exc:  # noqa: BLE001 - failure-safe per block
                logger.warning("[block_review] block call failed: %s", exc)
                stats["failed"] += 1
                continue
            stats["subcalls"] += 1
            for k in new_idxs:
                seg = segments[k]
                raw = out.get(seg.quote_id)
                if not raw:
                    continue
                new = aliases.canonicalize(str(raw).strip())
                if new in SKIP_NAMES or new not in valid_names:
                    continue
                stats["reviewed"] += 1
                if protect_explicit and _is_explicit(seg):
                    stats["protected"] += 1
                    continue
                if new == seg.speaker:
                    stats["confirmed"] += 1
                    continue
                old = seg.speaker
                seg.speaker = new
                seg.attribution_type = AttributionType.IMPLICIT
                seg.confidence = min(float(seg.confidence or 0.8), 0.8)
                seg.evidence = f"块级结构化复核：轮次归属 {old}→{new}"
                stats["corrected"] += 1

    return segments, stats
