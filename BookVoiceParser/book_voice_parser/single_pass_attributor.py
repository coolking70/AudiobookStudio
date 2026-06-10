"""单遍直出归因器（实验，2026-06-10）：整文阅读一次性归因，替代分批窗口初判。

动机：全样本消融显示分批初判（batch=8，320 字窗口）仅 ~85%，而裸单遍直出在 seg6/7
实测 92–97%——差距来自分批问答只见局部窗口、丢失跨批语境。本模块把"读全文再判"
做成与 BatchLLMAttributor 鸭子类型兼容的归因器，仅替换流水线第①遍，下游
block_review / 称呼回查等全部不变。用 parse_novel(..., first_pass="single") 启用。

实现：引文按序分块（每块 ≤80 句），每次调用都附【完整章节文本】+ 该块的编号引文
清单，要求逐句输出 {"id","s","c"} 紧凑 JSON 行。
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from typing import Any

from .schema import Attribution, AttributionType, QuoteSpan


class SinglePassAttributor:
    def __init__(self, config: Any, full_text: str, chunk_size: int = 80):
        self.config = config
        self.full_text = full_text
        self.chunk_size = chunk_size

    # —— 与 BatchLLMAttributor.attribute 同形 ——
    def attribute(self, quotes: list[QuoteSpan], all_candidates: Any = None, *,
                  role_hints: list[str] | None = None, block_hints: Any = None,
                  narrator: str | None = None, narrator_hints: Any = None,
                  on_progress: Any = None) -> dict[str, Attribution]:
        out: dict[str, Attribution] = {}
        done = 0
        for k in range(0, len(quotes), self.chunk_size):
            chunk = quotes[k:k + self.chunk_size]
            ans = self._ask(chunk, role_hints or [], narrator)
            for q in chunk:
                a = ans.get(q.quote_id)
                out[q.quote_id] = Attribution(
                    quote_id=q.quote_id,
                    speaker=str(a.get("s", "未知")) if a else "未知",
                    confidence=float(a.get("c", 0.7)) if a else 0.3,
                    evidence="单遍直出" if a else "单遍直出无应答",
                    attribution_type=AttributionType.IMPLICIT,
                )
            done += len(chunk)
            if on_progress:
                on_progress(done, len(quotes))
        return out

    def _ask(self, chunk: list[QuoteSpan], role_hints: list[str], narrator: str | None) -> dict[str, dict]:
        roster = "、".join(role_hints) if role_hints else "（无已知角色表）"
        nar = (f'本文为{narrator}第一人称叙述（叙述中的"我"={narrator}）。' if narrator
               else "本文为第三人称叙述。")
        qlist = "\n".join(f'{q.quote_id}「{q.text[:40]}」' for q in chunk)
        prompt = f"""你是小说说话人标注专家。{nar}
已知角色：{roster}

请通读下面的完整原文，然后判断文末清单中每句对话是谁说出口的。
归属约定：回忆/引用某人说过的话→说话人=被引用者；想象某人会说的话→被想象者；
叙述中引用的概念词/书名（无人说出口）→"旁白"。

【原文】
{self.full_text}

【待标注对话清单】（按出现顺序）
{qlist}

对清单中每句输出一行紧凑JSON：{{"id":"qXXXX","s":"角色全名","c":0.x}}，不要输出其他内容。"""
        text = self._call(prompt, max_tokens=max(2000, len(chunk) * 40))
        ans: dict[str, dict] = {}
        for m in re.finditer(r'\{"id"\s*:\s*"(q\d+)"[^}]*\}', text or ""):
            try:
                obj = json.loads(m.group())
                ans[obj["id"]] = obj
            except Exception:  # noqa: BLE001
                continue
        return ans

    def _call(self, prompt: str, max_tokens: int, retries: int = 5) -> str | None:
        cfg = self.config
        base = (getattr(cfg, "base_url", "") or "").rstrip("/")
        url = base if base.endswith("/chat/completions") else base + "/chat/completions"
        payload = {"model": cfg.model, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": max_tokens, "temperature": 0.0}
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={
            "Authorization": f"Bearer {getattr(cfg, 'api_key', '') or 'local'}",
            "Content-Type": "application/json"})
        for t in range(retries):
            try:
                r = json.load(urllib.request.urlopen(req, timeout=max(300, getattr(cfg, "timeout", 240))))
                return r["choices"][0]["message"]["content"]
            except Exception as e:  # noqa: BLE001
                if "429" in str(e) or "timed out" in str(e).lower():
                    time.sleep(2 ** t)
                    continue
                return None
        return None
