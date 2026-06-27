"""Generate compact IndexTTS style/instruct text for fixed speaker segments."""
from __future__ import annotations

import re
from typing import Any

from llm_client import OpenAICompatibleClient
from schemas import LLMConfig


AGNES_BASE_URL = "https://apihub.agnes-ai.com/v1"
AGNES_MODEL = "agnes-2.0-flash"


def compact_context(value: object, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def resolve_style_llm_config(llm: LLMConfig | None, agnes_api_key: str | None) -> LLMConfig:
    if llm is not None:
        return llm.model_copy(update={
            "temperature": 0.0,
            "max_tokens": max(512, int(llm.max_tokens or 2000)),
        })
    key = str(agnes_api_key or "").strip()
    if not key:
        raise RuntimeError("缺少 TTS 语气生成模型配置；请填写文本模型配置，或在服务器环境设置 AGNES_API_KEY。")
    return LLMConfig(
        base_url=AGNES_BASE_URL,
        api_key=key,
        model=AGNES_MODEL,
        temperature=0.0,
        max_tokens=2500,
        compatibility_mode="chat_compat",
    )


def build_style_prompt(items: list[dict[str, Any]]) -> list[dict[str, str]]:
    system = (
        "你是中文有声书 TTS 表演指导。现在 speaker 已经固定，"
        "你只生成给 IndexTTS2/IndexTTS 使用的朗读语气描述，不判断说话人，不改写台词。"
    )
    lines = [
        "为每条台词生成一个可直接传给 IndexTTS 的 tts_style。",
        "输出格式：序号|tts_style",
        "硬性要求：",
        "- 只允许描写声音表现：语气、情绪、语速、音量、停顿、重音。",
        "- 禁止出现：角色名、剧情解释、人物性格分析、动作描述、原因解释、听感评价。",
        "- 长度 8-22 个汉字，最多 3 个短语，用逗号分隔。",
        "- 短台词只能输出短指令，不得扩写后续心理。",
        "- 旁白/叙述段要自然平稳；短促惊呼要短促；疑问句要带疑问；吐槽/害羞/迟疑要体现但不要夸张。",
        "- 不确定时写：自然平稳，语速中等。",
        "",
        "反例与修正：",
        "错误：体现角色的原则性与冷淡",
        "正确：严肃直接，语速平稳，略冷淡",
        "错误：模拟短信内容，语速中等",
        "正确：平淡直白，语速中等",
        "错误：随即被推开的动作感",
        "正确：慌乱大声，语速很快",
        "",
    ]
    for item in items:
        lines.extend([
            f"--- {item['n']} ---",
            f"speaker: {item.get('speaker') or '旁白'}",
            f"前文: {compact_context(item.get('context_before'), 160)}",
            f"台词: {item.get('text') or ''}",
            f"后文: {compact_context(item.get('context_after'), 120)}",
        ])
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(lines)},
    ]


def parse_style_rows(text: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for line in str(text or "").splitlines():
        cleaned = line.strip().strip("`")
        if not cleaned:
            continue
        cleaned = cleaned.replace("｜", "|")
        if "|" not in cleaned:
            continue
        parts = [part.strip() for part in cleaned.split("|")]
        if len(parts) < 2:
            continue
        digits = re.sub(r"\D", "", parts[0])
        if not digits:
            continue
        style = normalize_tts_style(parts[1])
        if style:
            out[int(digits)] = style
    return out


def normalize_tts_style(style: object) -> str:
    text = "，".join(part.strip() for part in str(style or "").replace("；", "，").split("，") if part.strip())
    replacements = {
        "解释说明": "柔和说明",
        "陈述事实": "平和陈述",
        "内心独白": "轻声感慨",
        "模拟语气": "平淡直白",
        "模仿语气": "平淡直白",
        "动作感": "",
        "剧情": "",
        "体现": "",
        "表现出": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[A-Za-z]+", "", text)
    text = re.sub(r"[，,]{2,}", "，", text).strip("，,。；; ")
    if not text:
        return "自然平稳，语速中等"
    # Keep complete comma-separated phrases where possible.
    if len(text) > 28:
        parts = [part for part in text.split("，") if part]
        kept: list[str] = []
        total = 0
        for part in parts:
            next_total = total + len(part) + (1 if kept else 0)
            if kept and next_total > 28:
                break
            kept.append(part)
            total = next_total
        text = "，".join(kept) if kept else text[:28].rstrip("，,。；; ")
    return text


def build_style_items(segments: list[dict[str, Any]], start: int, end: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for index in range(start, end):
        segment = segments[index]
        before = segment.get("context_before")
        after = segment.get("context_after")
        if not before:
            before = "\n".join(str(s.get("text") or "") for s in segments[max(0, index - 3):index])
        if not after:
            after = "\n".join(str(s.get("text") or "") for s in segments[index + 1:min(len(segments), index + 4)])
        items.append({
            "n": index + 1,
            "index": index,
            "speaker": segment.get("speaker") or "旁白",
            "text": segment.get("text") or "",
            "context_before": before or "",
            "context_after": after or "",
        })
    return items


def generate_style_batch(
    client: OpenAICompatibleClient,
    items: list[dict[str, Any]],
    *,
    max_tokens: int,
) -> dict[int, str]:
    content = client.chat_text(
        build_style_prompt(items),
        max_tokens=max_tokens,
        purpose="IndexTTS语气描述生成",
    )
    parsed = parse_style_rows(content)
    if not parsed and len(items) == 1:
        single = normalize_tts_style(content.splitlines()[0] if content.splitlines() else content)
        if single:
            return {int(items[0]["n"]): single}
    missing = [int(item["n"]) for item in items if int(item["n"]) not in parsed]
    if missing:
        for item in items:
            n = int(item["n"])
            parsed.setdefault(n, "自然平稳，语速中等")
    return parsed
