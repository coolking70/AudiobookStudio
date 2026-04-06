from typing import Any

from llm_client import OpenAICompatibleClient
from schemas import LLMConfig

VALID_INSTRUCT_ITEMS = {
    "american accent",
    "australian accent",
    "british accent",
    "canadian accent",
    "child",
    "chinese accent",
    "elderly",
    "female",
    "high pitch",
    "indian accent",
    "japanese accent",
    "korean accent",
    "low pitch",
    "male",
    "middle-aged",
    "moderate pitch",
    "portuguese accent",
    "russian accent",
    "teenager",
    "very high pitch",
    "very low pitch",
    "whisper",
    "young adult",
}

DEFAULT_SYSTEM_PROMPT = """
你是一个“小说/对白文本分析器”。
请把用户提供的原文分析成适合 TTS 的 JSON 结构。

要求：
1. 输出必须是合法 JSON 对象。
2. 格式固定为：
{
  "segments": [
    {
      "speaker": "旁白",
      "text": "夜色降临。",
      "emotion": "neutral",
      "style": "female, moderate pitch"
    }
  ]
}
3. speaker 只填角色名或“旁白”。
4. emotion 只能从以下值中选：neutral, soft, cold, serious, curious, angry
5. style 必须只使用以下英文标签，且用英文逗号+空格分隔：
american accent, australian accent, british accent, canadian accent,
child, chinese accent, elderly, female, high pitch, indian accent,
japanese accent, korean accent, low pitch, male, middle-aged,
moderate pitch, portuguese accent, russian accent, teenager,
very high pitch, very low pitch, whisper, young adult
6. 不要输出任何解释、注释、markdown，只输出 JSON。
7. 如果无法判断，说话内容归为“旁白”。
8. 优先按“自然段 / 完整对白回合”切分，不要切得过碎。
9. 旁白连续叙述可合并成 1-3 句，只要语义完整即可。
10. 只有当单段明显过长时，才继续往下细分。
""".strip()



def normalize_style(style: str | None) -> str | None:
    if not style:
        return None
    parts = [x.strip().lower() for x in style.split(",") if x.strip()]
    parts = [x for x in parts if x in VALID_INSTRUCT_ITEMS]
    if not parts:
        return None
    dedup = []
    for item in parts:
        if item not in dedup:
            dedup.append(item)
    return ", ".join(dedup)


def fallback_style_by_role(speaker: str, emotion: str) -> str:
    if speaker == "旁白":
        return "female, moderate pitch"
    if emotion in {"cold", "serious", "angry"}:
        return "male, low pitch"
    return "male, young adult, moderate pitch"


def sanitize_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = []
    for seg in segments:
        speaker = str(seg.get("speaker", "旁白")).strip() or "旁白"
        text = str(seg.get("text", "")).strip()
        emotion = str(seg.get("emotion", "neutral")).strip().lower() or "neutral"
        style = normalize_style(seg.get("style"))

        if emotion not in {"neutral", "soft", "cold", "serious", "curious", "angry"}:
            emotion = "neutral"

        if not text:
            continue

        if not style:
            style = fallback_style_by_role(speaker, emotion)

        cleaned.append(
            {
                "speaker": speaker,
                "text": text,
                "emotion": emotion,
                "style": style,
                "ref_audio": None,
                "ref_text": None,
            }
        )
    return cleaned


def analyze_text_with_llm(text: str, llm_config: LLMConfig) -> list[dict[str, Any]]:
    client = OpenAICompatibleClient(llm_config)

    system_prompt = llm_config.system_prompt.strip() if llm_config.system_prompt else DEFAULT_SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"请分析下面这段文本，并输出 JSON：\n\n{text}",
        },
    ]

    result = client.chat_json(messages)
    segments = result.get("segments", [])
    if not isinstance(segments, list):
        raise RuntimeError("LLM 返回的 segments 不是数组")
    return sanitize_segments(segments)
