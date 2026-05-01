from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol
from urllib import error, request

from .schema import Attribution, AttributionType, CandidateSet, QuoteSpan


UNKNOWN_LABEL = "未知"
LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class SPCCandidate:
    label: str
    name: str


@dataclass(frozen=True)
class SPCTask:
    quote_id: str
    quote_text: str
    context_before: str
    context_after: str
    candidates: list[SPCCandidate]
    recent_speakers: list[str]


class CandidateRanker(Protocol):
    def rank(
        self,
        quote: QuoteSpan,
        candidates: CandidateSet,
        recent_speakers: list[str] | None = None,
    ) -> Attribution:
        ...


def _candidate_names(candidates: CandidateSet, limit: int = 8) -> list[str]:
    names = [name for name in candidates.candidates if name not in {"旁白"}]
    if UNKNOWN_LABEL not in names:
        names.append(UNKNOWN_LABEL)
    return names[:limit]


def _trim_after_context(text: str, max_chars: int) -> str:
    trimmed = text[:max_chars]
    second_quote = trimmed.find("「", 1)
    if second_quote != -1:
        trimmed = trimmed[:second_quote]
    return trimmed


def build_spc_task(
    quote: QuoteSpan,
    candidates: CandidateSet,
    recent_speakers: list[str] | None = None,
    context_chars: int = 220,
) -> SPCTask:
    names = _candidate_names(candidates, limit=len(LABELS))
    return SPCTask(
        quote_id=quote.quote_id,
        quote_text=quote.text,
        context_before=quote.context_before[-context_chars:],
        context_after=_trim_after_context(quote.context_after, context_chars),
        candidates=[SPCCandidate(label=LABELS[idx], name=name) for idx, name in enumerate(names)],
        recent_speakers=list(recent_speakers or [])[-4:],
    )


def spc_task_to_dict(task: SPCTask, gold_speaker: str | None = None) -> dict[str, Any]:
    data = {
        "quote_id": task.quote_id,
        "quote_text": task.quote_text,
        "context_before": task.context_before,
        "context_after": task.context_after,
        "recent_speakers": task.recent_speakers,
        "candidates": [{"label": item.label, "name": item.name} for item in task.candidates],
    }
    if gold_speaker is not None:
        data["gold_speaker"] = gold_speaker
        data["gold_label"] = next((item.label for item in task.candidates if item.name == gold_speaker), None)
    return data


def build_spc_prompt(task: SPCTask) -> list[dict[str, str]]:
    options = "\n".join(f"{item.label}. {item.name}" for item in task.candidates)
    user = (
        "请判断当前小说台词是谁说的。只能从候选项中选择一个标签。\n\n"
        "注意：下文中的显式说话人只说明下文那句是谁说的，不能直接反推当前台词。\n"
        "连续无提示对话常见 ABAB 轮替，但必须结合上文动作和在场人物判断。\n\n"
        f"上文：\n{task.context_before or '（无）'}\n\n"
        f"当前台词：\n「{task.quote_text}」\n\n"
        f"下文：\n{task.context_after or '（无）'}\n\n"
        f"最近已确定说话人：{', '.join(task.recent_speakers) if task.recent_speakers else '（无）'}\n\n"
        f"候选：\n{options}\n\n"
        "请只返回一行 JSON，不要换行，不要解释：{\"choice\":\"A\",\"confidence\":0.0到1.0,\"evidence\":\"20字以内\"}"
    )
    return [
        {
            "role": "system",
            "content": "你是中文小说引语说话人识别器，任务是做候选分类，不要生成候选外角色。",
        },
        {"role": "user", "content": user},
    ]


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _extract_json(text: str) -> dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("LLM returned empty content")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        choice = re.search(r'"?choice"?\s*:\s*"?([A-Z])', text)
        confidence = re.search(r'"?confidence"?\s*:\s*([01](?:\.\d+)?)', text)
        if choice:
            return {
                "choice": choice.group(1),
                "confidence": float(confidence.group(1)) if confidence else 0.5,
                "evidence": "partial JSON fallback",
            }
        raise


class OpenAICompatibleSPCRanker:
    def __init__(self, llm_config: Any) -> None:
        self.llm_config = llm_config

    def _chat(self, messages: list[dict[str, str]]) -> str:
        base_url = str(_config_value(self.llm_config, "base_url", "")).rstrip("/")
        if not base_url:
            raise RuntimeError("llm_config.base_url is required for llm_spc")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        model = _config_value(self.llm_config, "model", None) or _config_value(self.llm_config, "local_model_path", "")
        payload = {
            "model": model,
            "messages": messages,
            "temperature": float(_config_value(self.llm_config, "temperature", 0.0) or 0.0),
            # Qwen3 系列是 thinking 模型，reasoning tokens 约 700，需留足空间；
            # 其他模型 512 足够，取两者较大值以兼容。
            "max_tokens": max(min(int(_config_value(self.llm_config, "max_tokens", 1024) or 1024), 4096), 1024),
            # 禁用 Qwen3/DeepSeek 等模型的扩展思考模式（LM Studio 兼容参数）
            # 注意：reasoning_effort 是 OpenAI o 系列专用字段，会导致 LM Studio 返回 422
            "enable_thinking": False,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(
            f"{base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {_config_value(self.llm_config, 'api_key', 'local') or 'local'}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=180) as response:
                data = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            body_text = ""
            try:
                body_text = exc.read().decode("utf-8", errors="replace")[:300]
            except Exception:
                pass
            raise RuntimeError(f"HTTP {exc.code} from LLM: {body_text}") from exc
        content = data["choices"][0]["message"]["content"]
        if isinstance(content, list):
            return "".join(item.get("text", "") for item in content if isinstance(item, dict))
        # 剥离思维链（Qwen3 在禁用 thinking 失败时仍可能输出 <think>...</think>）
        text = re.sub(r"<think>.*?</think>", "", str(content), flags=re.DOTALL)
        text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
        return text.strip()

    def rank(
        self,
        quote: QuoteSpan,
        candidates: CandidateSet,
        recent_speakers: list[str] | None = None,
    ) -> Attribution:
        task = build_spc_task(quote, candidates, recent_speakers=recent_speakers)
        label_to_name = {item.label: item.name for item in task.candidates}
        parsed = _extract_json(self._chat(build_spc_prompt(task)))
        choice = str(parsed.get("choice") or "").strip().upper()
        speaker = label_to_name.get(choice, UNKNOWN_LABEL)
        confidence = float(parsed.get("confidence") or 0.5)
        evidence = str(parsed.get("evidence") or "SPC LLM candidate ranking")
        return Attribution(
            quote_id=quote.quote_id,
            speaker=speaker,
            confidence=max(0.0, min(1.0, confidence)),
            evidence=evidence,
            attribution_type=AttributionType.IMPLICIT if speaker != UNKNOWN_LABEL else AttributionType.UNKNOWN,
        )
