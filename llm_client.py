import json
from typing import Any

import httpx

from schemas import LLMConfig


class OpenAICompatibleClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    def _normalize_base_url(self) -> str:
        base = self.config.base_url.strip().rstrip("/")
        if base.endswith("/v1"):
            return base
        return f"{base}/v1"

    def _send_chat_request(
        self,
        client: httpx.Client,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> httpx.Response:
        return client.post(url, headers=headers, json=payload)

    def _extract_json_snippet(self, content: str) -> str:
        content = content.strip()
        if not content:
            raise RuntimeError("LLM 返回为空")

        for start_char, end_char in (("{", "}"), ("[", "]")):
            start = content.find(start_char)
            if start == -1:
                continue
            depth = 0
            in_string = False
            escape = False
            for idx in range(start, len(content)):
                char = content[idx]
                if in_string:
                    if escape:
                        escape = False
                    elif char == "\\":
                        escape = True
                    elif char == '"':
                        in_string = False
                    continue

                if char == '"':
                    in_string = True
                elif char == start_char:
                    depth += 1
                elif char == end_char:
                    depth -= 1
                    if depth == 0:
                        return content[start:idx + 1]

        raise RuntimeError(f"LLM 没有返回可提取的 JSON：{content[:300]}")

    def _parse_json_content(self, content: str) -> dict[str, Any]:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            snippet = self._extract_json_snippet(content)
            parsed = json.loads(snippet)

        if isinstance(parsed, list):
            return {"segments": parsed}
        if not isinstance(parsed, dict):
            raise RuntimeError(f"LLM 返回的 JSON 不是对象：{type(parsed).__name__}")
        return parsed

    def chat_text(self, messages: list[dict[str, Any]]) -> str:
        url = f"{self._normalize_base_url()}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": min(self.config.max_tokens, 128),
        }

        timeout = httpx.Timeout(connect=30.0, read=60.0, write=30.0, pool=30.0)
        try:
            with httpx.Client(timeout=timeout, trust_env=False) as client:
                resp = self._send_chat_request(client, url, headers, payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.ReadTimeout as e:
            raise RuntimeError("LLM 连通性测试超时。") from e
        except httpx.RemoteProtocolError as e:
            raise RuntimeError("LLM 服务在连通性测试中主动断开连接。") from e

        content = data["choices"][0]["message"]["content"]
        if not content:
            raise RuntimeError("LLM 连通性测试返回为空")
        return content.strip()

    def chat_json(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        url = f"{self._normalize_base_url()}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        strict_payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
        }
        fallback_payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        timeout = httpx.Timeout(connect=30.0, read=300.0, write=60.0, pool=30.0)
        try:
            with httpx.Client(timeout=timeout, trust_env=False) as client:
                use_chat_compat = self.config.compatibility_mode == "chat_compat"
                try:
                    resp = self._send_chat_request(
                        client,
                        url,
                        headers,
                        fallback_payload if use_chat_compat else strict_payload,
                    )
                except httpx.RemoteProtocolError:
                    resp = self._send_chat_request(client, url, headers, fallback_payload)

                if not use_chat_compat and resp.status_code == 400 and "response_format.type" in resp.text:
                    resp = self._send_chat_request(client, url, headers, fallback_payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.ReadTimeout as e:
            raise RuntimeError("LLM 读取超时，请稍后重试，或进一步减小单次分析文本块。") from e
        except httpx.RemoteProtocolError as e:
            raise RuntimeError("LLM 服务主动断开了连接，请检查 LM Studio 当前模型是否支持该请求格式。") from e
        except httpx.HTTPStatusError:
            raise

        choice = data["choices"][0]
        content = choice["message"]["content"]
        if not content:
            raise RuntimeError("LLM 返回为空")

        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            preview = content[:300]
            raise RuntimeError(
                f"LLM 输出被截断（finish_reason=length）。可稍微提高 max_tokens，"
                f"或减小单次分析文本块。返回片段：{preview}"
            )

        try:
            return self._parse_json_content(content)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"LLM 没有返回合法 JSON：{content}") from e
