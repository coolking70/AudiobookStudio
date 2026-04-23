import json
from typing import Any

import httpx

try:
    from openai import APIConnectionError, APITimeoutError, OpenAI
except ImportError:
    APIConnectionError = None
    APITimeoutError = None
    OpenAI = None

from schemas import LLMConfig
from local_llm import get_local_llm_runner


class OpenAICompatibleClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    def _normalize_base_url(self) -> str:
        base = self.config.base_url.strip().rstrip("/")
        if base.endswith("/v1"):
            return base
        return f"{base}/v1"

    def _resolve_model_id(self) -> str:
        return (self.config.local_model_path or self.config.model or "").strip()

    def _use_direct_local_runtime(self) -> bool:
        return (self.config.local_runtime or "").strip().lower() == "direct"

    def _should_disable_reasoning(self) -> bool:
        base_url = (self.config.base_url or "").strip().lower()
        model = self._resolve_model_id().lower()
        is_lm_studio = "127.0.0.1:1234" in base_url or "localhost:1234" in base_url
        is_reasoning_qwen = "qwen3.6" in model or "qwen3-6" in model
        return is_lm_studio and is_reasoning_qwen

    def _apply_reasoning_options(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._should_disable_reasoning():
            payload["reasoning_effort"] = "none"
        return payload

    def _is_bad_request_error(self, exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code == 400:
            return True
        response = getattr(exc, "response", None)
        return getattr(response, "status_code", None) == 400

    def _send_chat_request(self, payload: dict[str, Any], timeout: httpx.Timeout) -> Any:
        if OpenAI is not None:
            http_client = httpx.Client(timeout=timeout, trust_env=False)
            client = OpenAI(
                base_url=self._normalize_base_url(),
                api_key=self.config.api_key or "local",
                http_client=http_client,
            )
            try:
                return client.chat.completions.create(**payload)
            finally:
                http_client.close()

        url = f"{self._normalize_base_url()}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.api_key or 'local'}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            return client.post(url, headers=headers, json=payload)

    def _extract_message_content(self, response: Any) -> tuple[str, str | None]:
        if hasattr(response, "choices"):
            choice = response.choices[0]
            message = getattr(choice, "message", None)
            content = getattr(message, "content", "")
            if isinstance(content, list):
                content = "".join(
                    item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "")
                    for item in content
                )
            finish_reason = getattr(choice, "finish_reason", None)
            return (content or "").strip(), finish_reason

        data = response.json()
        choice = data["choices"][0]
        content = choice["message"]["content"]
        if isinstance(content, list):
            content = "".join(item.get("text", "") for item in content if isinstance(item, dict))
        return (content or "").strip(), choice.get("finish_reason")

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
        if self._use_direct_local_runtime():
            runner = get_local_llm_runner(
                model_path=self.config.local_model_path or self.config.model,
                engine=self.config.local_engine,
                device=self.config.local_device,
                ctx_tokens=self.config.local_ctx_tokens,
                gpu_layers=self.config.local_gpu_layers,
                threads=self.config.local_threads,
                batch_size=self.config.local_batch_size,
            )
            return runner.generate_text(messages, max_tokens=min(self.config.max_tokens, 128), temperature=self.config.temperature)

        payload = {
            "model": self._resolve_model_id(),
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": min(max(self.config.max_tokens, 256), 512),
        }
        self._apply_reasoning_options(payload)
        timeout = httpx.Timeout(connect=30.0, read=180.0, write=30.0, pool=30.0)
        try:
            response = self._send_chat_request(payload, timeout)
            if hasattr(response, "raise_for_status"):
                response.raise_for_status()
            content, _ = self._extract_message_content(response)
        except Exception as exc:
            if APITimeoutError is not None and isinstance(exc, APITimeoutError):
                raise RuntimeError("LLM 连通性测试超时。") from exc
            if APIConnectionError is not None and isinstance(exc, APIConnectionError):
                raise RuntimeError("LLM 服务在连通性测试中连接失败。") from exc
            if isinstance(exc, httpx.ReadTimeout):
                raise RuntimeError("LLM 连通性测试超时。") from exc
            if isinstance(exc, httpx.RemoteProtocolError):
                raise RuntimeError("LLM 服务在连通性测试中主动断开连接。") from exc
            raise

        if not content:
            raise RuntimeError("LLM 连通性测试返回为空")
        return content

    def chat_json(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        if self._use_direct_local_runtime():
            runner = get_local_llm_runner(
                model_path=self.config.local_model_path or self.config.model,
                engine=self.config.local_engine,
                device=self.config.local_device,
                ctx_tokens=self.config.local_ctx_tokens,
                gpu_layers=self.config.local_gpu_layers,
                threads=self.config.local_threads,
                batch_size=self.config.local_batch_size,
            )
            content = runner.generate_text(messages, max_tokens=self.config.max_tokens, temperature=self.config.temperature)
            if not content:
                raise RuntimeError("LLM 返回为空")
            return self._parse_json_content(content)

        strict_payload = {
            "model": self._resolve_model_id(),
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
        }
        fallback_payload = {
            "model": self._resolve_model_id(),
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        self._apply_reasoning_options(strict_payload)
        self._apply_reasoning_options(fallback_payload)

        timeout = httpx.Timeout(connect=30.0, read=300.0, write=60.0, pool=30.0)
        use_chat_compat = self.config.compatibility_mode == "chat_compat" or self._should_disable_reasoning()
        try:
            try:
                response = self._send_chat_request(
                    fallback_payload if use_chat_compat else strict_payload,
                    timeout,
                )
                if hasattr(response, "raise_for_status"):
                    response.raise_for_status()
            except Exception as exc:
                error_text = str(exc)
                if not use_chat_compat and (
                    "response_format.type" in error_text
                    or "json_object" in error_text
                    or self._is_bad_request_error(exc)
                ):
                    response = self._send_chat_request(fallback_payload, timeout)
                    if hasattr(response, "raise_for_status"):
                        response.raise_for_status()
                else:
                    raise
            content, finish_reason = self._extract_message_content(response)
        except Exception as exc:
            if APITimeoutError is not None and isinstance(exc, APITimeoutError):
                raise RuntimeError("LLM 读取超时，请稍后重试，或进一步减小单次分析文本块。") from exc
            if APIConnectionError is not None and isinstance(exc, APIConnectionError):
                raise RuntimeError("LLM 服务连接失败，请检查本地 OpenAI 兼容服务是否已启动。") from exc
            if isinstance(exc, httpx.ReadTimeout):
                raise RuntimeError("LLM 读取超时，请稍后重试，或进一步减小单次分析文本块。") from exc
            if isinstance(exc, httpx.RemoteProtocolError):
                raise RuntimeError("LLM 服务主动断开了连接，请检查当前模型是否支持该请求格式。") from exc
            raise

        if not content:
            raise RuntimeError("LLM 返回为空")

        if finish_reason == "length":
            preview = content[:300]
            raise RuntimeError(
                f"LLM 输出被截断（finish_reason=length）。可稍微提高 max_tokens，"
                f"或减小单次分析文本块。返回片段：{preview}"
            )

        try:
            return self._parse_json_content(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM 没有返回合法 JSON：{content}") from exc
