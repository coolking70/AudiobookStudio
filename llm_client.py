import json
import logging
import time
from datetime import datetime
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
from output_layout import get_temp_archive_dir


class LLMContentFilterError(RuntimeError):
    """Raised when the provider finishes a request via content filtering."""

    def __init__(self, purpose: str, content: str = ""):
        self.purpose = purpose
        self.content = content
        preview = (content or "").strip()[:300]
        suffix = f" 返回片段：{preview}" if preview else ""
        super().__init__(f"LLM {purpose}触发服务端内容过滤（finish_reason=content_filter）。{suffix}")


class OpenAICompatibleClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    def _normalize_base_url(self) -> str:
        base = (getattr(self.config, "base_url", "") or "").strip().rstrip("/")
        if base.endswith("/chat/completions"):
            base = base[: -len("/chat/completions")].rstrip("/")
        if base.endswith("/v1"):
            return base
        return f"{base}/v1"

    def _chat_completions_url(self) -> str:
        base = (getattr(self.config, "base_url", "") or "").strip().rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{self._normalize_base_url()}/chat/completions"

    def _resolve_model_id(self) -> str:
        return (getattr(self.config, "local_model_path", None) or getattr(self.config, "model", "") or "").strip()

    def _use_direct_local_runtime(self) -> bool:
        return (getattr(self.config, "local_runtime", None) or "").strip().lower() == "direct"

    def _should_disable_reasoning(self) -> bool:
        base_url = (getattr(self.config, "base_url", "") or "").strip().lower()
        model = self._resolve_model_id().lower()
        is_lm_studio = "127.0.0.1:1234" in base_url or "localhost:1234" in base_url
        is_reasoning_model = (
            "qwen3.6" in model
            or "qwen3-6" in model
            or "gemma-4" in model
            or "gemma4" in model
        )
        return is_lm_studio and is_reasoning_model

    def _should_disable_siliconflow_qwen_thinking(self) -> bool:
        base_url = (getattr(self.config, "base_url", "") or "").strip().lower()
        model = self._resolve_model_id().lower()
        is_siliconflow = "siliconflow" in base_url or "api.siliconflow.cn" in base_url
        is_qwen3 = "qwen/qwen3" in model or model.startswith("qwen3")
        return is_siliconflow and is_qwen3

    def _apply_reasoning_options(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._should_disable_reasoning():
            payload["reasoning_effort"] = "none"
        if self._should_disable_siliconflow_qwen_thinking():
            payload["enable_thinking"] = False
        return payload

    def _is_bad_request_error(self, exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code == 400:
            return True
        response = getattr(exc, "response", None)
        return getattr(response, "status_code", None) == 400

    @staticmethod
    def _status_of(obj: Any) -> int | None:
        status = getattr(obj, "status_code", None)
        if status is None:
            status = getattr(getattr(obj, "response", None), "status_code", None)
        return status

    def _is_rate_limited(self, exc: Exception) -> bool:
        if self._status_of(exc) == 429:
            return True
        return "429" in str(exc) or "rate limit" in str(exc).lower()

    @staticmethod
    def _retry_after_seconds(obj: Any, attempt: int) -> float:
        # Honor a Retry-After header if the provider sends one; otherwise back off
        # exponentially (5s, 10s, 20s, 40s, ...) capped at 60s to ride out per-minute limits.
        headers = getattr(getattr(obj, "response", None), "headers", None) or getattr(obj, "headers", None)
        if headers:
            raw = headers.get("Retry-After") or headers.get("retry-after")
            try:
                if raw is not None:
                    return min(120.0, max(1.0, float(raw)))
            except (TypeError, ValueError):
                pass
        return min(60.0, 5.0 * (2 ** attempt))

    def _send_chat_request(self, payload: dict[str, Any], timeout: httpx.Timeout) -> Any:
        max_retries = int(getattr(self.config, "rate_limit_retries", 6) or 0)
        attempt = 0
        while True:
            try:
                response = self._send_chat_request_once(payload, timeout)
            except Exception as exc:  # noqa: BLE001 - re-raised below if not a rate limit
                if self._is_rate_limited(exc) and attempt < max_retries:
                    delay = self._retry_after_seconds(exc, attempt)
                    logging.warning("[llm] HTTP 429 限流，%.0fs 后重试（第 %d/%d 次）", delay, attempt + 1, max_retries)
                    time.sleep(delay)
                    attempt += 1
                    continue
                raise
            # httpx fallback path returns a Response without raising on 429.
            if self._status_of(response) == 429 and attempt < max_retries:
                delay = self._retry_after_seconds(response, attempt)
                logging.warning("[llm] HTTP 429 限流，%.0fs 后重试（第 %d/%d 次）", delay, attempt + 1, max_retries)
                time.sleep(delay)
                attempt += 1
                continue
            return response

    def _send_chat_request_once(self, payload: dict[str, Any], timeout: httpx.Timeout) -> Any:
        if OpenAI is not None:
            request_payload = dict(payload)
            extra_body: dict[str, Any] = {}
            for extra_key in ("enable_thinking", "reasoning_effort"):
                if extra_key in request_payload:
                    extra_body[extra_key] = request_payload.pop(extra_key)
            if extra_body:
                request_payload["extra_body"] = extra_body

            http_client = httpx.Client(timeout=timeout, trust_env=False)
            client = OpenAI(
                base_url=self._normalize_base_url(),
                api_key=getattr(self.config, "api_key", None) or "local",
                http_client=http_client,
            )
            try:
                return client.chat.completions.create(**request_payload)
            finally:
                http_client.close()

        url = self._chat_completions_url()
        headers = {
            "Authorization": f"Bearer {getattr(self.config, 'api_key', None) or 'local'}",
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

    def _dump_llm_debug(self, kind: str, payload: dict[str, Any], content: str, finish_reason: str | None) -> None:
        try:
            out_dir = get_temp_archive_dir("llm_debug")
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            target = out_dir / f"{stamp}_{kind}.json"
            target.write_text(json.dumps({
                "kind": kind,
                "finish_reason": finish_reason,
                "model": self._resolve_model_id(),
                "base_url": getattr(self.config, "base_url", None),
                "local_runtime": getattr(self.config, "local_runtime", None),
                "max_tokens": getattr(self.config, "max_tokens", None),
                "payload": payload,
                "content": content,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

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

    def chat_text(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        purpose: str = "文本分析",
    ) -> str:
        resolved_max_tokens = max(32, int(max_tokens or getattr(self.config, "max_tokens", None) or 512))
        if self._use_direct_local_runtime():
            runner = get_local_llm_runner(
                model_path=getattr(self.config, "local_model_path", None) or getattr(self.config, "model", None),
                engine=getattr(self.config, "local_engine", None),
                device=getattr(self.config, "local_device", None),
                ctx_tokens=getattr(self.config, "local_ctx_tokens", None),
                gpu_layers=getattr(self.config, "local_gpu_layers", None),
                threads=getattr(self.config, "local_threads", None),
                batch_size=getattr(self.config, "local_batch_size", None),
            )
            return runner.generate_text(
                messages,
                max_tokens=resolved_max_tokens,
                temperature=getattr(self.config, "temperature", 0.2),
            )

        payload = {
            "model": self._resolve_model_id(),
            "messages": messages,
            "temperature": getattr(self.config, "temperature", 0.2),
            "max_tokens": resolved_max_tokens,
        }
        self._apply_reasoning_options(payload)
        timeout = httpx.Timeout(connect=30.0, read=180.0, write=30.0, pool=30.0)
        try:
            response = self._send_chat_request(payload, timeout)
            if hasattr(response, "raise_for_status"):
                response.raise_for_status()
            content, finish_reason = self._extract_message_content(response)
        except Exception as exc:
            if APITimeoutError is not None and isinstance(exc, APITimeoutError):
                raise RuntimeError(f"LLM {purpose}超时。") from exc
            if APIConnectionError is not None and isinstance(exc, APIConnectionError):
                raise RuntimeError(f"LLM 服务在{purpose}时连接失败。") from exc
            if isinstance(exc, httpx.ReadTimeout):
                raise RuntimeError(f"LLM {purpose}超时。") from exc
            if isinstance(exc, httpx.RemoteProtocolError):
                raise RuntimeError(f"LLM 服务在{purpose}时主动断开连接。") from exc
            raise

        if finish_reason == "content_filter":
            self._dump_llm_debug("finish_reason_content_filter", payload, content, finish_reason)
            raise LLMContentFilterError(purpose, content)
        if finish_reason == "length":
            self._dump_llm_debug("finish_reason_length", payload, content, finish_reason)
            preview = content[:300]
            raise RuntimeError(
                f"LLM {purpose}输出被截断（finish_reason=length）。可稍微提高 max_tokens，"
                f"或减小单次分析文本块。返回片段：{preview}"
            )
        if not content:
            raise RuntimeError(f"LLM {purpose}返回为空（finish_reason={finish_reason or 'unknown'}）")
        return content

    def chat_json(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        if self._use_direct_local_runtime():
            runner = get_local_llm_runner(
                model_path=getattr(self.config, "local_model_path", None) or getattr(self.config, "model", None),
                engine=getattr(self.config, "local_engine", None),
                device=getattr(self.config, "local_device", None),
                ctx_tokens=getattr(self.config, "local_ctx_tokens", None),
                gpu_layers=getattr(self.config, "local_gpu_layers", None),
                threads=getattr(self.config, "local_threads", None),
                batch_size=getattr(self.config, "local_batch_size", None),
            )
            content = runner.generate_text(
                messages,
                max_tokens=getattr(self.config, "max_tokens", 512),
                temperature=getattr(self.config, "temperature", 0.2),
            )
            if not content:
                raise RuntimeError("LLM 返回为空")
            return self._parse_json_content(content)

        strict_payload = {
            "model": self._resolve_model_id(),
            "messages": messages,
            "temperature": getattr(self.config, "temperature", 0.2),
            "max_tokens": getattr(self.config, "max_tokens", 512),
            "response_format": {"type": "json_object"},
        }
        fallback_payload = {
            "model": self._resolve_model_id(),
            "messages": messages,
            "temperature": getattr(self.config, "temperature", 0.2),
            "max_tokens": getattr(self.config, "max_tokens", 512),
        }
        self._apply_reasoning_options(strict_payload)
        self._apply_reasoning_options(fallback_payload)

        timeout = httpx.Timeout(connect=30.0, read=300.0, write=60.0, pool=30.0)
        use_chat_compat = getattr(self.config, "compatibility_mode", "strict_json") == "chat_compat" or self._should_disable_reasoning()
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

        if finish_reason == "content_filter":
            self._dump_llm_debug("finish_reason_content_filter", fallback_payload if use_chat_compat else strict_payload, content, finish_reason)
            raise LLMContentFilterError("JSON 分析", content)
        if finish_reason == "length":
            self._dump_llm_debug("finish_reason_length", fallback_payload if use_chat_compat else strict_payload, content, finish_reason)
            preview = content[:300]
            raise RuntimeError(
                f"LLM 输出被截断（finish_reason=length）。可稍微提高 max_tokens，"
                f"或减小单次分析文本块。返回片段：{preview}"
            )
        if not content:
            raise RuntimeError(f"LLM 返回为空（finish_reason={finish_reason or 'unknown'}）")

        try:
            return self._parse_json_content(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM 没有返回合法 JSON：{content}") from exc
