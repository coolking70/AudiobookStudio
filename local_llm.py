from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from threading import Lock, RLock
from typing import Any

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:
    from mlx_lm import generate as mlx_generate
    from mlx_lm import load as mlx_load
    from mlx_lm.sample_utils import make_sampler as mlx_make_sampler
except ImportError:
    mlx_generate = None
    mlx_load = None
    mlx_make_sampler = None

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None


def get_local_llm_runtime_status() -> dict[str, object]:
    return {
        "transformers_ready": AutoModelForCausalLM is not None and AutoTokenizer is not None and torch is not None,
        "mlx_ready": mlx_load is not None and mlx_generate is not None,
        "gguf_ready": Llama is not None,
        "torch_available": torch is not None,
        "mps_available": bool(torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    }


_RUNNER_CACHE: dict[tuple[str, str, str, int | None, int | None, int | None, int | None], "LocalLLMRunner"] = {}
_RUNNER_CACHE_LOCK = Lock()


def get_local_llm_runner(
    model_path: str,
    engine: str | None = None,
    device: str | None = None,
    ctx_tokens: int | None = None,
    gpu_layers: int | None = None,
    threads: int | None = None,
    batch_size: int | None = None,
) -> "LocalLLMRunner":
    resolved_path = str(Path(model_path).expanduser())
    resolved_engine = (engine or LocalLLMRunner.detect_engine(resolved_path)).strip().lower()
    resolved_device = (device or LocalLLMRunner.detect_device()).strip().lower()
    key = (resolved_path, resolved_engine, resolved_device, ctx_tokens, gpu_layers, threads, batch_size)
    with _RUNNER_CACHE_LOCK:
        runner = _RUNNER_CACHE.get(key)
        if runner is None:
            runner = LocalLLMRunner(
                resolved_path,
                engine=resolved_engine,
                device=resolved_device,
                ctx_tokens=ctx_tokens,
                gpu_layers=gpu_layers,
                threads=threads,
                batch_size=batch_size,
            )
            _RUNNER_CACHE[key] = runner
        return runner


def unload_all_local_llm_runners() -> None:
    with _RUNNER_CACHE_LOCK:
        runners = list(_RUNNER_CACHE.values())
        _RUNNER_CACHE.clear()
    for runner in runners:
        runner.unload()


def get_local_llm_status() -> dict[str, object]:
    with _RUNNER_CACHE_LOCK:
        runners = list(_RUNNER_CACHE.values())
    loaded = [runner for runner in runners if runner.is_loaded()]
    return {
        "runtime": get_local_llm_runtime_status(),
        "loaded": bool(loaded),
        "loaded_models": [
            {
                "model_path": runner.model_path,
                "engine": runner.engine,
                "device": runner.device,
                "ctx_tokens": runner.ctx_tokens,
                "gpu_layers": runner.gpu_layers,
                "threads": runner.threads,
                "batch_size": runner.batch_size,
                "last_fallback": runner.last_fallback,
            }
            for runner in loaded
        ],
    }


class LocalLLMRunner:
    def __init__(
        self,
        model_path: str,
        engine: str | None = None,
        device: str | None = None,
        ctx_tokens: int | None = None,
        gpu_layers: int | None = None,
        threads: int | None = None,
        batch_size: int | None = None,
    ):
        self.model_path = str(model_path).strip()
        self.engine = (engine or self.detect_engine(self.model_path)).strip().lower()
        self.device = (device or self.detect_device()).strip().lower()
        self.ctx_tokens = ctx_tokens
        self.gpu_layers = gpu_layers
        self.threads = threads
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None
        self._lock = RLock()
        self.last_fallback: dict[str, Any] | None = None

    def _default_threads(self) -> int:
        return self.threads or max(1, min(8, (os.cpu_count() or 4) // 2 or 1))

    def _default_ctx_tokens(self) -> int:
        return self.ctx_tokens or 8192

    def _default_gpu_layers(self) -> int | None:
        if self.engine != "gguf":
            return self.gpu_layers
        return self.gpu_layers if self.gpu_layers is not None else (-1 if self.device == "mps" else 0)

    def _default_batch_size(self) -> int:
        if self.batch_size is not None:
            return self.batch_size
        if self.engine == "gguf" and self.device == "mps":
            model_size_gb = 0.0
            try:
                model_size_gb = Path(self.model_path).expanduser().stat().st_size / (1024 ** 3)
            except Exception:
                model_size_gb = 0.0
            if model_size_gb >= 12:
                return 128
            if model_size_gb >= 7:
                return 256
        return 512

    @staticmethod
    def detect_device() -> str:
        if torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def detect_engine(model_path: str) -> str:
        path = Path(model_path).expanduser()
        if path.is_file() and path.suffix.lower() == ".gguf":
            return "gguf"
        if not path.exists():
            raise RuntimeError(f"本地模型路径不存在：{path}")
        if path.is_file():
            raise RuntimeError("请填写模型目录路径，而不是单个文件。")

        config_path = path / "config.json"
        if not config_path.exists():
            gguf_files = [item for item in sorted(path.glob("*.gguf")) if "mmproj" not in item.name.lower()]
            if len(gguf_files) == 1:
                return "gguf"
            if gguf_files:
                raise RuntimeError("当前目录包含多个 GGUF 文件，请直接选择其中一个主模型文件。")
            raise RuntimeError("当前目录缺少 config.json，无法识别为可直接加载的模型目录。")

        config = json.loads(config_path.read_text(encoding="utf-8"))
        quantization = config.get("quantization") or config.get("quantization_config") or {}
        if isinstance(quantization, dict) and quantization.get("bits") and not quantization.get("quant_method"):
            return "mlx"
        return "transformers"

    def load(self) -> dict[str, str]:
        with self._lock:
            if self.model is not None and self.tokenizer is not None:
                return {
                    "engine": self.engine,
                    "device": self.device,
                    "model_path": self.model_path,
                    "ctx_tokens": str(self.ctx_tokens) if self.ctx_tokens is not None else "",
                    "gpu_layers": str(self.gpu_layers) if self.gpu_layers is not None else "",
                    "threads": str(self.threads) if self.threads is not None else "",
                    "batch_size": str(self.batch_size) if self.batch_size is not None else "",
                    "last_fallback": self.last_fallback or {},
                }

            if self.engine == "mlx":
                if mlx_load is None:
                    raise RuntimeError("当前环境未安装 mlx-lm，无法直接加载 MLX 模型。")
                self.model, self.tokenizer = mlx_load(self.model_path)
            elif self.engine == "gguf":
                if Llama is None:
                    raise RuntimeError("当前环境未安装 llama-cpp-python，无法直接加载 GGUF 模型。")
                self.model = Llama(
                    model_path=self.model_path,
                    n_ctx=self._default_ctx_tokens(),
                    n_gpu_layers=self._default_gpu_layers(),
                    n_threads=self._default_threads(),
                    n_batch=self._default_batch_size(),
                    verbose=False,
                )
                self.tokenizer = self.model
            elif self.engine == "transformers":
                if AutoModelForCausalLM is None or AutoTokenizer is None or torch is None:
                    raise RuntimeError("当前环境缺少 transformers 或 torch，无法直接加载本地模型。")
                torch_dtype = torch.float16 if self.device == "mps" else torch.float32
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                )
                self.model.to(self.device)
                self.model.eval()
            else:
                raise RuntimeError(f"不支持的本地 LLM 引擎：{self.engine}")

            return {
                "engine": self.engine,
                "device": self.device,
                "model_path": self.model_path,
                "ctx_tokens": str(self.ctx_tokens) if self.ctx_tokens is not None else "",
                "gpu_layers": str(self.gpu_layers) if self.gpu_layers is not None else "",
                "threads": str(self.threads) if self.threads is not None else "",
                "batch_size": str(self.batch_size) if self.batch_size is not None else "",
                "last_fallback": self.last_fallback or {},
            }

    def unload(self) -> None:
        with self._lock:
            self.model = None
            self.tokenizer = None
        gc.collect()
        if torch is not None and self.device == "mps" and hasattr(torch, "mps"):
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def _build_prompt(self, messages: list[dict[str, Any]]) -> str:
        if self.tokenizer is None:
            raise RuntimeError("本地模型尚未加载。")

        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        rendered = []
        for item in messages:
            role = str(item.get("role", "user"))
            content = str(item.get("content", ""))
            rendered.append(f"{role.upper()}:\n{content}")
        rendered.append("ASSISTANT:\n")
        return "\n\n".join(rendered)

    def generate_text(self, messages: list[dict[str, Any]], max_tokens: int, temperature: float) -> str:
        with self._lock:
            self.load()

            if self.engine == "mlx":
                prompt = self._build_prompt(messages)
                sampler = None
                if mlx_make_sampler is not None:
                    sampler = mlx_make_sampler(temp=max(0.0, float(temperature)))
                text = mlx_generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    verbose=False,
                    sampler=sampler,
                )
                return str(text).strip()

            if self.engine == "transformers":
                prompt = self._build_prompt(messages)
                encoded = self.tokenizer(prompt, return_tensors="pt")
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                do_sample = float(temperature) > 0
                outputs = self.model.generate(
                    **encoded,
                    max_new_tokens=max_tokens,
                    do_sample=do_sample,
                    temperature=max(0.01, float(temperature)) if do_sample else None,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                generated = outputs[0][encoded["input_ids"].shape[1]:]
                return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

            if self.engine == "gguf":
                try:
                    response = self.model.create_chat_completion(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=max(0.0, float(temperature)),
                    )
                    choice = (response.get("choices") or [{}])[0]
                    message = choice.get("message") or {}
                    return str(message.get("content") or "").strip()
                except RuntimeError as exc:
                    message = str(exc)
                    if "llama_decode returned -3" not in message:
                        raise
                    current_batch = self._default_batch_size()
                    if current_batch <= 64:
                        raise RuntimeError(
                            "GGUF 推理失败：当前模型在现有上下文长度和 batch_size 下无法稳定解码。"
                            "建议降低 Batch Size、GPU Layers，或切换到更保守的智能推荐档位。"
                        ) from exc
                    next_batch = max(64, current_batch // 2)
                    self.last_fallback = {
                        "reason": "llama_decode returned -3",
                        "from_batch_size": current_batch,
                        "to_batch_size": next_batch,
                    }
                    self.batch_size = next_batch
                    self.unload()
                    self.load()
                    response = self.model.create_chat_completion(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=max(0.0, float(temperature)),
                    )
                    choice = (response.get("choices") or [{}])[0]
                    message_obj = choice.get("message") or {}
                    return str(message_obj.get("content") or "").strip()

            raise RuntimeError(f"不支持的本地 LLM 引擎：{self.engine}")
