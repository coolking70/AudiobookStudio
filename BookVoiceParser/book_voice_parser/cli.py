from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .parser import parse_novel


def _load_role_hints(path: Path | None) -> dict[str, list[str] | str] | list[str] | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, (dict, list)):
        raise SystemExit("role hints must be a JSON object or array")
    return data


def _write_output(data: Any, output: Path | None, jsonl: bool) -> None:
    if jsonl:
        text = "\n".join(json.dumps(item, ensure_ascii=False) for item in data) + "\n"
    else:
        text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if output is None:
        print(text, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse Chinese novel dialogue into speaker-attributed segments.")
    parser.add_argument("input", type=Path, help="Input txt/md file")
    parser.add_argument("-o", "--output", type=Path, help="Output JSON/JSONL path")
    parser.add_argument("--role-hints", type=Path, help="JSON file: ['张三'] or {'张三': ['三哥']}")
    parser.add_argument("--jsonl", action="store_true", help="Write segments as JSONL instead of a ParseResult JSON object")
    parser.add_argument("--review-threshold", type=float, default=0.7)
    parser.add_argument("--enable-hanlp", action="store_true", help="Use HanLP NER for candidate generation")
    parser.add_argument("--hanlp-model", help="HanLP NER model name or URL. Defaults to MSRA_NER_ELECTRA_SMALL_ZH")
    parser.add_argument("--no-alias-inference", action="store_true", help="Disable local title/surname alias inference")
    parser.add_argument("--implicit-strategy", choices=["heuristic", "llm_spc"], default="heuristic")
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:1234", help="OpenAI-compatible base URL")
    parser.add_argument("--llm-model", default="qwen/qwen3.6-35b-a3b", help="Model id for llm_spc")
    parser.add_argument("--llm-api-key", default="local")
    args = parser.parse_args()

    text = args.input.read_text(encoding="utf-8")
    llm_config = None
    if args.implicit_strategy == "llm_spc":
        llm_config = SimpleNamespace(
            base_url=args.llm_base_url,
            model=args.llm_model,
            api_key=args.llm_api_key,
            temperature=0.0,
            max_tokens=256,
        )
    result = parse_novel(
        text,
        role_hints=_load_role_hints(args.role_hints),
        llm_config=llm_config,
        review_threshold=args.review_threshold,
        use_hanlp=args.enable_hanlp,
        hanlp_model=args.hanlp_model,
        enable_alias_inference=not args.no_alias_inference,
        implicit_strategy=args.implicit_strategy,
        return_result=True,
    )
    payload = [item.model_dump(mode="json") for item in result.segments] if args.jsonl else result.model_dump(mode="json")
    _write_output(payload, args.output, args.jsonl)


if __name__ == "__main__":
    main()
