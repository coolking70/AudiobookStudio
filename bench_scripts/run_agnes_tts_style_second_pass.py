from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
SAMPLES = REPO / "docs" / "samples"
OUT_ROOT = REPO / "bench_outputs" / "tts_style_second_pass"
AGNES_URL = "https://apihub.agnes-ai.com/v1/chat/completions"
AGNES_MODEL = "agnes-2.0-flash"


def load_dotenv_if_needed() -> None:
    if os.environ.get("AGNES_API_KEY"):
        return
    env_path = REPO / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def compact(value: object, limit: int = 180) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


def post_chat(messages: list[dict[str, str]], *, api_key: str, max_tokens: int, timeout: int) -> str:
    payload = {
        "model": AGNES_MODEL,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "enable_thinking": False,
    }
    req = urllib.request.Request(
        AGNES_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return str(message.get("content") or "").strip()


def parse_style_rows(text: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for line in str(text or "").splitlines():
        line = line.strip().strip("`")
        if not line or "|" not in line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 2:
            continue
        digits = re.sub(r"\D", "", parts[0])
        if not digits:
            continue
        style = parts[1]
        if style:
            out[int(digits)] = style
    return out


def build_prompt(items: list[dict[str, Any]], *, prompt_mode: str = "base") -> list[dict[str, str]]:
    system = (
        "你是中文有声书 TTS 表演指导。现在 speaker 已经人工固定，"
        "你只生成给 IndexTTS2/IndexTTS 使用的朗读语气描述，不判断说话人，不改写台词。"
    )
    if prompt_mode == "strict":
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
    else:
        lines = [
            "为每条台词生成一个短 tts_style。",
            "输出格式：序号|tts_style",
            "要求：",
            "- 12-28 个汉字左右，中文短语。",
            "- 包含语气/情绪，必要时包含语速、音量、停顿。",
            "- 不要写角色名，不要解释剧情，不要加英文标签。",
            "- 旁白/叙述段要自然平稳；短促惊呼要短促；疑问句要带疑问；吐槽/害羞/迟疑要体现但不要夸张。",
            "- 不确定时写：自然平稳，语速中等。",
            "",
        ]
    for item in items:
        lines.extend([
            f"--- {item['n']} ---",
            f"speaker: {item['speaker']}",
            f"前文: {compact(item.get('context_before'), 160)}",
            f"台词: {item['text']}",
            f"后文: {compact(item.get('context_after'), 120)}",
        ])
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(lines)},
    ]


def load_items(seg: str) -> list[dict[str, Any]]:
    parse = json.loads((SAMPLES / f"{seg}_parse.json").read_text(encoding="utf-8"))["segments"]
    gt = json.loads((SAMPLES / f"{seg}_groundtruth.json").read_text(encoding="utf-8"))["segments"]
    items: list[dict[str, Any]] = []
    for item in gt:
        i = int(item["i"])
        if i >= len(parse):
            continue
        source = parse[i]
        items.append({
            "i": i,
            "n": i + 1,
            "speaker": item.get("speaker") or source.get("speaker") or "旁白",
            "text": item.get("text") or source.get("text") or "",
            "context_before": source.get("context_before") or "",
            "context_after": source.get("context_after") or "",
        })
    return items


def run(seg: str, *, batch_size: int, timeout: int, max_tokens: int, prompt_mode: str) -> dict[str, Any]:
    api_key = os.environ["AGNES_API_KEY"]
    items = load_items(seg)
    styled: list[dict[str, Any]] = []
    raw_outputs: list[dict[str, Any]] = []
    started = time.perf_counter()
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        content = post_chat(build_prompt(batch, prompt_mode=prompt_mode), api_key=api_key, max_tokens=max_tokens, timeout=timeout)
        parsed = parse_style_rows(content)
        raw_outputs.append({"start": start, "count": len(batch), "content": content})
        for item in batch:
            style = parsed.get(int(item["n"]), "")
            styled.append({**item, "tts_style": style})
        print(f"[style] {min(start + batch_size, len(items))}/{len(items)}", flush=True)
    elapsed = round(time.perf_counter() - started, 2)
    missing = [item["n"] for item in styled if not item.get("tts_style")]
    return {
        "sample": seg,
        "model": AGNES_MODEL,
        "prompt_mode": prompt_mode,
        "elapsed_sec": elapsed,
        "items": len(styled),
        "style_count": sum(1 for item in styled if item.get("tts_style")),
        "missing": missing,
        "styled_segments": styled,
        "raw_outputs": raw_outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate IndexTTS style/instruct in a separate Agnes pass after speaker attribution is fixed.")
    parser.add_argument("--seg", default="muli4_seg8")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=2500)
    parser.add_argument("--prompt", choices=["base", "strict"], default="base")
    args = parser.parse_args()

    load_dotenv_if_needed()
    if not os.environ.get("AGNES_API_KEY"):
        raise SystemExit("AGNES_API_KEY is not set.")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUT_ROOT / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run(args.seg, batch_size=args.batch_size, timeout=args.timeout, max_tokens=args.max_tokens, prompt_mode=args.prompt)
    out_path = out_dir / f"{args.seg}_tts_style_second_pass_{args.prompt}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "sample": result["sample"],
        "model": result["model"],
        "prompt_mode": result["prompt_mode"],
        "elapsed_sec": result["elapsed_sec"],
        "items": result["items"],
        "style_count": result["style_count"],
        "coverage": round(result["style_count"] / max(1, result["items"]), 4),
        "missing": result["missing"],
        "output": str(out_path),
    }
    summary_path = out_dir / f"{args.seg}_tts_style_second_pass_{args.prompt}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
