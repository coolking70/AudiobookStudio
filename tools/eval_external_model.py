"""用任意 OpenAI 兼容模型跑完整 BatchLLM 流水线，在全部 11 个权威样本上对照评分。

评分与口径完全复用 run_regression（具名段 / acceptable 集合 / 简单≤2人·密集≥3人分桶 /
文本漂移防护），与 Agnes 基线（docs/regression/baseline_2026-06-10.json，总 92.86%）
直接可比。

用法：
  MODEL_BASE_URL=https://xxx/v1 MODEL_NAME=some-model MODEL_API_KEY=sk-... \\
    .venv/bin/python tools/eval_external_model.py [--only seg5,seg8] [--out result.json]

可选环境变量：MODEL_BATCH_SIZE(8) / MODEL_MAX_TOKENS(5000) / MODEL_TIMEOUT(240) /
MODEL_DISABLE_THINKING(1)。结果可用 run_regression compare 与基线出差值表。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "BookVoiceParser"))
sys.path.insert(0, str(REPO / "tools"))

from run_regression import (MANIFEST, SAMP, ROLE_HINTS, pick_segs,  # noqa: E402
                            print_table, save_summary, score_sample)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="逗号分隔样本子集，如 seg5,seg8（外部模型慢时建议先子集）")
    ap.add_argument("--out", help="评分 JSON 输出路径（可与基线 compare）")
    args = ap.parse_args()

    for v in ("MODEL_BASE_URL", "MODEL_NAME"):
        if not os.environ.get(v):
            raise SystemExit(f"请设置 {v}")
    from book_voice_parser import BatchConfig, parse_novel

    model = os.environ["MODEL_NAME"]
    cfg = BatchConfig(
        base_url=os.environ["MODEL_BASE_URL"], api_key=os.environ.get("MODEL_API_KEY", ""),
        model=model,
        batch_size=int(os.environ.get("MODEL_BATCH_SIZE", "8")),
        max_tokens=int(os.environ.get("MODEL_MAX_TOKENS", "5000")),
        temperature=0.0, timeout=int(os.environ.get("MODEL_TIMEOUT", "240")),
        context_chars=320, output_mode="compact",
        disable_thinking=os.environ.get("MODEL_DISABLE_THINKING", "1") == "1",
    )
    out_dir = REPO / "outputs/regression" / f"external-{model.replace('/', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"模型: {model} @ {cfg.base_url}\n", flush=True)

    results = []
    for s in pick_segs(args.only):
        raw = (SAMP / f"{s}_sample.txt").read_text(encoding="utf-8")
        print(f"[{s}] 解析 {len(raw)} 字（narrator={MANIFEST[s] or '无'}）…", flush=True)
        result = parse_novel(raw, role_hints=ROLE_HINTS, batch_llm_config=cfg,
                             narrator=MANIFEST[s], return_result=True,
                             include_narration=False, review_threshold=0.7,
                             enable_block_review=True)
        pp = out_dir / f"{s}_parse.json"
        pp.write_text(json.dumps(
            {"segments": [x.model_dump(mode="json") for x in result.segments],
             "stats": result.stats}, ensure_ascii=False, indent=1), encoding="utf-8")
        r = score_sample(s, pp)
        if r:
            results.append(r)
            print(f"  → 具名 {r['named']} 准确率 {r['acc']:.2%}（漂移跳过 {r['skipped_drift']}）", flush=True)

    print_table(results, f"external: {model}")
    print("\n对照 Agnes 基线: 总 92.86%（简单 95.0% / 密集 87.0%），"
          "compare docs/regression/baseline_2026-06-10.json 可出逐样本差值。")
    save_summary(results, Path(args.out) if args.out else out_dir / "summary.json",
                 {"mode": "external", "model": model, "base_url": cfg.base_url,
                  "time": datetime.now().isoformat(timespec="seconds")})


if __name__ == "__main__":
    main()
