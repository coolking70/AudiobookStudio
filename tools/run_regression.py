"""全样本回归跑分器：任何流水线改动 → 在全部权威样本上重跑/评分 → 分桶对比。

三种模式：
  score    给现有 parse 评分（不调 API）。默认评 docs/samples 下已提交的 parse，
           也可用 --parse-dir 指向一批新 parse。
  run      用当前流水线代码重新解析每个样本（调 Agnes API，需 `source .env`），
           parse 存到 outputs/regression/<tag>/，随即评分。
  compare  对比两份评分 JSON，输出逐样本/分桶的差值表。

评分口径（与 groundtruth 固化口径一致）：
  - 只评具名段（最终真值为 群众·*/旁白/未知* 的段不计分）；
  - 命中 acceptable 集合内任一即算对；
  - 按 ±4 窗口内不同具名说话人数分桶：简单 ≤2 人 / 密集 ≥3 人；
  - 逐段按索引对齐，文本归一化不一致的段跳过（防引文抽取漂移），并报告覆盖率。

典型流程：
  .venv/bin/python tools/run_regression.py score --out docs/regression/baseline.json
  # ……改流水线代码……
  source .env && .venv/bin/python tools/run_regression.py run --tag my-change
  .venv/bin/python tools/run_regression.py compare docs/regression/baseline.json \\
      outputs/regression/my-change/summary.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SAMP = REPO / "docs/samples"
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "BookVoiceParser"))

from evaluate_agnes_bookmark_review import ROLE_HINTS  # noqa: E402

# 各样本的叙述者（run 模式重解析时用；score 模式不需要）
MANIFEST: dict[str, str | None] = {
    "muli4_seg1": "甘织玲奈子",
    "muli4_seg2": "甘织玲奈子",
    "muli4_seg3": "甘织玲奈子",
    "muli4_seg4": "甘织玲奈子",
    "muli4_seg5": "甘织玲奈子",
    "muli4_seg6": "甘织玲奈子",
    "muli4_seg7": "甘织玲奈子",
    "muli4_seg7b_side_kaho": "小柳香穗",
    "muli4_seg8": "甘织玲奈子",
    "muli4_seg8b_side_mayu": None,  # 第三人称
    "muli4_seg9": "甘织玲奈子",
}

_A2C: dict[str, str] = {}
for _c, _al in ROLE_HINTS.items():
    _names = _al if isinstance(_al, (list, tuple)) else _al.get("aliases", [])
    for _a in _names:
        _A2C[_a] = _c


def canon(x: str) -> str:
    return _A2C.get(x or "", x or "")


def is_crowd(x: str) -> bool:
    x = x or ""
    return x.startswith(("群众·", "厕所女生")) or x in {"未知", "未知临时人物", "旁白", "其他", ""}


def norm(s: str) -> str:
    s = re.sub(r"[「」『』《》〈〉]", "", str(s or ""))
    return "".join(s.split()).replace("彷佛", "仿佛").replace("姊", "姐")


def scene_size(truths: list[str], i: int, w: int = 4) -> int:
    return len({truths[j] for j in range(max(0, i - w), min(len(truths), i + w + 1))
                if not is_crowd(truths[j])})


def score_sample(seg: str, parse_path: Path) -> dict | None:
    gt_p = SAMP / f"{seg}_groundtruth.json"
    if not gt_p.exists() or not parse_path.exists():
        return None
    gt = json.loads(gt_p.read_text(encoding="utf-8"))
    gsegs = gt["segments"]
    psegs = json.loads(parse_path.read_text(encoding="utf-8"))["segments"]
    truths = [g["speaker"] for g in gsegs]

    simple = [0, 0]
    dense = [0, 0]
    skipped = 0
    for g in gsegs:
        i = g["i"]
        if is_crowd(g["speaker"]):
            continue
        if i >= len(psegs) or norm(psegs[i].get("text", "")) != norm(g["text"]):
            skipped += 1  # 引文抽取漂移，无法逐段对齐
            continue
        acc = {canon(x) for x in (g.get("acceptable") or [g["speaker"]])}
        acc.add(canon(g["speaker"]))
        ok = canon(psegs[i].get("speaker", "")) in acc
        b = simple if scene_size(truths, i) <= 2 else dense
        b[1] += 1
        b[0] += int(ok)

    named = simple[1] + dense[1]
    correct = simple[0] + dense[0]
    return {
        "seg": seg,
        "named": named,
        "correct": correct,
        "acc": round(correct / named, 4) if named else None,
        "simple": simple,
        "dense": dense,
        "skipped_drift": skipped,
    }


def fmt_bucket(b: list[int]) -> str:
    return f"{b[0]}/{b[1]}={b[0] / b[1]:.1%}" if b[1] else "—"


def print_table(results: list[dict], title: str) -> None:
    print(f"\n== {title} ==")
    print(f"{'样本':<24} {'具名':>5} {'准确率':>8}  {'简单≤2人':>14}  {'密集≥3人':>14}  漂移跳过")
    ts = [0, 0]
    td = [0, 0]
    skip = 0
    for r in results:
        print(f"{r['seg']:<24} {r['named']:>5} {r['acc']:>8.2%}  {fmt_bucket(r['simple']):>14}  "
              f"{fmt_bucket(r['dense']):>14}  {r['skipped_drift'] or ''}")
        ts[0] += r["simple"][0]; ts[1] += r["simple"][1]
        td[0] += r["dense"][0]; td[1] += r["dense"][1]
        skip += r["skipped_drift"]
    tot = ts[1] + td[1]
    cor = ts[0] + td[0]
    print("-" * 86)
    print(f"{'合计':<24} {tot:>5} {cor / tot:>8.2%}  {fmt_bucket(ts):>14}  {fmt_bucket(td):>14}  {skip or ''}")


def pick_segs(only: str | None) -> list[str]:
    segs = list(MANIFEST)
    if only:
        want = {w if w.startswith("muli4_") else f"muli4_{w}" for w in only.split(",")}
        segs = [s for s in segs if s in want]
        missing = want - set(segs)
        if missing:
            raise SystemExit(f"未知样本: {missing}（可用: {list(MANIFEST)}）")
    return [s for s in segs if (SAMP / f"{s}_groundtruth.json").exists()]


def save_summary(results: list[dict], out: Path, meta: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n✓ 评分已存 {out}")


def cmd_score(args) -> None:
    segs = pick_segs(args.only)
    pdir = Path(args.parse_dir) if args.parse_dir else SAMP
    results = [r for s in segs if (r := score_sample(s, pdir / f"{s}_parse.json"))]
    print_table(results, f"score: {pdir}")
    if args.out:
        save_summary(results, Path(args.out),
                     {"mode": "score", "parse_dir": str(pdir),
                      "time": datetime.now().isoformat(timespec="seconds")})


def cmd_run(args) -> None:
    import os
    api_key = os.environ.get("AGNES_API_KEY")
    if not api_key:
        raise SystemExit("未设置 AGNES_API_KEY —— 请先 `source .env`")
    from book_voice_parser import BatchConfig, parse_novel

    segs = pick_segs(args.only)
    out_dir = REPO / "outputs/regression" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for s in segs:
        raw = (SAMP / f"{s}_sample.txt").read_text(encoding="utf-8")
        print(f"[{s}] 解析 {len(raw)} 字（narrator={MANIFEST[s] or '无'}）…", flush=True)
        cfg = BatchConfig(base_url="https://apihub.agnes-ai.com/v1", api_key=api_key,
                          model="agnes-2.0-flash", batch_size=8, max_tokens=5000,
                          temperature=0.0, timeout=240, context_chars=320,
                          output_mode="compact", disable_thinking=True)
        result = parse_novel(raw, role_hints=ROLE_HINTS, batch_llm_config=cfg,
                             narrator=MANIFEST[s], return_result=True,
                             include_narration=False, review_threshold=0.7,
                             enable_block_review=not args.no_block_review)
        pp = out_dir / f"{s}_parse.json"
        pp.write_text(json.dumps(
            {"segments": [x.model_dump(mode="json") for x in result.segments],
             "stats": result.stats}, ensure_ascii=False, indent=1), encoding="utf-8")
        r = score_sample(s, pp)
        if r:
            results.append(r)
            print(f"  → 具名 {r['named']} 准确率 {r['acc']:.2%}（漂移跳过 {r['skipped_drift']}）", flush=True)
    print_table(results, f"run: {args.tag}")
    save_summary(results, out_dir / "summary.json",
                 {"mode": "run", "tag": args.tag,
                  "no_block_review": args.no_block_review,
                  "time": datetime.now().isoformat(timespec="seconds")})


def cmd_compare(args) -> None:
    a = json.loads(Path(args.a).read_text(encoding="utf-8"))
    b = json.loads(Path(args.b).read_text(encoding="utf-8"))
    ra = {r["seg"]: r for r in a["results"]}
    rb = {r["seg"]: r for r in b["results"]}
    print(f"A = {args.a}  ({a['meta'].get('tag') or a['meta'].get('parse_dir', '')})")
    print(f"B = {args.b}  ({b['meta'].get('tag') or b['meta'].get('parse_dir', '')})")
    print(f"\n{'样本':<24} {'A准确率':>9} {'B准确率':>9} {'Δ':>8}   {'简单Δ':>8} {'密集Δ':>8}")
    tota = [0, 0]; totb = [0, 0]
    for seg in ra:
        if seg not in rb:
            continue
        x, y = ra[seg], rb[seg]
        d = (y["acc"] or 0) - (x["acc"] or 0)
        ds = (y["simple"][0] / y["simple"][1] if y["simple"][1] else 0) - \
             (x["simple"][0] / x["simple"][1] if x["simple"][1] else 0)
        dd = (y["dense"][0] / y["dense"][1] if y["dense"][1] else 0) - \
             (x["dense"][0] / x["dense"][1] if x["dense"][1] else 0)
        mark = "▲" if d > 0.001 else ("▼" if d < -0.001 else " ")
        print(f"{seg:<24} {x['acc']:>9.2%} {y['acc']:>9.2%} {d:>+8.2%}{mark}  {ds:>+8.2%} {dd:>+8.2%}")
        tota[0] += x["correct"]; tota[1] += x["named"]
        totb[0] += y["correct"]; totb[1] += y["named"]
    if tota[1] and totb[1]:
        da = tota[0] / tota[1]
        db = totb[0] / totb[1]
        print("-" * 78)
        print(f"{'合计':<24} {da:>9.2%} {db:>9.2%} {db - da:>+8.2%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("score", help="给现有 parse 评分（不调 API）")
    p.add_argument("--only", help="逗号分隔样本子集，如 seg5,seg7")
    p.add_argument("--parse-dir", help="parse 文件目录（默认 docs/samples）")
    p.add_argument("--out", help="评分 JSON 输出路径")
    p.set_defaults(func=cmd_score)
    p = sub.add_parser("run", help="用当前流水线重新解析并评分（调 API）")
    p.add_argument("--tag", required=True, help="本次运行标签（输出目录名）")
    p.add_argument("--only", help="逗号分隔样本子集")
    p.add_argument("--no-block-review", action="store_true", help="关闭块级复核（消融实验）")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("compare", help="对比两份评分 JSON")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(func=cmd_compare)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
