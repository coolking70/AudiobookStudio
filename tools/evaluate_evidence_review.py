"""Review structured Agnes baseline evidence with a second model.

This experiment asks the reviewer to verify/refute the baseline evidence
(`E=...;R=...;S=...`) rather than simply re-guessing the speaker.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # offline scoring/gating does not require the client SDK
    OpenAI = None

REPO = Path(__file__).resolve().parents[1]
SAMP = REPO / "docs/samples"
sys.path.insert(0, str(REPO / "tools"))

from run_regression import canon, is_crowd, norm, score_sample  # noqa: E402
from openai_retry import call_with_404_backoff  # noqa: E402


PROVIDERS = {
    "tokenhub": "https://tokenhub.tencentmaas.com/v1",
    "agnes": "https://apihub.agnes-ai.com/v1",
    "sensenova": "https://token.sensenova.cn/v1",
}


STRUCTURED_RE = re.compile(
    r"(?:^|[；;\s])E=(?P<e>[^;；\n]+)[;；]R=(?P<r>[^;；\n]+)(?:[;；]S=(?P<s>[^;；\n]+))?",
)

STRONG_REASON_RE = re.compile(
    r"明示|明确|证实|台词第一人称|第一人称|屏幕|被拉|语气.*喵|喵|伸手邀请|惊讶回应|嘟囔"
)


SYSTEM_PROMPT = """\
你是中文小说有声书说话人复核专家。

任务不是重新猜一次，而是验证/反驳基线归因的结构化证据：
- 若基线证据 E/R/S 与原文一致，且没有更强反证，decision=keep。
- 若基线证据不成立，或存在更强原文证据指向另一位说话人，decision=revise。
- 台词中被称呼的人通常是受话人，不是说话人。
- 仅凭对话轮换/上一句推测属于弱证据；多人同场时要更保守。
- 不要输出解释性正文，只输出 JSON 数组。
- 如果你认为 speaker 不需要改变，decision 必须为 keep，不要返回 revise。
- revise 只用于改成与当前 speaker 不同的角色。

返回每个目标：
{"index":整数,"decision":"keep|revise|uncertain","speaker":"角色名","confidence":0.0到1.0,"reason":"≤24字，指出证据成立或反证"}
"""


AUDIT_SAFE_SYSTEM_PROMPT = """\
你是中文小说有声书说话人证据审计专家。

你的任务不是用强模型替代基线重判，而是判断“是否有足够证据安全推翻当前 speaker”。

必须逐项检查：
1. baseline_evidence_valid：当前 E/R/S 或原 evidence 是否真的支持当前 speaker。
2. counter_evidence_type：若要改判，必须指出更强反证类型：
   - explicit_before：台词前动作主语明确说/问/喊/答
   - explicit_after：台词后明确“X说/问/喊/答/如此说道”
   - first_person：第一人称视角/内心锚定明确
   - address_term：称呼反推明确，且被称呼者不是说话人
   - identity_alias：别名/身份归并明确（如 moon/帕曼小姐=琴纱月）
   - semantic_reply：被提问/被点名者接话，语义邻接非常明确
   - none：没有足够强反证
3. independent_signal：除强模型判断外，是否有独立风险/分歧信号支持复核目标，例如低置信、风险标签、机器审计第二意见分歧、密集多人场景、候选/称呼冲突。

自动覆盖原则：
- 只有 baseline_evidence_valid=false，counter_evidence_type 不是 none，且 independent_signal=true，才允许 auto_apply_safe=true。
- 仅凭“我觉得另一人更像”或普通轮换，不允许 auto_apply_safe=true。
- 如果证据不足但怀疑错误，decision=uncertain，auto_apply_safe=false。
- 如果当前 speaker 仍可被证据支持，decision=keep。
- revise 只用于改成与当前 speaker 不同的规范角色名。

规范名要求：
- 甘织玲奈子：我、小玲奈、玲奈亲
- 小柳香穗：小香穗、香穗
- 琴纱月：纱月、小纱月、moon、moon小姐、帕曼小姐
- 濑名紫阳花：紫阳花同学、小紫

不要输出解释性正文，只输出 JSON 数组。每个目标格式：
{"index":整数,"decision":"keep|revise|uncertain","speaker":"角色名","confidence":0.0到1.0,"baseline_evidence_valid":true|false,"counter_evidence_type":"explicit_before|explicit_after|first_person|address_term|identity_alias|semantic_reply|none","independent_signal":true|false,"auto_apply_safe":true|false,"reason":"≤28字"}
"""


STRONG_COUNTER_EVIDENCE = {
    "explicit_before",
    "explicit_after",
    "first_person",
    "address_term",
    "identity_alias",
    "semantic_reply",
}

DEFAULT_COUNTER_TYPE_THRESHOLDS = {
    "explicit_after": 0.65,
    "first_person": 0.75,
    "identity_alias": 0.75,
    "address_term": 0.80,
    "semantic_reply": 0.85,
}


def parse_evidence(evidence: str) -> dict[str, str]:
    m = STRUCTURED_RE.search(str(evidence or ""))
    if not m:
        return {"E": "", "R": "", "S": ""}
    return {"E": m.group("e") or "", "R": m.group("r") or "", "S": m.group("s") or ""}


def load_segments(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("segments", data if isinstance(data, list) else [])


def load_truth(seg: str) -> list[dict[str, Any]]:
    return json.loads((SAMP / f"{seg}_groundtruth.json").read_text(encoding="utf-8"))["segments"]


def scored_rows(seg: str, parse_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for g in load_truth(seg):
        i = int(g["i"])
        if is_crowd(g["speaker"]) or i >= len(parse_segments):
            continue
        p = parse_segments[i]
        if norm(p.get("text", "")) != norm(g["text"]):
            continue
        acceptable = {canon(x) for x in (g.get("acceptable") or [g["speaker"]])}
        acceptable.add(canon(g["speaker"]))
        got = canon(str(p.get("speaker") or ""))
        rows.append({
            "i": i,
            "ok": got in acceptable,
            "truth": canon(str(g["speaker"])),
            "speaker": got,
        })
    return rows


def has_risk(seg: dict[str, Any]) -> bool:
    ev = str(seg.get("evidence") or "")
    tags = parse_evidence(ev)
    risks = {x.strip() for x in tags["R"].split(",") if x.strip() and x.strip() != "none"}
    try:
        conf = float(seg.get("confidence"))
    except (TypeError, ValueError):
        conf = 1.0
    return conf < 0.7 or bool(risks)


def weak_evidence(seg: dict[str, Any]) -> bool:
    tags = parse_evidence(str(seg.get("evidence") or ""))
    e_tags = {x.strip() for x in tags["E"].split(",") if x.strip()}
    weak = {"dialogue_alternation", "narrator_anchor", "scene_presence", "address_term", "unknown_weak"}
    return bool(e_tags & weak)


def load_audit(path: Path | None) -> dict[int, dict[str, Any]]:
    if not path or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = data.get("segments") or {}
    out: dict[int, dict[str, Any]] = {}
    for key, value in segments.items():
        try:
            out[int(key)] = value if isinstance(value, dict) else {}
        except (TypeError, ValueError):
            continue
    return out


def audit_disagrees(parse_seg: dict[str, Any], audit_item: dict[str, Any]) -> bool:
    reask = str(audit_item.get("reask") or "").strip()
    if not reask:
        return False
    current = str(parse_seg.get("speaker") or "").strip()
    return bool(reask) and canon(reask) != canon(current)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value or "").strip().lower()
    return text in {"true", "yes", "y", "1", "是", "对"}


def parse_type_thresholds(value: str | None) -> dict[str, float]:
    if not value:
        return {}
    if value.strip().lower() in {"default", "strong-default"}:
        return dict(DEFAULT_COUNTER_TYPE_THRESHOLDS)
    out: dict[str, float] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        key, sep, raw = item.partition("=")
        if not sep:
            raise argparse.ArgumentTypeError(f"invalid type threshold item: {item}")
        try:
            out[key.strip()] = float(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid threshold value: {item}") from exc
    return out


def choose_targets(
    seg_name: str,
    parse_segments: list[dict[str, Any]],
    mode: str,
    limit: int | None,
    audit: dict[int, dict[str, Any]] | None = None,
) -> list[int]:
    rows = scored_rows(seg_name, parse_segments)
    by_i = {r["i"]: r for r in rows}
    if mode == "errors":
        indices = [r["i"] for r in rows if not r["ok"]]
    elif mode == "errors-and-controls":
        errors = [r["i"] for r in rows if not r["ok"]]
        controls = [r["i"] for r in rows if r["ok"] and has_risk(parse_segments[r["i"]])]
        indices = errors + controls[: max(len(errors), 12)]
    elif mode == "audit-disagree":
        audit = audit or {}
        indices = [
            i for i in sorted(by_i)
            if i in audit
            and audit_disagrees(parse_segments[i], audit[i])
            and (has_risk(parse_segments[i]) or weak_evidence(parse_segments[i]))
        ]
    elif mode == "audit-disagree-all":
        audit = audit or {}
        indices = [
            i for i in sorted(by_i)
            if i in audit and audit_disagrees(parse_segments[i], audit[i])
        ]
    else:
        indices = [i for i in sorted(by_i) if has_risk(parse_segments[i])]
    return indices[:limit] if limit else indices


def build_user_prompt(
    seg_name: str,
    parse_segments: list[dict[str, Any]],
    indices: list[int],
    audit: dict[int, dict[str, Any]] | None = None,
) -> str:
    lines = [f"样本：{seg_name}", "请复核以下目标："]
    for i in indices:
        s = parse_segments[i]
        ev = parse_evidence(str(s.get("evidence") or ""))
        candidates = "、".join(str(x) for x in (s.get("candidates") or [])[:16])
        audit_item = (audit or {}).get(i) or {}
        audit_line = ""
        if audit_item.get("reask"):
            audit_line = f"机器审计第二意见={audit_item.get('reask')}；理由={audit_item.get('reask_reason') or '-'}"
        independent_signals = []
        try:
            conf = float(s.get("confidence"))
        except (TypeError, ValueError):
            conf = 1.0
        if conf < 0.7:
            independent_signals.append("low_confidence")
        if ev["R"] and ev["R"] != "none":
            independent_signals.append(f"risk={ev['R']}")
        if weak_evidence(s):
            independent_signals.append("weak_evidence")
        if audit_item.get("reask") and audit_disagrees(s, audit_item):
            independent_signals.append("audit_disagree")
        scene = s.get("scene_characters") or []
        if isinstance(scene, list) and len(scene) >= 3:
            independent_signals.append("multi_speaker_scene")
        lines.extend([
            "",
            f"index={i}",
            f"当前speaker={s.get('speaker')} confidence={s.get('confidence')}",
            f"结构化依据: E={ev['E'] or '-'}; R={ev['R'] or '-'}; S={ev['S'] or '-'}",
            f"独立信号: {', '.join(independent_signals) if independent_signals else 'none'}",
            audit_line,
            f"候选/在场: {candidates or '（无）'}",
            f"前文: {str(s.get('context_before') or '')[-420:]}",
            f"台词: 「{s.get('text') or ''}」",
            f"后文: {str(s.get('context_after') or '')[:420]}",
        ])
    return "\n".join(lines)


def extract_json_array(text: str) -> list[dict[str, Any]]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL).strip()
    m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def call_review(
    client: OpenAI,
    model: str,
    seg_name: str,
    parse_segments: list[dict[str, Any]],
    indices: list[int],
    timeout: int,
    audit: dict[int, dict[str, Any]] | None = None,
    review_style: str = "evidence",
) -> list[dict[str, Any]]:
    user = build_user_prompt(seg_name, parse_segments, indices, audit=audit)
    system_prompt = AUDIT_SAFE_SYSTEM_PROMPT if review_style == "audit-safe" else SYSTEM_PROMPT
    resp = call_with_404_backoff(
        lambda: client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=2500,
            timeout=timeout,
        )
    )
    content = (resp.choices[0].message.content or "").strip()
    return extract_json_array(content)


def apply_reviews(
    parse_segments: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    min_confidence: float,
    reason_gate: str = "none",
    review_style: str = "evidence",
    type_thresholds: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    updated = [dict(s) for s in parse_segments]
    stats = {
        "reviewed": 0,
        "revised": 0,
        "kept": 0,
        "uncertain": 0,
        "ignored": 0,
        "blocked_by_reason_gate": 0,
        "blocked_by_audit_gate": 0,
        "blocked_by_address_term_gate": 0,
        "type_threshold_applied": 0,
    }
    chain_review_indices: set[int] = set()
    type_thresholds = type_thresholds or {}
    for r in reviews:
        try:
            i = int(r.get("index"))
        except (TypeError, ValueError):
            stats["ignored"] += 1
            continue
        if i < 0 or i >= len(updated):
            stats["ignored"] += 1
            continue
        decision = str(r.get("decision") or "").strip()
        speaker = str(r.get("speaker") or "").strip()
        try:
            conf = float(r.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        reason = str(r.get("reason") or "").strip()
        stats["reviewed"] += 1
        old = str(updated[i].get("speaker") or "").strip()
        if decision == "revise" and speaker == old:
            updated[i]["evidence"] = f"{updated[i].get('evidence') or ''}；证据复核确认({conf:.2f}): {reason}"
            stats["kept"] += 1
        elif decision == "revise" and speaker:
            counter_type = str(r.get("counter_evidence_type") or "none").strip()
            apply_threshold = float(type_thresholds.get(counter_type, min_confidence))
            if conf + 1e-9 < apply_threshold:
                updated[i]["evidence"] = (
                    f"{updated[i].get('evidence') or ''}；证据复核待人工({conf:.2f}): "
                    f"低于{counter_type}阈值{apply_threshold:.2f}: {reason}"
                )
                stats["uncertain"] += 1
                continue
            if counter_type in type_thresholds and apply_threshold != min_confidence:
                stats["type_threshold_applied"] += 1
            if review_style == "audit-safe":
                baseline_valid = as_bool(r.get("baseline_evidence_valid"))
                independent = as_bool(r.get("independent_signal"))
                auto_safe = as_bool(r.get("auto_apply_safe"))
                if (
                    baseline_valid
                    or counter_type not in STRONG_COUNTER_EVIDENCE
                    or not independent
                    or not auto_safe
                ):
                    updated[i]["evidence"] = (
                        f"{updated[i].get('evidence') or ''}；证据审计待人工({conf:.2f}): "
                        f"gate blocked type={counter_type} baseline_valid={baseline_valid} "
                        f"independent={independent} auto_safe={auto_safe}; {reason}"
                    )
                    stats["blocked_by_audit_gate"] += 1
                    stats["uncertain"] += 1
                    continue
            if counter_type == "address_term":
                # A vocative identifies the addressee, not the speaker. Only
                # an explicit first-person/action-subject explanation can
                # override this hard gate.
                address_override = re.search(r"第一人称|动作主语|我(?:说|问|道)|(?:说|问|回答|喊|道)出|explicit", reason, re.I)
                if not address_override:
                    updated[i]["evidence"] = (
                        f"{updated[i].get('evidence') or ''}; address_term_gate_blocked: {reason}"
                    )
                    stats["blocked_by_address_term_gate"] += 1
                    stats["uncertain"] += 1
                    continue
            if reason_gate == "strong" and not STRONG_REASON_RE.search(reason):
                updated[i]["evidence"] = f"{updated[i].get('evidence') or ''}；证据复核待人工({conf:.2f}): 反证不够强：{reason}"
                stats["blocked_by_reason_gate"] += 1
                stats["uncertain"] += 1
                continue
            old = updated[i].get("speaker")
            updated[i]["speaker"] = speaker
            updated[i]["confidence"] = max(conf, float(updated[i].get("confidence") or 0.0))
            updated[i]["evidence"] = f"证据复核改判：{old}→{speaker}({conf:.2f}); {reason}; {updated[i].get('evidence') or ''}"
            stats["revised"] += 1
            chain_review_indices.update(j for j in (i - 1, i, i + 1) if 0 <= j < len(updated))
        elif decision == "keep":
            updated[i]["evidence"] = f"{updated[i].get('evidence') or ''}；证据复核确认({conf:.2f}): {reason}"
            stats["kept"] += 1
        else:
            updated[i]["evidence"] = f"{updated[i].get('evidence') or ''}；证据复核待人工({conf:.2f}): {reason}"
            stats["uncertain"] += 1
    stats["chain_review_indices"] = sorted(chain_review_indices)
    return updated, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg", default="muli4_seg8")
    ap.add_argument("--parse", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs/regression/evidence-review")
    ap.add_argument("--provider", default="tokenhub", choices=sorted(PROVIDERS))
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--target-mode", default="flagged",
                    choices=["flagged", "errors", "errors-and-controls", "audit-disagree", "audit-disagree-all"])
    ap.add_argument("--audit", type=Path, help="Optional machine-audit JSON with reask second opinions")
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--min-confidence", type=float, default=0.85)
    ap.add_argument("--reason-gate", default="none", choices=["none", "strong"],
                    help="Automatic revise gate: strong requires explicit-counterevidence words in reviewer reason")
    ap.add_argument("--review-style", default="evidence", choices=["evidence", "audit-safe"],
                    help="evidence keeps the original verify/refute prompt; audit-safe requires explicit proof and independent signals before auto-apply.")
    ap.add_argument("--type-thresholds",
                    help="Optional per counter_evidence_type thresholds, e.g. explicit_before=0.65,first_person=0.75,semantic_reply=0.85. Use 'default' for the tuned audit-safe defaults.")
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    key_name = {
        "agnes": "AGNES_API_KEY",
        "sensenova": "SENSENOVA_API_KEY",
        "tokenhub": "TOKENHUB_API_KEY",
    }[args.provider]
    api_key = os.environ.get(key_name)
    if not api_key:
        raise SystemExit(f"Missing {key_name}")

    parse_segments = load_segments(args.parse)
    type_thresholds = parse_type_thresholds(args.type_thresholds)
    audit = load_audit(args.audit)
    targets = choose_targets(args.seg, parse_segments, args.target_mode, args.limit, audit=audit)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    client = OpenAI(base_url=PROVIDERS[args.provider], api_key=api_key)

    print(f"provider={args.provider} model={args.model} targets={len(targets)} mode={args.target_mode}")
    all_reviews: list[dict[str, Any]] = []
    for start in range(0, len(targets), args.batch_size):
        batch = targets[start:start + args.batch_size]
        print(f"review batch {start // args.batch_size + 1}: {batch}", flush=True)
        try:
            reviews = call_review(
                client,
                args.model,
                args.seg,
                parse_segments,
                batch,
                args.timeout,
                audit=audit,
                review_style=args.review_style,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"batch failed: {exc}", flush=True)
            reviews = []
        all_reviews.extend(reviews)
        time.sleep(0.5)

    updated, stats = apply_reviews(
        parse_segments,
        all_reviews,
        args.min_confidence,
        reason_gate=args.reason_gate,
        review_style=args.review_style,
        type_thresholds=type_thresholds,
    )
    parse_out = args.out_dir / f"{args.seg}_parse.json"
    parse_out.write_text(json.dumps({"segments": updated, "stats": {"evidence_review": stats}},
                                    ensure_ascii=False, indent=1), encoding="utf-8")
    review_out = args.out_dir / f"{args.seg}_evidence_reviews.json"
    review_out.write_text(json.dumps({
        "targets": targets,
        "reviews": all_reviews,
        "stats": stats,
        "review_style": args.review_style,
        "type_thresholds": type_thresholds,
    },
                                     ensure_ascii=False, indent=1), encoding="utf-8")

    score = score_sample(args.seg, parse_out)
    print(json.dumps({"stats": stats, "score": score}, ensure_ascii=False, indent=2))
    print(f"saved: {parse_out}")
    print(f"saved: {review_out}")


if __name__ == "__main__":
    main()
