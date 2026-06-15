"""统一项目文件（project.json）的持久化与领域操作。

存储布局：
    outputs/projects/<project_id>/
        project.json
        clips/seg_000001.wav
        <project_id>.wav   (merge 产物)
        <project_id>.lrc

本模块只做纯数据/文件操作，不依赖 app.py（避免循环导入）。生成（TTS）的编排
由 app.py 的端点负责，本模块提供"选哪些段要生成""算 content_hash""写回 gen 记录"
等支撑函数。
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from audio_utils import build_lrc, get_wav_duration_seconds, join_wavs_auto
from character_registry import SKIP_SPEAKERS
from output_layout import OUTPUT_DIR, _sanitize_component
from schemas import (
    PROJECT_SCHEMA_VERSION,
    CastMember,
    GenerationRecord,
    Project,
    ProjectSegment,
    VoiceAssignment,
)


class ProjectError(Exception):
    pass


# ── 路径 ─────────────────────────────────────────────────────────────────────
def projects_root() -> Path:
    root = OUTPUT_DIR / "projects"
    root.mkdir(parents=True, exist_ok=True)
    return root


def normalize_project_id(project_id: str) -> str:
    return _sanitize_component(project_id, "project")


def project_dir(project_id: str) -> Path:
    return projects_root() / normalize_project_id(project_id)


def project_file(project_id: str) -> Path:
    return project_dir(project_id) / "project.json"


def clips_dir(project_id: str) -> Path:
    d = project_dir(project_id) / "clips"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ── 读写（原子） ─────────────────────────────────────────────────────────────
def _migrate(raw: dict) -> dict:
    """schema_version 迁移钩子。当前只有 v1。"""
    version = int(raw.get("schema_version") or 1)
    # 未来：if version < N: ...升级...
    raw["schema_version"] = PROJECT_SCHEMA_VERSION if version <= PROJECT_SCHEMA_VERSION else version
    return raw


def load_project(project_id: str) -> Project:
    path = project_file(project_id)
    if not path.exists():
        raise ProjectError(f"项目不存在：{project_id}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Project.model_validate(_migrate(raw))


def save_project(project: Project) -> Path:
    project.updated_at = _now()
    pdir = project_dir(project.project_id)
    pdir.mkdir(parents=True, exist_ok=True)
    path = pdir / "project.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(project.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)  # 原子替换
    return path


def project_exists(project_id: str) -> bool:
    return project_file(project_id).exists()


def list_projects() -> list[dict]:
    out: list[dict] = []
    root = projects_root()
    for pdir in sorted(root.iterdir()) if root.exists() else []:
        pf = pdir / "project.json"
        if not pf.exists():
            continue
        try:
            raw = json.loads(pf.read_text(encoding="utf-8"))
        except Exception:
            continue
        segments = raw.get("segments") or []
        done = sum(1 for s in segments if (s.get("gen") or {}).get("status") == "done")
        out.append(
            {
                "project_id": raw.get("project_id") or pdir.name,
                "title": raw.get("title") or "",
                "series": raw.get("series"),
                "segments": len(segments),
                "done": done,
                "cast": len(raw.get("cast") or []),
                "updated_at": raw.get("updated_at"),
            }
        )
    return out


# ── 创建 / 导入角色表 ────────────────────────────────────────────────────────
def create_project(
    project_id: str,
    title: str = "",
    *,
    series: Optional[str] = None,
    from_project: Optional[str] = None,
    defaults: Optional[dict] = None,
    overwrite: bool = False,
) -> Project:
    pid = normalize_project_id(project_id)
    if project_exists(pid) and not overwrite:
        raise ProjectError(f"项目已存在：{pid}（如需覆盖请传 overwrite=True）")
    project = Project(project_id=pid, title=title or pid, series=series, created_at=_now())
    if defaults:
        project.defaults = project.defaults.model_copy(update=defaults)
    if from_project:
        src = load_project(from_project)
        project.cast = [m.model_copy(deep=True) for m in src.cast]
        project.source = {"from_project": normalize_project_id(from_project)}
    save_project(project)
    return project


def import_cast(target_id: str, from_project: str, *, replace: bool = True) -> Project:
    """把源项目的 cast 拷入目标项目。replace=True 整体替换；否则按 canonical 合并。"""
    target = load_project(target_id)
    src = load_project(from_project)
    if replace:
        target.cast = [m.model_copy(deep=True) for m in src.cast]
    else:
        by_name = {m.canonical: m for m in target.cast}
        for m in src.cast:
            by_name[m.canonical] = m.model_copy(deep=True)
        target.cast = list(by_name.values())
    target.source = {**(target.source or {}), "cast_from": normalize_project_id(from_project)}
    save_project(target)
    return target


# ── 角色 / 音色解析 ──────────────────────────────────────────────────────────
def _alias_index(project: Project) -> dict[str, CastMember]:
    """别名/规范名 → CastMember。"""
    idx: dict[str, CastMember] = {}
    for member in project.cast:
        idx[member.canonical] = member
        for alias in member.aliases:
            idx.setdefault(alias, member)
    return idx


def canonical_speaker(project: Project, speaker: str) -> str:
    member = _alias_index(project).get((speaker or "").strip())
    return member.canonical if member else (speaker or "").strip()


def resolve_voice_for_segment(project: Project, seg: ProjectSegment) -> VoiceAssignment:
    """段级 override > cast(按 canonical/alias 命中) > 空。"""
    if seg.voice_override is not None:
        return seg.voice_override
    member = _alias_index(project).get((seg.speaker or "").strip())
    if member is not None:
        return member.voice
    return VoiceAssignment()


# ── content_hash / 状态 ─────────────────────────────────────────────────────
def compute_content_hash(
    project: Project, seg: ProjectSegment, *, engine: str, params: dict
) -> str:
    """对（文本 + 规范说话人 + 解析后音色 + 引擎 + 生成参数）做 sha256。
    任一变化都会改变 hash，从而触发该段 stale → 重生。"""
    voice = resolve_voice_for_segment(project, seg)
    payload = {
        "text": seg.text,
        "speaker": canonical_speaker(project, seg.speaker),
        "voice": voice.model_dump(),
        "engine": engine,
        "params": params,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def effective_engine_and_params(project: Project) -> tuple[str, dict]:
    engine = project.defaults.engine or "index-tts"
    params = dict(project.defaults.generation_options or {})
    return engine, params


def recompute_status(project: Project) -> Project:
    """重算每段 hash 与状态：
    - hash 与记录一致且 clip 存在 → done
    - 记录里有 clip 但 hash 变了 → stale
    - 从未生成（无 clip） → pending
    """
    engine, params = effective_engine_and_params(project)
    pdir = project_dir(project.project_id)
    for seg in project.segments:
        new_hash = compute_content_hash(project, seg, engine=engine, params=params)
        gen = seg.gen
        clip_ok = bool(gen.clip) and (pdir / gen.clip).exists()
        if not clip_ok:
            gen.status = "pending" if gen.status != "error" else "error"
        elif gen.content_hash == new_hash:
            gen.status = "done"
        else:
            gen.status = "stale"
    return project


# ── seg_id / 段管理 ─────────────────────────────────────────────────────────
def _next_seg_id(project: Project) -> str:
    sid = f"seg_{project.next_seg_seq:06d}"
    project.next_seg_seq += 1
    return sid


def _to_voice_assignment(profile: Optional[dict]) -> VoiceAssignment:
    if not profile:
        return VoiceAssignment()
    keys = VoiceAssignment.model_fields.keys()
    return VoiceAssignment(**{k: profile.get(k) for k in keys if profile.get(k) is not None})


def build_project_from_state(
    project_id: str,
    title: str,
    segments: list[dict],
    *,
    role_profiles: Optional[dict] = None,
    alias_map: Optional[dict] = None,
    series: Optional[str] = None,
    defaults: Optional[dict] = None,
    overwrite: bool = False,
) -> Project:
    """从当前 localStorage 风格状态迁移建项目：
    - segments：分段 JSON（speaker/text/emotion/style，可含 ref_audio/ref_text）
    - role_profiles：{speaker: {voice_engine, ref_audio, ref_text, style,...}}
    - alias_map：{alias: canonical}（来自 CharacterRegistry）
    cast 由 role_profiles 推导，并合并 alias_map。
    """
    pid = normalize_project_id(project_id)
    if project_exists(pid) and not overwrite:
        raise ProjectError(f"项目已存在：{pid}")
    project = Project(project_id=pid, title=title or pid, series=series, created_at=_now())
    if defaults:
        project.defaults = project.defaults.model_copy(update=defaults)

    # 反向别名：canonical → [aliases]
    aliases_by_canonical: dict[str, list[str]] = {}
    for alias, canonical in (alias_map or {}).items():
        if alias and canonical and alias != canonical:
            aliases_by_canonical.setdefault(canonical, []).append(alias)

    # cast：role_profiles 的每个 speaker 一个条目
    cast: list[CastMember] = []
    seen: set[str] = set()
    for speaker, profile in (role_profiles or {}).items():
        sp = str(speaker or "").strip()
        if not sp or sp in seen:
            continue
        seen.add(sp)
        cast.append(
            CastMember(
                canonical=sp,
                aliases=sorted(set(aliases_by_canonical.get(sp, []))),
                voice=_to_voice_assignment(profile),
            )
        )
    project.cast = cast

    # segments → ProjectSegment（分配稳定 seg_id）
    for order, seg in enumerate(segments):
        sp = str(seg.get("speaker") or "旁白").strip() or "旁白"
        text = str(seg.get("text") or "")
        override = None
        # 段内若直接带了 ref_audio 等且与角色表不同，记为段级覆盖
        seg_voice = _to_voice_assignment(seg)
        if seg_voice.model_dump(exclude_none=True):
            override = seg_voice
        project.segments.append(
            ProjectSegment(
                seg_id=_next_seg_id(project),
                order=order,
                speaker=sp,
                text=text,
                emotion=str(seg.get("emotion") or "neutral"),
                style=seg.get("style"),
                voice_override=override,
                gen=GenerationRecord(status="pending"),
            )
        )

    recompute_status(project)
    save_project(project)
    return project


# ── 生成选择 / 写回 ─────────────────────────────────────────────────────────
def select_to_generate(
    project: Project,
    *,
    seg_ids: Optional[Iterable[str]] = None,
    force: bool = False,
) -> list[ProjectSegment]:
    """挑出要生成的段，并按 resolved ref_audio 分组排序（同一参考音频连续，
    配合常驻 bridge 让单槽缓存跨段生效）。
    - seg_ids 指定：只这些；force：忽略状态全做。
    - 否则：status in {pending, stale, error}。
    """
    recompute_status(project)
    id_filter = set(seg_ids) if seg_ids is not None else None
    selected: list[ProjectSegment] = []
    for seg in project.segments:
        if id_filter is not None:
            if seg.seg_id in id_filter:
                selected.append(seg)
            continue
        if force or seg.gen.status in {"pending", "stale", "error"}:
            selected.append(seg)

    def bucket_key(seg: ProjectSegment) -> str:
        voice = resolve_voice_for_segment(project, seg)
        return voice.ref_audio or f"__noref__:{canonical_speaker(project, seg.speaker)}"

    # 稳定分组：保持各组首次出现顺序，组内按 order
    order_of_key: dict[str, int] = {}
    for seg in selected:
        order_of_key.setdefault(bucket_key(seg), len(order_of_key))
    selected.sort(key=lambda s: (order_of_key[bucket_key(s)], s.order))
    return selected


def record_generation(
    project: Project,
    seg: ProjectSegment,
    *,
    clip_abs_path: Path,
    engine: str,
    params: dict,
    ok: bool,
    error: Optional[str] = None,
) -> None:
    """把一次生成结果写回段的 gen 记录（不落盘，由调用方统一 save_project）。"""
    pdir = project_dir(project.project_id)
    rel = str(clip_abs_path.relative_to(pdir)).replace("\\", "/")
    gen = seg.gen
    gen.engine = engine
    gen.params = dict(params)
    gen.generated_at = _now()
    if ok and clip_abs_path.exists() and clip_abs_path.stat().st_size > 0:
        gen.clip = rel
        gen.duration = round(get_wav_duration_seconds(clip_abs_path), 3)
        gen.content_hash = compute_content_hash(project, seg, engine=engine, params=params)
        gen.status = "done"
        gen.error = None
    else:
        gen.status = "error"
        gen.error = error or "generation failed"


def clip_path_for(project: Project, seg: ProjectSegment) -> Path:
    return clips_dir(project.project_id) / f"{seg.seg_id}.wav"


# ── 合并 ─────────────────────────────────────────────────────────────────────
def merge_project(project: Project, *, silence_ms: Optional[int] = None, write_lrc: bool = True) -> dict:
    """按 order 合并已 done 的片段 → outputs/projects/<id>/<id>.wav (+ .lrc)。"""
    silence = project.defaults.silence_ms if silence_ms is None else silence_ms
    pdir = project_dir(project.project_id)
    ordered = sorted(project.segments, key=lambda s: s.order)
    wav_paths: list[str] = []
    lyric_lines: list[str] = []
    for seg in ordered:
        if seg.gen.status == "done" and seg.gen.clip:
            clip = pdir / seg.gen.clip
            if clip.exists():
                wav_paths.append(str(clip))
                lyric_lines.append(f"{seg.speaker}：{seg.text}" if seg.speaker not in SKIP_SPEAKERS else seg.text)
    if not wav_paths:
        raise ProjectError("没有可合并的已生成片段（done）。")
    final_path = pdir / f"{project.project_id}.wav"
    merge_result = join_wavs_auto(wav_paths, final_path, silence_ms=silence)
    result = {"file": str(final_path), "segments": len(wav_paths)}
    if isinstance(merge_result, dict):
        result.update(merge_result)
    if write_lrc:
        lrc_path = pdir / f"{project.project_id}.lrc"
        build_lrc(wav_paths, lyric_lines, lrc_path, silence_ms=silence)
        result["lrc"] = str(lrc_path)
    return result
