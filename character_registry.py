"""角色注册表：收集、归并、规范化角色名。"""
from __future__ import annotations

SKIP_SPEAKERS: frozenset[str] = frozenset({"旁白", "UNKNOWN", ""})


class CharacterRegistry:
    """轻量级角色注册表，支持别名归并和规范化。

    工作流：
    1. observe / observe_segments  ← 收集每个 chunk 分析结果中的 speaker
    2. load_alias_groups           ← 加载 LLM 别名归并结果（可选）
    3. canonicalize / apply_to_segments ← 规范化最终 segments
    """

    def __init__(self) -> None:
        self._observed: set[str] = set()
        self._alias_map: dict[str, str] = {}  # alias → canonical

    # ── 收集 ─────────────────────────────────────────────────────────────────
    def observe(self, name: str) -> None:
        n = (name or "").strip()
        if n and n not in SKIP_SPEAKERS:
            self._observed.add(n)

    def observe_many(self, names: list[str]) -> None:
        for n in names:
            self.observe(n)

    def observe_segments(self, segments: list[dict]) -> None:
        for seg in segments:
            self.observe(seg.get("speaker", ""))

    # ── 查询 ─────────────────────────────────────────────────────────────────
    def observed_names(self) -> list[str]:
        """所有出现过的原始角色名（未归并）。"""
        return sorted(self._observed)

    def canonical_names(self) -> list[str]:
        """去重后的规范化角色名列表。"""
        canonicals: set[str] = set()
        for name in self._observed:
            canonicals.add(self._alias_map.get(name, name))
        return sorted(canonicals)

    def canonicalize(self, name: str) -> str:
        return self._alias_map.get(name, name)

    def alias_dict(self) -> dict[str, str]:
        return dict(self._alias_map)

    def is_known(self, name: str) -> bool:
        """判断名字是否在注册表中（含别名）。"""
        return name in self._observed or name in self._alias_map

    # ── 别名归并 ──────────────────────────────────────────────────────────────
    def load_alias_groups(self, groups: list[dict]) -> None:
        """加载 LLM 返回的别名分组，更新 alias_map。

        groups 格式: [{"canonical": "米拉", "aliases": ["琉德米拉", "米拉大人"]}]
        只归并注册表中真正出现过的别名，避免误合并。
        """
        for group in (groups or []):
            canonical = str(group.get("canonical") or "").strip()
            if not canonical:
                continue
            for alias in (group.get("aliases") or []):
                a = str(alias or "").strip()
                if a and a != canonical and a in self._observed:
                    self._alias_map[a] = canonical

    def apply_to_segments(self, segments: list[dict]) -> list[dict]:
        """将 segments 中的 speaker 规范化（in-place），返回原列表。"""
        for seg in segments:
            sp = seg.get("speaker", "")
            if sp and sp not in SKIP_SPEAKERS:
                seg["speaker"] = self.canonicalize(sp)
        return segments

    # ── 导入/导出（用于 context 跨调用传递）─────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "observed": sorted(self._observed),
            "alias_map": dict(self._alias_map),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CharacterRegistry":
        reg = cls()
        reg._observed = set(data.get("observed") or [])
        reg._alias_map = dict(data.get("alias_map") or {})
        return reg
