from __future__ import annotations

from dataclasses import dataclass, field


SKIP_NAMES = {"", "旁白", "未知", "UNKNOWN", "众人"}
TITLE_SUFFIXES = ("公子", "姑娘", "小姐", "夫人", "大人", "先生", "师兄", "师姐", "师妹", "师父")


@dataclass
class AliasRegistry:
    alias_map: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_role_hints(cls, role_hints: dict[str, list[str] | str] | list[str] | None) -> "AliasRegistry":
        registry = cls()
        if isinstance(role_hints, dict):
            for canonical, aliases in role_hints.items():
                registry.add(canonical, canonical)
                if isinstance(aliases, str):
                    registry.add(aliases, canonical)
                else:
                    for alias in aliases or []:
                        registry.add(alias, canonical)
        elif isinstance(role_hints, list):
            for name in role_hints:
                registry.add(name, name)
        return registry

    def add(self, alias: str, canonical: str) -> None:
        alias = (alias or "").strip()
        canonical = (canonical or "").strip()
        if alias and canonical and alias not in SKIP_NAMES:
            self.alias_map[alias] = canonical

    def update_inferred_aliases(self, names: list[str]) -> dict[str, str]:
        """Infer aliases like 陆公子 -> 陆沉 when there is one same-surname canonical."""

        for name in names:
            cleaned = (name or "").strip()
            if cleaned and cleaned not in SKIP_NAMES and not self.alias_map.get(cleaned):
                self.add(cleaned, cleaned)

        canonicals = [name for name in self.known_names() if len(name) > 1]
        inferred: dict[str, str] = {}
        for name in names:
            alias = (name or "").strip()
            if alias in SKIP_NAMES or alias in self.alias_map and self.alias_map[alias] != alias:
                continue
            suffix = next((item for item in TITLE_SUFFIXES if alias.endswith(item)), "")
            if not suffix or len(alias) <= len(suffix):
                continue
            surname = alias[: -len(suffix)]
            matches = [canonical for canonical in canonicals if canonical != alias and canonical.startswith(surname)]
            if len(matches) != 1:
                continue
            self.add(alias, matches[0])
            inferred[alias] = matches[0]
        return inferred

    def canonicalize(self, name: str) -> str:
        cleaned = (name or "").strip()
        return self.alias_map.get(cleaned, cleaned)

    def known_names(self) -> list[str]:
        return sorted(set(self.alias_map.values()))

    def match_names(self) -> list[str]:
        return sorted((name for name in set(self.alias_map) | set(self.alias_map.values()) if len(name) > 1), key=len, reverse=True)

    def has_hints(self) -> bool:
        return bool(self.alias_map)
