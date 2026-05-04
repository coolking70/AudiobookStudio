from __future__ import annotations

from dataclasses import dataclass, field


SKIP_NAMES = {"", "旁白", "未知", "UNKNOWN", "众人"}
TITLE_SUFFIXES = ("公子", "姑娘", "小姐", "夫人", "大人", "先生", "师兄", "师姐", "师妹", "师父")


@dataclass
class RelationRole:
    """无真实姓名、以关系称谓出场的角色（如「遥奈妈妈」）。

    owner 在场景上下文中出现时，aliases 中的称谓（如「妈妈」）才作为候选激活。
    这避免了「妈妈」全局映射到某个固定角色，而是根据「谁的妈妈在场」动态判断。

    role_hints 中的声明格式：
        "遥奈妈妈": {"aliases": ["妈妈", "母亲"], "owner": "甘织遥奈"}
    """
    canonical: str                             # 规范名，如「遥奈妈妈」
    owner: str                                 # 关联的已命名角色，如「甘织遥奈」
    aliases: list[str] = field(default_factory=list)  # 称谓别名，如 ["妈妈", "母亲"]


@dataclass
class AliasRegistry:
    alias_map: dict[str, str] = field(default_factory=dict)
    relation_roles: list[RelationRole] = field(default_factory=list)

    @classmethod
    def from_role_hints(cls, role_hints: dict[str, list[str] | str | dict] | list[str] | None) -> "AliasRegistry":
        registry = cls()
        if isinstance(role_hints, dict):
            for canonical, value in role_hints.items():
                if isinstance(value, dict) and "owner" in value:
                    # 关系角色格式：{"aliases": [...], "owner": "角色名"}
                    owner = str(value.get("owner", "")).strip()
                    aliases_list = [str(a).strip() for a in (value.get("aliases") or []) if a]
                    if canonical and owner:
                        # 只将规范名自身加入 alias_map（不加 term aliases，保持 owner 条件激活）
                        registry.add(canonical, canonical)
                        registry.relation_roles.append(
                            RelationRole(canonical=canonical, owner=owner, aliases=aliases_list)
                        )
                else:
                    # 普通角色格式：["别名1", "别名2"] 或 "别名"
                    registry.add(canonical, canonical)
                    if isinstance(value, str):
                        registry.add(value, canonical)
                    else:
                        for alias in (value or []):
                            registry.add(alias, canonical)
        elif isinstance(role_hints, list):
            for name in role_hints:
                registry.add(name, name)
        return registry

    def get_relation_roles(self, term: str) -> list[RelationRole]:
        """返回以 term 为别名的所有关系角色（按 owner 区分，通常 0-2 个）。"""
        return [rr for rr in self.relation_roles if term in rr.aliases]

    def is_relation_role(self, canonical: str) -> bool:
        """该规范名是否为关系角色（无真实姓名，由 owner 条件激活）。"""
        return any(rr.canonical == canonical for rr in self.relation_roles)

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
