"""
endpoint_constraints.py — 可选的 record_type/endpoint 兼容性约束（Stage 2 Extraction）。

设计目标：
- 让特定领域 preset（如 pancan）在抽取后做确定性的 record_type ↔ endpoint 校验，
  把 endpoint 不属于其 record_type 允许列表的 record 直接剔除。
- 没有配置该机制的其他领域完全保持旧行为（build_* 返回 None → 不做任何过滤）。

仅实现两种模式：
- ``strict``：按 record_type 限制 endpoint，并把 endpoint 规范化为 canonical 名称。
- ``unrestricted``（或缺失配置）：不做任何约束，返回 None。

未来扩展位置：
- 可在 ``build_endpoint_constraint`` 中支持 LLM 自动生成 endpoint mapping 的模式；
  本版仅保留接口，不实现，不产生额外 token 成本。
"""

from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# 规范化：仅用于比较和 canonical 输出，不改变无约束模式下的任何原文
# ---------------------------------------------------------------------------

def _normalize(text: Any) -> str:
    """统一大小写、空白、连字符/下划线/斜杠，用于兼容性比较。

    例：
      "in_vitro_efficacy" / "in vitro efficacy" / "In-Vitro Efficacy" -> "in vitro efficacy"
      "Overall Survival"                                              -> "overall survival"
    """
    if text is None:
        return ""
    s = str(text).strip().lower()
    # 连字符（含 en/em dash）、下划线、斜杠统一为空格
    s = re.sub(r"[\-‐-―_/]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


class EndpointConstraint:
    """按 record_type 限制 endpoint 的确定性校验器（strict 模式）。"""

    def __init__(
        self,
        by_record_type: dict[str, list[str]],
        aliases: dict[str, list[str]] | None = None,
    ) -> None:
        # allowed[rt_norm] = { endpoint_norm: canonical_display }
        self.allowed: dict[str, dict[str, str]] = {}
        for rt, endpoints in (by_record_type or {}).items():
            rt_norm = _normalize(rt)
            ep_map: dict[str, str] = {}
            for ep in endpoints or []:
                ep_map[_normalize(ep)] = str(ep)
            self.allowed[rt_norm] = ep_map

        # alias_norm -> canonical_display（全局；canonical 必须出现在某个 record_type 列表里）
        self.aliases: dict[str, str] = {}
        for canonical, alias_list in (aliases or {}).items():
            for alias in alias_list or []:
                self.aliases[_normalize(alias)] = str(canonical)

    def resolve(self, record_type: Any, endpoint: Any) -> str | None:
        """返回兼容时的 canonical endpoint，否则返回 None。"""
        rt_norm = _normalize(record_type)
        allowed_map = self.allowed.get(rt_norm)
        if not allowed_map:
            return None

        ep_norm = _normalize(endpoint)
        if not ep_norm:
            return None

        # 1) 直接匹配该 record_type 的 canonical endpoint
        if ep_norm in allowed_map:
            return allowed_map[ep_norm]

        # 2) alias 解析：alias -> canonical，且 canonical 必须属于该 record_type 允许列表
        canonical = self.aliases.get(ep_norm)
        if canonical is not None:
            canonical_norm = _normalize(canonical)
            if canonical_norm in allowed_map:
                return allowed_map[canonical_norm]

        return None

    def apply(
        self,
        records: list[dict],
        *,
        record_type_field: str = "record_type",
        endpoint_field: str = "endpoint",
    ) -> tuple[list[dict], dict]:
        """过滤不兼容 record，并把兼容 record 的 endpoint 改写为 canonical 名称。

        返回 (kept_records, stats):
          - kept_records: 通过校验的 record（endpoint 已规范化为 canonical）
          - stats: {
                "endpoint_constraint_rejected": int,
                "endpoint_constraint_rejected_by_combo": {"<record_type> | <endpoint>": count},
            }
        """
        kept: list[dict] = []
        rejected = 0
        by_combo: dict[str, int] = {}

        for rec in records:
            if not isinstance(rec, dict):
                rejected += 1
                by_combo["<invalid> | <invalid>"] = by_combo.get("<invalid> | <invalid>", 0) + 1
                continue

            rt = rec.get(record_type_field)
            ep = rec.get(endpoint_field)
            canonical = self.resolve(rt, ep)

            if canonical is None:
                rejected += 1
                key = f"{rt} | {ep}"
                by_combo[key] = by_combo.get(key, 0) + 1
                continue

            if canonical != ep:
                rec = dict(rec)
                rec[endpoint_field] = canonical
            kept.append(rec)

        return kept, {
            "endpoint_constraint_rejected": rejected,
            "endpoint_constraint_rejected_by_combo": by_combo,
        }


def build_endpoint_constraint(config: Any) -> EndpointConstraint | None:
    """从 preset 的 endpoint_constraints 配置构建校验器。

    返回 None 表示无约束（unrestricted / 缺失 / 无效配置）——调用方应保持旧行为。

    仅当 mode == "strict" 且 by_record_type 非空时才返回校验器。
    """
    if not config or not isinstance(config, dict):
        return None

    mode = str(config.get("mode", "unrestricted")).strip().lower()
    if mode != "strict":
        return None

    by_record_type = config.get("by_record_type") or {}
    if not isinstance(by_record_type, dict) or not by_record_type:
        return None

    aliases = config.get("aliases") or {}
    if not isinstance(aliases, dict):
        aliases = {}

    return EndpointConstraint(by_record_type, aliases)
