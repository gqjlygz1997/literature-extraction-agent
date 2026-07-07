"""Preset discovery helpers.

Preset files let a project use hand-written, ALLMAT-style stable configs before
falling back to DSPy/LLM generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def default_preset_dir() -> Path:
    """Return the repository-level preset directory."""

    return Path(__file__).resolve().parents[2] / "presets"


def load_project_name(requirements_path: str | Path) -> str:
    """Read project_name from user_requirements.yaml without full validation."""

    with open(requirements_path, encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    project_name = str(data.get("project_name", "")).strip()
    if not project_name:
        raise ValueError(f"user_requirements.yaml missing project_name: {requirements_path}")
    return project_name


def find_preset_file(
    requirements_path: str | Path,
    filename: str,
    preset_dir: str | Path | None = None,
) -> Path | None:
    """Find presets/<project_name>/<filename>, returning None if absent."""

    root = Path(preset_dir) if preset_dir else default_preset_dir()
    project_name = load_project_name(requirements_path)
    candidate = root / project_name / filename
    return candidate if candidate.exists() else None


def render_prompt_template(text: str, record: object) -> str:
    """Render simple {{...}} placeholders in extraction prompt presets."""

    fields = getattr(record, "fields", [])
    field_lines = []
    for field in fields:
        name = getattr(field, "name", "")
        definition = getattr(field, "definition", "")
        field_lines.append(f"- {name}: {definition}")

    replacements = {
        "{{record_name}}": getattr(record, "name", ""),
        "{{record_meaning}}": str(getattr(record, "meaning", "")).strip(),
        "{{field_list}}": "\n".join(field_lines),
    }
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text
