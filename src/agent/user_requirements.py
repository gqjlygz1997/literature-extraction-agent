"""User-facing minimal requirements schema and loader.

Users write experiments/<domain>/user_requirements.yaml; this module
loads it into a UserRequirements object that ConfigGenerator consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class FieldSpec:
    """One target field as described by the user.

    Stage 1 keeps this intentionally minimal:
      - name and definition are required
      - type is optional and defaults to "string"
    """

    name: str
    definition: str
    type: str = "string"


@dataclass
class RecordSpec:
    """Final record shape requested by the user.

    fields is the complete set of fields that should appear in each final
    extracted record. In the simplified MVP, any field value change may create
    a separate record during extraction/post-processing.
    """

    name: str
    meaning: str
    fields: list[FieldSpec]


@dataclass
class UserRequirements:
    """Parsed content of user_requirements.yaml."""

    project_name: str
    domain_description: str
    target_fields: list[FieldSpec]
    record: RecordSpec | None = None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_user_requirements(path: str | Path) -> UserRequirements:
    """Load and validate a user_requirements.yaml file."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"user_requirements.yaml not found: {path}")

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ValueError(f"user_requirements.yaml must be a YAML mapping, got {type(data)}")

    # required top-level keys
    for key in ("project_name", "domain_description"):
        if key not in data:
            raise ValueError(f"user_requirements.yaml missing required key: '{key}'")

    record = _parse_record(data.get("record"))
    if record is not None:
        target_fields = record.fields
    else:
        if "target_fields" not in data:
            raise ValueError(
                "user_requirements.yaml must define either 'record.fields' "
                "or legacy 'target_fields'"
            )
        target_fields = _parse_fields(data["target_fields"], "target_fields")

    if not target_fields:
        raise ValueError("record.fields / target_fields must not be empty")

    return UserRequirements(
        project_name=str(data["project_name"]),
        domain_description=str(data["domain_description"]).strip(),
        target_fields=target_fields,
        record=record,
    )


def _parse_record(record_data) -> RecordSpec | None:
    """Parse optional record block from user_requirements.yaml."""

    if record_data is None:
        return None
    if not isinstance(record_data, dict):
        raise ValueError(f"record must be a YAML mapping, got {type(record_data)}")

    name = str(record_data.get("name", "")).strip()
    meaning = str(record_data.get("meaning", record_data.get("description", ""))).strip()
    if not name:
        raise ValueError("record.name is required")
    if not meaning:
        raise ValueError("record.meaning is required")
    if "fields" not in record_data:
        raise ValueError("record.fields is required")

    fields = _parse_fields(record_data["fields"], "record.fields")
    return RecordSpec(name=name, meaning=meaning, fields=fields)


def _parse_fields(items, source: str) -> list[FieldSpec]:
    """Parse a list of field specs from record.fields or legacy target_fields."""

    if not isinstance(items, list):
        raise ValueError(f"{source} must be a list")

    fields: list[FieldSpec] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(
                f"{source} items must be mappings with required keys "
                f"'name' and 'definition'; got item {index}: {item!r}"
            )

        name = str(item.get("name", "")).strip()
        definition = str(item.get("definition", item.get("description", ""))).strip()
        if not name:
            raise ValueError(f"{source}[{index}].name is required")
        if not definition:
            raise ValueError(f"{source}[{index}].definition is required")

        fields.append(FieldSpec(
            name=name,
            definition=definition,
            type=str(item.get("type", "string")).strip() or "string",
        ))

    return fields


def build_user_requirements_from_args(
    domain: str,
    fields: str,
    field_definitions: str = "",
) -> UserRequirements:
    """Build a UserRequirements object from CLI --domain / --fields args.

    fields: comma-separated field names, e.g. "treatment_regimen,os,pfs"
    field_definitions: semicolon-separated key=value pairs; every field needs
                       a definition, e.g. "os=overall survival;pfs=progression-free survival"
    """
    defs: dict[str, str] = {}
    if field_definitions:
        for pair in field_definitions.split(";"):
            pair = pair.strip()
            if "=" in pair:
                k, _, v = pair.partition("=")
                defs[k.strip()] = v.strip()

    field_list = [f.strip() for f in fields.split(",") if f.strip()]
    if not field_list:
        raise ValueError("--fields must contain at least one field name")

    target_fields = [
        FieldSpec(name=name, definition=defs.get(name, ""))
        for name in field_list
    ]
    missing_defs = [f.name for f in target_fields if not f.definition]
    if missing_defs:
        raise ValueError(
            "--field-definitions must define every --fields item. "
            f"Missing: {', '.join(missing_defs)}"
        )

    return UserRequirements(
        project_name="cli_run",
        domain_description=domain.strip(),
        target_fields=target_fields,
        record=RecordSpec(
            name="cli_record",
            meaning=(
                "One record contains all requested fields. If any field value "
                "differs, extraction may create a separate record."
            ),
            fields=target_fields,
        ),
    )
