"""Stage 3 post-processing utilities.

This module follows the ALLMAT post-processing idea with a smaller, generic
surface:

- LLM extraction keeps original evidence wording.
- Rule/preset based post-processing makes records easier to compare and export.
- Strict duplicate removal is used by default; fuzzy/LLM entity resolution is a
  later extension point.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import yaml


DEFAULT_NULL_VALUES = {
    "",
    "-",
    "—",
    "na",
    "n/a",
    "nr",
    "not reported",
    "not available",
    "none",
    "null",
}

NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
SPACE_RE = re.compile(r"\s+")


def load_postprocess_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load postprocess_config.yaml, returning an empty config when absent."""

    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"postprocess config not found: {path}")
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"postprocess config must be a mapping: {path}")
    return data


def resolve_numeric_fields(field_specs: list[Any], config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return numeric field config from YAML plus user_requirements field types."""

    numeric_fields: dict[str, dict[str, Any]] = {}
    for name, cfg in (config.get("numeric_fields") or {}).items():
        numeric_fields[str(name)] = dict(cfg or {})

    for field in field_specs:
        name = getattr(field, "name", None) or field.get("name")
        field_type = (getattr(field, "type", None) or field.get("type", "string")).lower()
        if field_type in {"number", "numeric", "float", "integer", "int"}:
            numeric_fields.setdefault(name, {})
            if field_type in {"integer", "int"}:
                numeric_fields[name].setdefault("integer", True)

    return numeric_fields


def clean_scalar(value: Any, null_values: set[str] | None = None) -> Any:
    """Normalize empty values and whitespace without changing semantic content."""

    nulls = null_values or DEFAULT_NULL_VALUES
    if value is None:
        return None
    if isinstance(value, list):
        cleaned = [clean_scalar(v, nulls) for v in value]
        return [v for v in cleaned if v is not None]
    if isinstance(value, dict):
        return value
    text = SPACE_RE.sub(" ", str(value).strip())
    if text.lower() in nulls:
        return None
    return text


def parse_numeric_value(
    raw: Any,
    default_unit: str | None = None,
    *,
    detect_unit: bool = True,
    parse_ranges: bool = True,
    allow_multiple_numbers: bool = False,
) -> dict[str, Any] | None:
    """Parse a value like '200-300 MPa', '<0.001', or '24.3 months'.

    The output is intentionally simple and CSV-friendly:

    {
      "raw": "200-300 MPa",
      "operator": "range",
      "value": 250.0,
      "value_min": 200.0,
      "value_max": 300.0,
      "unit": "MPa"
    }
    """

    cleaned = clean_scalar(raw)
    if cleaned is None or isinstance(cleaned, (list, dict)):
        return None

    text = _normalize_numeric_text(str(cleaned))
    unit = (_detect_unit(text) if detect_unit else None) or _canonical_unit(default_unit)

    # A leading estimate followed by a parenthesized range is common in clinical
    # reporting, e.g. "median OS 17.1 (0.6-61.9) months". The leading number is
    # the outcome; averaging the parenthesized range would change its meaning.
    primary_range_match = _find_primary_with_parenthesized_range(text)
    if primary_range_match:
        value, lower, upper = primary_range_match
        value, unit = _convert_one(value, unit, default_unit)
        lower, upper, unit = _convert_pair(lower, upper, unit, default_unit)
        return {
            "raw": str(cleaned),
            "operator": "reported_with_range",
            "value": _round_float(value),
            "value_min": _round_float(lower),
            "value_max": _round_float(upper),
            "error": None,
            "unit": unit,
        }

    pm_match = re.search(
        rf"({NUMBER_RE.pattern})\s*(?:±|\+/-|\+-)\s*({NUMBER_RE.pattern})",
        text,
        flags=re.I,
    )
    if pm_match:
        value = float(pm_match.group(1))
        error = abs(float(pm_match.group(2)))
        raw_unit = unit
        value, unit = _convert_one(value, raw_unit, default_unit)
        error, _ = _convert_one(error, raw_unit, default_unit)
        return {
            "raw": str(cleaned),
            "operator": "plus_minus",
            "value": _round_float(value),
            "value_min": None,
            "value_max": None,
            "error": _round_float(error),
            "unit": unit,
        }

    number_count = len(_measurement_numbers(text))
    range_match = _find_range(text) if parse_ranges else None
    if range_match:
        if not allow_multiple_numbers and number_count != 2:
            return None
        left, right = range_match
        left, right, unit = _convert_pair(left, right, unit, default_unit)
        return {
            "raw": str(cleaned),
            "operator": "range",
            "value": _round_float((left + right) / 2),
            "value_min": _round_float(left),
            "value_max": _round_float(right),
            "error": None,
            "unit": unit,
        }

    # Do not turn a compound statement such as "liver 7 (33%), lung 2 (10%)"
    # into one arbitrary scalar. The raw value remains available for review.
    if not allow_multiple_numbers and number_count != 1:
        return None

    operator = "eq"
    if re.search(r"(<=|≤)", text):
        operator = "<="
    elif re.search(r"(>=|≥)", text):
        operator = ">="
    elif re.search(r"(^|\s)<", text):
        operator = "<"
    elif re.search(r"(^|\s)>", text):
        operator = ">"
    elif re.search(r"(^|\s)(~|≈|about|around|approx\.?|approximately)\b", text, re.I):
        operator = "approx"

    number_match = NUMBER_RE.search(text)
    if not number_match:
        return None

    value = float(number_match.group(0))
    value, unit = _convert_one(value, unit, default_unit)
    result = {
        "raw": str(cleaned),
        "operator": operator,
        "value": _round_float(value),
        "value_min": None,
        "value_max": None,
        "error": None,
        "unit": unit,
    }
    if operator in {"<", "<="}:
        result["value_max"] = result["value"]
    elif operator in {">", ">="}:
        result["value_min"] = result["value"]
    return result


def standardize_value(raw: Any, field_config: dict[str, Any], multiple: bool = False) -> Any:
    """Map synonyms to canonical labels using a preset config."""

    value = clean_scalar(raw)
    if value is None:
        return None

    terms = field_config.get("terms", field_config)
    if not isinstance(terms, dict):
        return value

    multiple = bool(field_config.get("multiple", multiple))
    match_mode = str(field_config.get("match", "contains")).lower()
    aliases: list[tuple[str, str]] = []
    for canonical, synonyms in terms.items():
        aliases.append((str(canonical), _compact(canonical)))
        if isinstance(synonyms, str):
            synonyms = [synonyms]
        for synonym in synonyms or []:
            aliases.append((str(canonical), _compact(synonym)))

    if isinstance(value, list):
        return _dedupe_preserve_order(
            standardize_value(v, field_config, multiple=False) for v in value
        )

    text = str(value).strip()
    compact_text = _compact(text)

    exact_matches = [canonical for canonical, alias in aliases if compact_text == alias]
    if exact_matches:
        return exact_matches[0]

    if multiple:
        hits = []
        for canonical, alias in aliases:
            if alias and alias in compact_text:
                hits.append(canonical)
        hits = _dedupe_preserve_order(hits)
        return hits if hits else text

    if match_mode != "exact":
        for canonical, alias in aliases:
            if alias and alias in compact_text:
                return canonical
    return text


def postprocess_records(
    records: list[dict[str, Any]],
    field_specs: list[Any],
    config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Clean, standardize, validate, and strictly deduplicate extracted records."""

    config = config or {}
    field_names = [getattr(f, "name", None) or f.get("name") for f in field_specs]
    field_types = {
        (getattr(f, "name", None) or f.get("name")):
        (getattr(f, "type", None) or f.get("type", "string")).lower()
        for f in field_specs
    }
    numeric_fields = resolve_numeric_fields(field_specs, config)
    standardize_cfg = config.get("standardize") or {}
    validity_cfg = config.get("validity") or {}
    required_all = list(validity_cfg.get("required_all") or [])
    required_any = list(validity_cfg.get("required_any") or field_names)
    null_values = DEFAULT_NULL_VALUES | {
        str(v).lower() for v in (config.get("null_values") or [])
    }

    cleaned_rows: list[dict[str, Any]] = []
    invalid_removed = 0
    numeric_parsed = 0
    standardized_values = 0

    for record in records:
        row: dict[str, Any] = {
            "paper_id": record.get("paper_id"),
            "record_id": record.get("record_id"),
        }

        for field in field_names:
            value = clean_scalar(record.get(field), null_values)
            before = value

            field_std_cfg = standardize_cfg.get(field)
            if field_std_cfg:
                value = standardize_value(
                    value,
                    field_std_cfg,
                    multiple=field_types.get(field) == "list",
                )
                if value != before:
                    standardized_values += 1

            row[field] = value

            if field in numeric_fields:
                field_numeric_cfg = numeric_fields[field]
                default_unit = field_numeric_cfg.get("unit")
                unit_source = field_numeric_cfg.get("unit_from_field")
                if unit_source:
                    source_value = clean_scalar(record.get(unit_source), null_values)
                    if source_value is not None:
                        default_unit = _detect_unit(str(source_value)) or default_unit
                norm = parse_numeric_value(
                    before,
                    default_unit,
                    detect_unit=field_numeric_cfg.get("detect_unit", True),
                    parse_ranges=field_numeric_cfg.get("parse_ranges", True),
                    allow_multiple_numbers=field_numeric_cfg.get(
                        "allow_multiple_numbers", False
                    ),
                )
                if norm is not None:
                    if numeric_fields[field].get("integer") and norm.get("value") is not None:
                        norm["value"] = int(round(norm["value"]))
                    row[f"{field}_norm"] = norm
                    numeric_parsed += 1

        row["source_chunk_ids"] = list(record.get("source_chunk_ids") or [])

        if not _is_valid(row, required_all, required_any):
            invalid_removed += 1
            continue

        cleaned_rows.append(row)

    deduped_rows, duplicates_removed = strict_deduplicate(cleaned_rows, field_names)

    summary = {
        "records_input": len(records),
        "records_after_validation": len(cleaned_rows),
        "records_output": len(deduped_rows),
        "invalid_removed": invalid_removed,
        "duplicates_removed": duplicates_removed,
        "numeric_values_parsed": numeric_parsed,
        "standardized_values": standardized_values,
        "numeric_fields": sorted(f for f in numeric_fields if f in field_names),
        "standardized_fields": sorted(f for f in standardize_cfg if f in field_names),
    }
    return deduped_rows, summary


def strict_deduplicate(
    records: list[dict[str, Any]],
    field_names: list[str],
) -> tuple[list[dict[str, Any]], int]:
    """Strict duplicate removal using post-processed field values."""

    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        key = (
            record.get("paper_id"),
            *(_dedupe_key_value(record, field) for field in field_names),
        )
        if key not in seen:
            seen[key] = record
            continue

        existing = seen[key]
        source_ids = _dedupe_preserve_order(
            list(existing.get("source_chunk_ids") or [])
            + list(record.get("source_chunk_ids") or [])
        )
        existing["source_chunk_ids"] = source_ids

    return list(seen.values()), len(records) - len(seen)


def write_records_csv(
    path: str | Path,
    records: list[dict[str, Any]],
    field_specs: list[Any],
    numeric_fields: dict[str, dict[str, Any]],
) -> None:
    """Write a flat CSV for downstream analysis."""

    path = Path(path)
    field_names = [getattr(f, "name", None) or f.get("name") for f in field_specs]
    columns = ["paper_id", "record_id", *field_names]
    for field in field_names:
        if field in numeric_fields:
            columns.extend([
                f"{field}_value",
                f"{field}_unit",
                f"{field}_operator",
                f"{field}_value_min",
                f"{field}_value_max",
                f"{field}_error",
            ])
    columns.append("source_chunk_ids")

    # Excel on macOS/Windows reliably detects UTF-8 when the CSV includes a BOM.
    # JSONL remains plain UTF-8; this applies only to the spreadsheet export.
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for record in records:
            row: dict[str, Any] = {}
            for column in columns:
                row[column] = ""
            row["paper_id"] = record.get("paper_id") or ""
            row["record_id"] = record.get("record_id") or ""
            for field in field_names:
                row[field] = _csv_value(record.get(field))
                norm = record.get(f"{field}_norm") or {}
                if norm:
                    row[f"{field}_value"] = _csv_value(norm.get("value"))
                    row[f"{field}_unit"] = _csv_value(norm.get("unit"))
                    row[f"{field}_operator"] = _csv_value(norm.get("operator"))
                    row[f"{field}_value_min"] = _csv_value(norm.get("value_min"))
                    row[f"{field}_value_max"] = _csv_value(norm.get("value_max"))
                    row[f"{field}_error"] = _csv_value(norm.get("error"))
            row["source_chunk_ids"] = ";".join(record.get("source_chunk_ids") or [])
            writer.writerow(row)


def _normalize_numeric_text(text: str) -> str:
    text = text.replace("\u2212", "-")
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("≤", "<=").replace("≥", ">=")
    text = text.replace("≈", "~")
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = re.sub(r"(?<=\d)\s+(?=\d{3}\b)", "", text)
    return SPACE_RE.sub(" ", text.strip())


def _find_range(text: str) -> tuple[float, float] | None:
    range_re = re.compile(
        rf"({NUMBER_RE.pattern})\s*(?:-|to|~|,)\s*({NUMBER_RE.pattern})",
        flags=re.I,
    )
    matches = list(range_re.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    left = float(match.group(1))
    right = float(match.group(2))
    if left > right:
        left, right = right, left
    return left, right


def _find_primary_with_parenthesized_range(text: str) -> tuple[float, float, float] | None:
    match = re.search(
        rf"({NUMBER_RE.pattern})\s*\(\s*({NUMBER_RE.pattern})\s*(?:-|to|~)\s*({NUMBER_RE.pattern})\s*\)",
        text,
        flags=re.I,
    )
    if not match:
        return None
    value = float(match.group(1))
    lower = float(match.group(2))
    upper = float(match.group(3))
    if lower > upper:
        lower, upper = upper, lower
    return value, lower, upper


def _measurement_numbers(text: str) -> list[str]:
    """Return number tokens while ignoring the 2 in dose units such as mg/m2."""
    matches = []
    for match in NUMBER_RE.finditer(text):
        if (
            match.group(0) == "2"
            and match.start() > 0
            and text[match.start() - 1].lower() == "m"
        ):
            continue
        matches.append(match.group(0))
    return matches


def _detect_unit(text: str) -> str | None:
    lowered = text.lower()
    concentration_match = re.search(r"\b(nM|uM|µM|μM|mM)\b", text)
    if concentration_match:
        raw = concentration_match.group(1)
        return {"µM": "uM", "μM": "uM"}.get(raw, raw)
    dose_area_match = re.search(
        r"\b(?:ng|ug|µg|μg|mg|g)\s*/\s*m(?:\^?2|²)\b", text, flags=re.I
    )
    if dose_area_match:
        return _canonical_unit(dose_area_match.group(0))
    if re.search(r"\b(ng|ug|µg|μg|mg)/m[lL]\b", text):
        return _canonical_unit(re.search(r"\b(ng|ug|µg|μg|mg)/m[lL]\b", text).group(0))
    if re.search(r"\b(mg|ug|µg|μg)/kg(?:/day)?\b", lowered):
        return _canonical_unit(re.search(r"\b(mg|ug|µg|μg)/kg(?:/day)?\b", lowered).group(0))
    if re.search(r"\bml/min/kg\b", lowered):
        return "mL/min/kg"
    if "%" in text or re.search(r"\bpercent(age)?\b", lowered):
        return "percent"
    if re.search(r"\bmonths?\b|\bmos?\b", lowered):
        return "month"
    if re.search(r"\byears?\b|\byrs?\b", lowered):
        return "year"
    if re.search(r"\bweeks?\b|\bwks?\b", lowered):
        return "week"
    if re.search(r"\bdays?\b", lowered):
        return "day"
    if re.search(r"\bhours?\b|\bhrs?\b|\bh\b", lowered):
        return "h"
    if re.search(r"\bmin(utes?|s)?\b", lowered):
        return "min"
    if re.search(r"\bgpa\b", lowered):
        return "GPa"
    if re.search(r"\bmpa\b", lowered):
        return "MPa"
    if re.search(r"\bnm\b", lowered):
        return "nm"
    if re.search(r"\b(mm|millimeters?)\b", lowered):
        return "mm"
    if re.search(r"\b(um|μm|µm|micrometers?|microns?)\b", lowered):
        return "um"
    if re.search(r"°\s*c\b|\bcelsius\b|\bdeg(?:ree)?\s*c\b|\b\d+\s*c\b", lowered):
        return "C"
    return None


def _canonical_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    lookup = {
        "%": "percent",
        "percentage": "percent",
        "ug/ml": "ug/mL",
        "µg/ml": "ug/mL",
        "μg/ml": "ug/mL",
        "ng/ml": "ng/mL",
        "mg/ml": "mg/mL",
        "mg/kg/day": "mg/kg/day",
        "mg/m2": "mg/m2",
        "mg/m^2": "mg/m2",
        "mg/m²": "mg/m2",
        "ug/kg": "ug/kg",
        "µg/kg": "ug/kg",
        "μg/kg": "ug/kg",
        "months": "month",
        "mo": "month",
        "mos": "month",
        "years": "year",
        "yr": "year",
        "yrs": "year",
        "weeks": "week",
        "wks": "week",
        "days": "day",
        "hours": "h",
        "hrs": "h",
        "minute": "min",
        "minutes": "min",
        "mins": "min",
        "μm": "um",
        "µm": "um",
        "micrometer": "um",
        "micrometers": "um",
        "micron": "um",
        "microns": "um",
        "celsius": "C",
        "°c": "C",
    }
    raw = str(unit).strip()
    return lookup.get(raw.lower(), raw)


def _convert_pair(
    left: float,
    right: float,
    unit: str | None,
    default_unit: str | None,
) -> tuple[float, float, str | None]:
    left, unit = _convert_one(left, unit, default_unit)
    right, unit = _convert_one(right, unit, default_unit)
    return left, right, unit


def _convert_one(
    value: float,
    unit: str | None,
    default_unit: str | None,
) -> tuple[float, str | None]:
    unit = _canonical_unit(unit)
    target = _canonical_unit(default_unit)
    if not target:
        return value, unit
    if not unit:
        return value, target
    if unit == target:
        return value, target

    factors = {
        ("GPa", "MPa"): 1000.0,
        ("MPa", "GPa"): 0.001,
        ("year", "month"): 12.0,
        ("month", "year"): 1 / 12.0,
        ("nm", "um"): 0.001,
        ("mm", "um"): 1000.0,
        ("um", "nm"): 1000.0,
    }
    factor = factors.get((unit, target))
    if factor is None:
        return value, unit
    return value * factor, target


def _is_valid(row: dict[str, Any], required_all: list[str], required_any: list[str]) -> bool:
    for field in required_all:
        if _is_empty(row.get(field)) and _is_empty(row.get(f"{field}_norm")):
            return False
    if required_any:
        return any(
            not _is_empty(row.get(field)) or not _is_empty(row.get(f"{field}_norm"))
            for field in required_any
        )
    return True


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False


def _dedupe_key_value(record: dict[str, Any], field: str) -> str:
    norm = record.get(f"{field}_norm")
    if norm is not None:
        comparable = {k: v for k, v in norm.items() if k != "raw"}
        return json.dumps(comparable, ensure_ascii=False, sort_keys=True)
    value = record.get(field)
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True).lower()
    return SPACE_RE.sub(" ", str(value or "").strip()).lower()


def _compact(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _dedupe_preserve_order(values) -> list[Any]:
    result = []
    seen = set()
    for value in values:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (list, dict)) else str(value)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _round_float(value: float) -> float:
    rounded = round(float(value), 10)
    if rounded == int(rounded):
        return float(int(rounded))
    return rounded
