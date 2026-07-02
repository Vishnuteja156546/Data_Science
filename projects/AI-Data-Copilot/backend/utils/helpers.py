from __future__ import annotations

import math
from typing import Any
import numpy as np
import pandas as pd


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def preview_records(frame: pd.DataFrame, rows: int) -> list[dict[str, Any]]:
    return json_safe(frame.head(rows).replace({np.nan: None}).to_dict(orient="records"))


def choose_default_target(frame: pd.DataFrame) -> str | None:
    candidates = [col for col in frame.columns if frame[col].nunique(dropna=True) > 1]
    if not candidates:
        return None
    for col in reversed(candidates):
        lower = col.lower()
        if any(word in lower for word in ("target", "label", "price", "sales", "revenue", "amount", "churn")):
            return col
    return candidates[-1]


def compact_context(profile: dict, sample: list[dict]) -> str:
    columns = profile.get("columns", [])
    parts = [
        f"Rows: {profile.get('rows')}",
        f"Columns: {profile.get('column_count')}",
        "Column profiles:",
    ]
    for col in columns[:30]:
        parts.append(
            f"- {col['name']}: {col['semantic_type']}, missing {col['missing_count']}, unique {col['unique_count']}"
        )
    parts.append(f"Sample rows: {sample}")
    return "\n".join(parts)
