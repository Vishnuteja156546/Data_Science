from __future__ import annotations

import numpy as np
import pandas as pd
import warnings
from backend.utils.helpers import json_safe
from backend.utils.plotting import bar_config, correlation_config, histogram_config, line_config, scatter_config


def detect_semantic_type(series: pd.Series, name: str) -> str:
    lower = name.lower()
    non_null = series.dropna()
    unique_count = non_null.nunique()
    if unique_count == len(non_null) and unique_count > 0 and any(token in lower for token in ("id", "uuid", "key")):
        return "id"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if series.dtype == object:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(non_null.head(50), errors="coerce")
        if len(parsed) and parsed.notna().mean() > 0.8:
            return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        if unique_count <= max(20, int(len(series) * 0.05)):
            return "numeric_discrete"
        return "numeric_continuous"
    avg_len = non_null.astype(str).str.len().mean() if len(non_null) else 0
    if avg_len and avg_len > 50:
        return "free_text"
    return "categorical"


def profile_dataframe(frame: pd.DataFrame) -> dict:
    duplicate_count = int(frame.duplicated().sum())
    columns = []
    for name in frame.columns:
        series = frame[name]
        semantic_type = detect_semantic_type(series, name)
        item = {
            "name": name,
            "dtype": str(series.dtype),
            "semantic_type": semantic_type,
            "missing_count": int(series.isna().sum()),
            "missing_percent": round(float(series.isna().mean() * 100), 2),
            "unique_count": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            q1 = numeric.quantile(0.25) if len(numeric) else np.nan
            q3 = numeric.quantile(0.75) if len(numeric) else np.nan
            iqr = q3 - q1 if len(numeric) else 0
            outliers = numeric[(numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)] if iqr else []
            item.update(
                {
                    "mean": numeric.mean() if len(numeric) else None,
                    "median": numeric.median() if len(numeric) else None,
                    "min": numeric.min() if len(numeric) else None,
                    "max": numeric.max() if len(numeric) else None,
                    "std": numeric.std() if len(numeric) else None,
                    "outlier_count": int(len(outliers)),
                }
            )
        else:
            top = series.astype(str).value_counts().head(5)
            item["top_values"] = top.to_dict()
        columns.append(item)
    missing_cells = int(frame.isna().sum().sum())
    total_cells = int(frame.shape[0] * frame.shape[1])
    health = max(0, 100 - round((missing_cells / max(total_cells, 1)) * 60) - min(30, duplicate_count))
    return json_safe(
        {
            "rows": int(frame.shape[0]),
            "column_count": int(frame.shape[1]),
            "columns": columns,
            "duplicate_rows": duplicate_count,
            "missing_cells": missing_cells,
            "health_score": health,
        }
    )


def build_eda_charts(frame: pd.DataFrame, profile: dict) -> list[dict]:
    charts: list[dict] = []
    numeric = [col["name"] for col in profile["columns"] if col["semantic_type"].startswith("numeric")]
    categoricals = [col["name"] for col in profile["columns"] if col["semantic_type"] == "categorical"]
    datetimes = [col["name"] for col in profile["columns"] if col["semantic_type"] == "datetime"]
    for column in numeric[:3]:
        chart = histogram_config(frame, column)
        if chart:
            charts.append(chart)
    for column in categoricals[:3]:
        chart = bar_config(frame, column)
        if chart:
            charts.append(chart)
    if len(numeric) >= 2:
        charts.append(scatter_config(frame, numeric[0], numeric[1]))
        corr = correlation_config(frame)
        if corr:
            charts.append(corr)
    if datetimes and numeric:
        temp = frame.copy()
        temp[datetimes[0]] = pd.to_datetime(temp[datetimes[0]], errors="coerce")
        temp = temp.sort_values(datetimes[0])
        charts.append(line_config(temp, datetimes[0], numeric[0]))
    return charts
