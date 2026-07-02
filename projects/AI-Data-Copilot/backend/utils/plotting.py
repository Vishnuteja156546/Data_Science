from __future__ import annotations

import pandas as pd
from backend.config import settings
from backend.utils.helpers import json_safe


def chart_config(chart_type: str, title: str, labels: list, datasets: list[dict]) -> dict:
    return json_safe({"type": chart_type, "title": title, "labels": labels, "datasets": datasets})


def histogram_config(frame: pd.DataFrame, column: str) -> dict:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return {}
    counts, edges = pd.cut(values, bins=min(settings.histogram_bins, max(3, values.nunique())), retbins=True)
    data = counts.value_counts().sort_index()
    labels = [f"{round(interval.left, 2)} to {round(interval.right, 2)}" for interval in data.index]
    return chart_config("bar", f"Distribution of {column}", labels, [{"label": column, "data": data.tolist()}])


def bar_config(frame: pd.DataFrame, column: str) -> dict:
    counts = frame[column].astype(str).fillna("Missing").value_counts().head(15)
    return chart_config("bar", f"Top values in {column}", counts.index.tolist(), [{"label": "Count", "data": counts.tolist()}])


def scatter_config(frame: pd.DataFrame, x: str, y: str) -> dict:
    sample = frame[[x, y]].dropna().head(settings.max_plot_points)
    points = [{"x": row[x], "y": row[y]} for _, row in sample.iterrows()]
    return chart_config("scatter", f"{y} by {x}", [], [{"label": f"{x} vs {y}", "data": points}])


def line_config(frame: pd.DataFrame, x: str, y: str, title: str | None = None) -> dict:
    sample = frame[[x, y]].dropna().head(settings.max_plot_points)
    return chart_config("line", title or f"{y} over {x}", sample[x].astype(str).tolist(), [{"label": y, "data": sample[y].tolist()}])


def correlation_config(frame: pd.DataFrame) -> dict:
    numeric = frame.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return {}
    corr = numeric.corr(numeric_only=True).round(3)
    labels = corr.columns.tolist()
    rows = [{"column": index, "values": corr.loc[index].tolist()} for index in corr.index]
    return json_safe({"type": "heatmap", "title": "Correlation Heatmap", "labels": labels, "rows": rows})
