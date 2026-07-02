from __future__ import annotations


def business_report(session) -> str:
    profile = session.profile or {}
    summary = session.summary or {}
    lines = [
        "# AI Data Copilot Business Report",
        "",
        f"Dataset: {session.filename}",
        f"Rows: {profile.get('rows')}",
        f"Columns: {profile.get('column_count')}",
        f"Health score: {profile.get('health_score')}",
        "",
        "## Summary",
    ]
    for item in summary.get("summary", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Recommendations"])
    for item in summary.get("recommendations", []):
        lines.append(f"- {item}")
    return "\n".join(lines)


def eda_report(session) -> str:
    profile = session.profile or {}
    lines = ["# EDA Report", "", f"Rows: {profile.get('rows')}", f"Columns: {profile.get('column_count')}", ""]
    for column in profile.get("columns", []):
        lines.append(f"## {column['name']}")
        lines.append(f"- Type: {column['semantic_type']}")
        lines.append(f"- Missing: {column['missing_count']} ({column['missing_percent']}%)")
        lines.append(f"- Unique: {column['unique_count']}")
    return "\n".join(lines)


def prediction_report(session) -> str:
    result = session.model_result or {}
    lines = ["# Prediction Report", "", f"Target: {result.get('target')}", f"Task: {result.get('task')}", "", "## Metrics"]
    for key, value in result.get("metrics", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Feature Importance"])
    for item in result.get("feature_importance", []):
        lines.append(f"- {item['feature']}: {round(item['importance'], 4)}")
    return "\n".join(lines)
