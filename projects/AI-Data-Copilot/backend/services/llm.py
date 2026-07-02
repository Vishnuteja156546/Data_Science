from __future__ import annotations

import json
import re
import httpx
from backend.config import settings
from backend.utils.helpers import compact_context


SYSTEM_PROMPT = (
    "You are AI Data Copilot, a practical data analyst and data scientist. "
    "Use only the dataset context provided. Do not invent numbers. "
    "When chart data is needed, describe the chart intent rather than fabricating values."
)


async def call_groq(messages: list[dict]) -> str | None:
    if not settings.groq_api_key:
        return None
    headers = {"Authorization": f"Bearer {settings.groq_api_key}", "Content-Type": "application/json"}
    payload = {"model": settings.groq_model, "messages": messages, "temperature": 0.2}
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(settings.groq_api_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


def fallback_insights(profile: dict) -> dict:
    columns = profile.get("columns", [])
    missing = sorted(columns, key=lambda col: col.get("missing_percent", 0), reverse=True)[:3]
    numeric = [col for col in columns if col.get("semantic_type", "").startswith("numeric")]
    categorical = [col for col in columns if col.get("semantic_type") == "categorical"]
    summary = [
        f"The dataset has {profile.get('rows')} rows and {profile.get('column_count')} columns.",
        f"It contains {len(numeric)} numeric columns and {len(categorical)} categorical columns.",
        f"There are {profile.get('duplicate_rows')} duplicate rows and {profile.get('missing_cells')} missing cells.",
    ]
    recommendations = ["Review high-missing columns before modeling.", "Use categorical breakdowns for business segmentation."]
    if missing and missing[0].get("missing_percent", 0) > 0:
        recommendations.insert(0, f"Prioritize cleaning {missing[0]['name']} because it has the most missing values.")
    return {"summary": summary, "kpis": {"health_score": profile.get("health_score"), "rows": profile.get("rows")}, "recommendations": recommendations}


async def dataset_summary(profile: dict, sample: list[dict]) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Create concise business-friendly JSON with summary, kpis, risks, opportunities, recommendations.\n{compact_context(profile, sample)}"},
    ]
    try:
        content = await call_groq(messages)
        if content:
            match = re.search(r"\{.*\}", content, re.S)
            return json.loads(match.group(0) if match else content)
    except Exception:
        return fallback_insights(profile)
    return fallback_insights(profile)


def extract_chart_intent(message: str, columns: list[str]) -> dict | None:
    text = message.lower()
    if not any(word in text for word in ("plot", "chart", "graph", "show", "visualize")):
        return None
    selected = [col for col in columns if col.lower() in text]
    chart_type = "bar"
    if "scatter" in text:
        chart_type = "scatter"
    elif "line" in text or "trend" in text:
        chart_type = "line"
    elif "hist" in text or "distribution" in text:
        chart_type = "histogram"
    return {"chart_type": chart_type, "x": selected[0] if selected else None, "y": selected[1] if len(selected) > 1 else None}


async def chat_response(message: str, profile: dict, sample: list[dict], history: list[dict]) -> str:
    recent = history[-settings.max_chat_history :]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *recent, {"role": "user", "content": f"{compact_context(profile, sample)}\n\nQuestion: {message}"}]
    try:
        content = await call_groq(messages)
        if content:
            return content
    except Exception as exc:
        return f"I could not reach Groq right now, so here is a local answer. The dataset has {profile.get('rows')} rows, {profile.get('column_count')} columns, and health score {profile.get('health_score')}. API detail: {exc}"
    return f"Groq is not configured yet. Locally, I can see {profile.get('rows')} rows, {profile.get('column_count')} columns, {profile.get('duplicate_rows')} duplicate rows, and {profile.get('missing_cells')} missing cells."
