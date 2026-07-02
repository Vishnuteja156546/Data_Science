from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
import pandas as pd

from backend.config import FRONTEND_DIR, settings
from backend.model import AutoMLRequest, ChatRequest, ForecastRequest, ReportRequest
from backend.services.automl import run_automl
from backend.services.cleaning import clean_dataframe
from backend.services.eda import build_eda_charts, profile_dataframe
from backend.services.exports import dataframe_csv_bytes
from backend.services.llm import chat_response, dataset_summary, extract_chart_intent
from backend.services.reports import business_report, eda_report, prediction_report
from backend.store import store
from backend.utils.helpers import json_safe, preview_records
from backend.utils.plotting import bar_config, histogram_config, line_config, scatter_config
from backend.utils.validators import read_dataset_bytes


app = FastAPI(title="AI Data Copilot", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


def session_or_404(session_id: str):
    try:
        return store.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Dataset session not found.") from exc


def build_chart_from_intent(frame: pd.DataFrame, intent: dict) -> dict | None:
    chart_type = intent.get("chart_type")
    x = intent.get("x")
    y = intent.get("y")
    if chart_type == "histogram" and x in frame.columns:
        return histogram_config(frame, x)
    if chart_type == "bar" and x in frame.columns:
        return bar_config(frame, x)
    if chart_type == "scatter" and x in frame.columns and y in frame.columns:
        return scatter_config(frame, x, y)
    if chart_type == "line" and x in frame.columns and y in frame.columns:
        return line_config(frame, x, y)
    return None


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "groq_configured": bool(settings.groq_api_key)}


@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict:
    content = await file.read()
    try:
        frame = read_dataset_bytes(file.filename or "dataset.csv", content, settings.max_upload_rows)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    session = store.create(file.filename or "dataset.csv", frame)
    session.profile = profile_dataframe(frame)
    session.charts = build_eda_charts(frame, session.profile)
    return {
        "session_id": session.session_id,
        "filename": session.filename,
        "profile": session.profile,
        "preview": preview_records(frame, settings.max_preview_rows),
        "charts": session.charts,
    }


@app.get("/api/{session_id}/summary")
async def summary(session_id: str) -> dict:
    session = session_or_404(session_id)
    session.summary = await dataset_summary(session.profile, preview_records(store.frame(session_id), settings.max_sample_rows))
    return json_safe(session.summary)


@app.post("/api/clean")
def clean(request: AutoMLRequest) -> dict:
    session = session_or_404(request.session_id)
    cleaned, log = clean_dataframe(store.frame(request.session_id, cleaned=False))
    session.cleaned_dataframe = cleaned
    session.profile = profile_dataframe(cleaned)
    session.charts = build_eda_charts(cleaned, session.profile)
    return {"log": log, "profile": session.profile, "preview": preview_records(cleaned, settings.max_preview_rows), "charts": session.charts}


@app.post("/api/automl")
def automl(request: AutoMLRequest) -> dict:
    session_or_404(request.session_id)
    try:
        result = run_automl(store.frame(request.session_id), request.target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    store.get(request.session_id).model_result = result
    return result


@app.post("/api/forecast")
def forecast(request: ForecastRequest) -> dict:
    session = session_or_404(request.session_id)
    frame = store.frame(request.session_id)
    date_col = request.date_column
    value_col = request.value_column
    profile = session.profile
    if not date_col:
        date_col = next((col["name"] for col in profile["columns"] if col["semantic_type"] == "datetime"), None)
    if not value_col:
        value_col = next((col["name"] for col in profile["columns"] if col["semantic_type"].startswith("numeric")), None)
    if not date_col or not value_col:
        raise HTTPException(status_code=400, detail="Forecasting needs a datetime column and a numeric value column.")
    data = frame[[date_col, value_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna().sort_values(date_col)
    if len(data) < 4:
        raise HTTPException(status_code=400, detail="Forecasting needs at least 4 valid time-series rows.")
    grouped = data.groupby(date_col)[value_col].sum().reset_index()
    x = range(len(grouped))
    coef = pd.Series(grouped[value_col]).rolling(3, min_periods=1).mean()
    slope = (coef.iloc[-1] - coef.iloc[0]) / max(len(coef) - 1, 1)
    last_date = grouped[date_col].iloc[-1]
    freq = pd.infer_freq(grouped[date_col]) or "D"
    future_dates = pd.date_range(last_date, periods=request.periods + 1, freq=freq)[1:]
    forecast_values = [float(coef.iloc[-1] + slope * (i + 1)) for i in range(request.periods)]
    result = {
        "date_column": date_col,
        "value_column": value_col,
        "history": [{"date": str(row[date_col].date()), "value": row[value_col]} for _, row in grouped.tail(settings.max_plot_points).iterrows()],
        "forecast": [{"date": str(date.date()), "value": value} for date, value in zip(future_dates, forecast_values)],
    }
    session.forecast_result = json_safe(result)
    return session.forecast_result


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict:
    session = session_or_404(request.session_id)
    frame = store.frame(request.session_id)
    columns = frame.columns.tolist()
    intent = extract_chart_intent(request.message, columns)
    chart = build_chart_from_intent(frame, intent) if intent else None
    answer = await chat_response(request.message, session.profile, preview_records(frame, settings.max_sample_rows), session.chat_history)
    session.chat_history.extend([{"role": "user", "content": request.message}, {"role": "assistant", "content": answer}])
    return {"answer": answer, "chart": chart}


@app.post("/api/report")
def report(request: ReportRequest) -> Response:
    session = session_or_404(request.session_id)
    report_type = request.report_type.lower()
    if report_type == "eda":
        content = eda_report(session)
    elif report_type == "prediction":
        content = prediction_report(session)
    else:
        if not session.summary:
            session.summary = {"summary": [], "recommendations": []}
        content = business_report(session)
    return Response(content=content, media_type="text/markdown", headers={"Content-Disposition": f"attachment; filename={report_type}-report.md"})


@app.get("/api/{session_id}/download")
def download(session_id: str) -> Response:
    frame = store.frame(session_id)
    return Response(content=dataframe_csv_bytes(frame), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=processed_dataset.csv"})


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
