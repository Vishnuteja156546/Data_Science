from typing import Any
from pydantic import BaseModel, Field


class AutoMLRequest(BaseModel):
    session_id: str
    target: str | None = None


class ForecastRequest(BaseModel):
    session_id: str
    date_column: str | None = None
    value_column: str | None = None
    periods: int = Field(default=12, ge=1, le=60)


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1, max_length=2000)


class ReportRequest(BaseModel):
    session_id: str
    report_type: str = "business"


class ChartRequest(BaseModel):
    session_id: str
    chart_type: str
    x: str | None = None
    y: str | None = None
    group_by: str | None = None


JsonDict = dict[str, Any]
