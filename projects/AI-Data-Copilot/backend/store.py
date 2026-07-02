from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
import pandas as pd


@dataclass
class DatasetSession:
    session_id: str
    filename: str
    dataframe: pd.DataFrame
    cleaned_dataframe: pd.DataFrame | None = None
    profile: dict = field(default_factory=dict)
    charts: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    model_result: dict = field(default_factory=dict)
    forecast_result: dict = field(default_factory=dict)
    chat_history: list[dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, DatasetSession] = {}

    def create(self, filename: str, dataframe: pd.DataFrame) -> DatasetSession:
        session_id = str(uuid4())
        session = DatasetSession(session_id=session_id, filename=filename, dataframe=dataframe)
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> DatasetSession:
        if session_id not in self._sessions:
            raise KeyError(f"Session not found: {session_id}")
        return self._sessions[session_id]

    def frame(self, session_id: str, cleaned: bool = True) -> pd.DataFrame:
        session = self.get(session_id)
        if cleaned and session.cleaned_dataframe is not None:
            return session.cleaned_dataframe
        return session.dataframe


store = SessionStore()
