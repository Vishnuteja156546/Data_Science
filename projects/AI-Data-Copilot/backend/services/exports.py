from __future__ import annotations

from io import StringIO
import pandas as pd


def dataframe_csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = StringIO()
    frame.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")
