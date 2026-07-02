from io import BytesIO
from pathlib import Path
import pandas as pd


ALLOWED_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls"}


def validate_dataset_filename(filename: str) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError("Only CSV, TSV, XLSX, and XLS files are supported.")


def read_dataset_bytes(filename: str, content: bytes, max_rows: int) -> pd.DataFrame:
    validate_dataset_filename(filename)
    suffix = Path(filename).suffix.lower()
    try:
        if suffix in {".xlsx", ".xls"}:
            frame = pd.read_excel(BytesIO(content))
        else:
            sep = "\t" if suffix == ".tsv" else ","
            frame = pd.read_csv(BytesIO(content), sep=sep)
    except Exception as exc:
        raise ValueError(f"Could not read dataset: {exc}") from exc
    if frame.empty:
        raise ValueError("The uploaded dataset is empty.")
    if len(frame) > max_rows:
        frame = frame.head(max_rows).copy()
    frame.columns = [str(col).strip() or f"column_{i}" for i, col in enumerate(frame.columns)]
    return frame
