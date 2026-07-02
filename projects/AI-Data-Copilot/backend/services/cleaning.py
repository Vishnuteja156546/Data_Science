from __future__ import annotations

import pandas as pd


def clean_dataframe(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cleaned = frame.copy()
    log: list[str] = []
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    removed = before - len(cleaned)
    if removed:
        log.append(f"Removed {removed} duplicate rows.")
    for column in cleaned.columns:
        missing = int(cleaned[column].isna().sum())
        if not missing:
            continue
        if pd.api.types.is_numeric_dtype(cleaned[column]):
            fill_value = cleaned[column].median()
            cleaned[column] = cleaned[column].fillna(fill_value)
            log.append(f"Filled {missing} missing values in {column} with the median.")
        else:
            mode = cleaned[column].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            cleaned[column] = cleaned[column].fillna(fill_value)
            log.append(f"Filled {missing} missing values in {column} with the mode.")
    for column in cleaned.select_dtypes(include="number").columns:
        q1 = cleaned[column].quantile(0.25)
        q3 = cleaned[column].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        before_outliers = int(((cleaned[column] < lower) | (cleaned[column] > upper)).sum())
        if before_outliers:
            cleaned[column] = cleaned[column].clip(lower, upper)
            log.append(f"Capped {before_outliers} outliers in {column}.")
    if not log:
        log.append("No cleaning changes were required.")
    return cleaned, log
