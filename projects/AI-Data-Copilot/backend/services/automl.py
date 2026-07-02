from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import numpy as np

from backend.config import settings
from backend.utils.helpers import choose_default_target, json_safe


def run_automl(frame: pd.DataFrame, target: str | None = None) -> dict:
    target = target or choose_default_target(frame)
    if not target or target not in frame.columns:
        raise ValueError("A valid target column is required for AutoML.")
    data = frame.dropna(subset=[target]).copy()
    if len(data) < 10:
        raise ValueError("AutoML needs at least 10 rows with a target value.")
    y = data[target]
    x = data.drop(columns=[target])
    x = x.drop(columns=[col for col in x.columns if x[col].nunique(dropna=True) <= 1], errors="ignore")
    if x.empty:
        raise ValueError("No usable feature columns were found.")
    numeric_features = x.select_dtypes(include="number").columns.tolist()
    categorical_features = [col for col in x.columns if col not in numeric_features]
    task = "regression" if pd.api.types.is_numeric_dtype(y) and y.nunique() > 15 else "classification"
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
        ],
        remainder="drop",
    )
    model = (
        RandomForestRegressor(n_estimators=120, random_state=settings.random_state)
        if task == "regression"
        else RandomForestClassifier(n_estimators=120, random_state=settings.random_state, class_weight="balanced")
    )
    pipeline = Pipeline([("prep", preprocessor), ("model", model)])
    stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=settings.random_state, stratify=stratify)
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    if task == "regression":
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        metrics = {"rmse": rmse, "r2": float(r2_score(y_test, predictions))}
    else:
        metrics = {"accuracy": float(accuracy_score(y_test, predictions)), "f1": float(f1_score(y_test, predictions, average="weighted"))}
    feature_names = numeric_features + list(
        pipeline.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(categorical_features)
        if categorical_features
        else []
    )
    importances = pipeline.named_steps["model"].feature_importances_
    feature_importance = sorted(
        [{"feature": name, "importance": float(value)} for name, value in zip(feature_names, importances)],
        key=lambda item: item["importance"],
        reverse=True,
    )[:15]
    return json_safe({"target": target, "task": task, "metrics": metrics, "feature_importance": feature_importance, "rows_used": len(data)})
