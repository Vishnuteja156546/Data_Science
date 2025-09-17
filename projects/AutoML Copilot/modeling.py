import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

def quick_prep(df, target):
    df2 = df.copy()
    for col in df2.select_dtypes(include=[np.number]).columns:
        df2[col] = df2[col].fillna(df2[col].median())
    for col in df2.select_dtypes(exclude=[np.number]).columns:
        df2[col] = df2[col].fillna(df2[col].mode().iloc[0] if not df2[col].mode().empty else "")
    return df2

def is_classification(df, target):
    return df[target].dtype == 'object' or df[target].nunique() <= 20 and df[target].dtype != 'float'

def simple_automl(df, target, mode="fast"):
    df2 = quick_prep(df, target)
    X = df2.drop(columns=[target])
    y = df2[target]

    for col in X.select_dtypes(include=['object','category']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    if y.dtype == 'object' or y.dtype.name == 'category':
        y = LabelEncoder().fit_transform(y.astype(str))
        problem = "classification"
    elif y.nunique() <= 20 and not np.issubdtype(y.dtype, np.floating):
        y = LabelEncoder().fit_transform(y)
        problem = "classification"
    else:
        problem = "regression"

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if problem == "classification":
        model = RandomForestClassifier(n_estimators=100 if mode=='fast' else 300, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        return {"problem": problem, "accuracy": acc, "f1_weighted": f1, "model": model}
    else:
        model = RandomForestRegressor(n_estimators=100 if mode=='fast' else 300, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        return {"problem": problem, "rmse": rmse, "model": model}
