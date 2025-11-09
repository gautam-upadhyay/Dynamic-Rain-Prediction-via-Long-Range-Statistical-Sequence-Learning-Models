# generate_models.py
# Builds and saves model_lr.pkl and model_rf.pkl from weatherAUS.csv

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import sys
from pathlib import Path

def main():
    csv_path = Path("weatherAUS.csv")
    if not csv_path.exists():
        print("ERROR: weatherAUS.csv not found in the current directory.")
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Features matching the intended UI
    num_features = [
        'MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine','WindGustSpeed',
        'Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Temp9am','Temp3pm'
    ]
    cat_features = ['RainToday','Location']
    all_features = num_features + cat_features
    target = 'RainTomorrow'

    # Ensure numeric for numeric columns
    for c in num_features:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Keep only necessary columns + target, drop rows with missing
    keep_cols = all_features + [target]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing expected columns in CSV: {missing}")
        sys.exit(1)

    df = df.dropna(subset=keep_cols)

    # Map target to 0/1
    y = df[target].map({'No': 0, 'Yes': 1})
    mask = y.isna()
    if mask.any():
        # Drop rows with unexpected target values
        df = df.loc[~mask]
        y = y.loc[~mask]

    X = df[all_features]

    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
        ]
    )

    # Models
    lr = LogisticRegression(solver='liblinear', penalty='l1', random_state=42)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    )

    lr_pipe = Pipeline([('preprocessor', preprocessor), ('model', lr)])
    rf_pipe = Pipeline([('preprocessor', preprocessor), ('model', rf)])

    # Quick holdout metrics for reference
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    lr_pipe.fit(X_train, y_train)
    rf_pipe.fit(X_train, y_train)

    def quick_eval(name, pipe):
        preds = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, 'predict_proba') else None
        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, proba) if proba is not None else np.nan
        print(f"{name:16s} | accuracy={acc:.3f}  f1={f1:.3f}  auc={auc:.3f}")

    print("Holdout metrics (20% test):")
    quick_eval("LogisticRegression", lr_pipe)
    quick_eval("RandomForest",      rf_pipe)

    # Retrain on all data for final artifacts
    lr_pipe.fit(X, y)
    rf_pipe.fit(X, y)

    joblib.dump(lr_pipe, "model_lr.pkl")
    joblib.dump(rf_pipe, "model_rf.pkl")
    print("Saved: model_lr.pkl, model_rf.pkl")

if __name__ == "__main__":
    main()