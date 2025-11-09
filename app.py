import os
import io
import base64
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "rainfall-secret-2025")

# -----------------------------
# Config / constants
# -----------------------------
REQUIRED_NUMERIC = [
    "MinTemp","MaxTemp","Rainfall","Evaporation","Sunshine","WindGustSpeed",
    "Humidity9am","Humidity3pm","Pressure9am","Pressure3pm","Temp9am","Temp3pm"
]
REQUIRED_CATEGORICAL = ["RainToday","Location"]
REQUIRED_COLUMNS = REQUIRED_NUMERIC + REQUIRED_CATEGORICAL
TARGET_COLUMN = "RainTomorrow"

# -----------------------------
# Load models and dataset
# -----------------------------
def ensure_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")

ensure_file("model_lr.pkl")
ensure_file("model_rf.pkl")
ensure_file("weatherAUS.csv")

model_lr = joblib.load("model_lr.pkl")
model_rf = joblib.load("model_rf.pkl")

df_raw = pd.read_csv("weatherAUS.csv")

# Prepare dropdown choices
locations = sorted([x for x in df_raw["Location"].dropna().unique().tolist()])

# Pre-compute simple imputation defaults from the dataset for robust CSV predictions
# Numeric: median; Categorical: most frequent (fallbacks provided)
_num_frame = df_raw.copy()
for _c in REQUIRED_NUMERIC:
    _num_frame[_c] = pd.to_numeric(_num_frame.get(_c), errors="coerce")

NUMERIC_MEDIANS = _num_frame[REQUIRED_NUMERIC].median()
RAIN_TODAY_MODE = (
    df_raw.get("RainToday").astype(str).str.strip().str.title().mode().iat[0]
    if "RainToday" in df_raw.columns and not df_raw["RainToday"].dropna().empty
    else "No"
)
LOCATION_MODE = (
    df_raw.get("Location").astype(str).str.strip().mode().iat[0]
    if "Location" in df_raw.columns and not df_raw["Location"].dropna().empty
    else (locations[0] if locations else "")
)

# -----------------------------
# Utility: plotting to base64
# -----------------------------
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# -----------------------------
# Evaluation (for result page)
# -----------------------------
def normalize_dataframe_cols(df):
    # Ensure numeric columns are numeric and impute; categorical normalize + impute
    for c in REQUIRED_NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Impute numeric medians
    try:
        df[REQUIRED_NUMERIC] = df[REQUIRED_NUMERIC].fillna(NUMERIC_MEDIANS)
    except Exception:
        # Graceful fallback if columns mismatch
        for c in REQUIRED_NUMERIC:
            if c in df.columns:
                df[c] = df[c].fillna(NUMERIC_MEDIANS.get(c, df[c].median()))

    if "RainToday" in df.columns:
        df["RainToday"] = df["RainToday"].astype(str).str.strip().str.title()
        df["RainToday"] = df["RainToday"].replace({"": RAIN_TODAY_MODE, "Nan": RAIN_TODAY_MODE})
        df["RainToday"] = df["RainToday"].fillna(RAIN_TODAY_MODE)
    if "Location" in df.columns:
        df["Location"] = df["Location"].astype(str).str.strip()
        df["Location"] = df["Location"].replace({"": LOCATION_MODE, "Nan": LOCATION_MODE})
        df["Location"] = df["Location"].fillna(LOCATION_MODE)
    return df

def compute_quick_metrics(pipe, X, y):
    preds = pipe.predict(X)
    proba = pipe.predict_proba(X)[:, 1] if hasattr(pipe, "predict_proba") else None
    acc = accuracy_score(y, preds)
    f1  = f1_score(y, preds)
    auc = roc_auc_score(y, proba) if proba is not None else np.nan
    return {"accuracy": acc, "f1": f1, "auc": auc}

def build_holdout_metrics():
    # Build quick metrics on a consistent split for both models
    df = df_raw.copy()
    needed = REQUIRED_COLUMNS + [TARGET_COLUMN]
    df = df.dropna(subset=[c for c in needed if c in df.columns])
    y = df[TARGET_COLUMN].map({"No":0, "Yes":1})
    X = df[REQUIRED_COLUMNS]
    X = normalize_dataframe_cols(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    # Pipelines are already fit; just evaluate on holdout
    m_lr = compute_quick_metrics(model_lr, X_test, y_test)
    m_rf = compute_quick_metrics(model_rf, X_test, y_test)
    # Pick "better" by accuracy, then f1 as tie-break
    better = "Random Forest" if (m_rf["accuracy"], m_rf["f1"]) >= (m_lr["accuracy"], m_lr["f1"]) else "Logistic Regression"
    return m_lr, m_rf, better

metrics_lr, metrics_rf, better_model_name = build_holdout_metrics()

# -----------------------------
# Visualizations (matplotlib)
# -----------------------------
def make_trend_plot_base64():
    df = df_raw.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        tmp = df.dropna(subset=["Date","Rainfall"]).copy()
        tmp["Year"] = tmp["Date"].dt.year
        yearly = tmp.groupby("Year")["Rainfall"].sum().reset_index()
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(yearly["Year"], yearly["Rainfall"], marker="o")
        ax.set_title("Historical Rainfall Trends (Yearly Total)")
        ax.set_xlabel("Year"); ax.set_ylabel("Total Rainfall")
        ax.grid(True, alpha=0.3)
        return fig_to_base64(fig)
    # Fallback empty plot
    fig, ax = plt.subplots(figsize=(8,4))
    ax.text(0.5, 0.5, "No Date column to plot", ha="center", va="center")
    return fig_to_base64(fig)

def get_feature_names_from_pipeline(pipe):
    """Robustly extract feature names from preprocessor whether cat is OHE or a pipeline with OHE."""
    pre = pipe.named_steps["preprocessor"]
    # Numeric original columns list
    try:
        num_cols = pre.named_transformers_["num"].feature_names_in_.tolist()  # if available
    except Exception:
        num_cols = pre.transformers_[0][2]

    # Categorical OHE names
    try:
        # If cat is a pipeline
        ohe = pre.named_transformers_["cat"].named_steps.get("onehot")
        cat_cols = pre.transformers_[1][2]
        if ohe is not None:
            ohe_names = ohe.get_feature_names_out(cat_cols)
        else:
            raise AttributeError
    except Exception:
        # If cat is directly an OHE
        try:
            ohe = pre.transformers_[1][1]
            cat_cols = pre.transformers_[1][2]
            ohe_names = ohe.get_feature_names_out(cat_cols)
        except Exception:
            ohe_names = []
    return list(num_cols) + list(ohe_names)

def make_feature_importance_plot_base64():
    # Use Random Forest importances
    try:
        feature_names = get_feature_names_from_pipeline(model_rf)
        rf = model_rf.named_steps["model"]
        importances = rf.feature_importances_
        # Top 20
        idx = np.argsort(importances)[::-1][:20]
        fig, ax = plt.subplots(figsize=(8,6))
        ax.barh(np.array(feature_names)[idx][::-1], importances[idx][::-1], color="skyblue")
        ax.set_title("Random Forest Feature Importance (Top 20)")
        ax.set_xlabel("Importance")
        return fig_to_base64(fig)
    except Exception:
        fig, ax = plt.subplots(figsize=(8,4))
        ax.text(0.5, 0.5, "Could not compute RF feature importances", ha="center", va="center")
        return fig_to_base64(fig)

def make_confusion_matrix_plot_base64():
    df = df_raw.copy()
    needed = REQUIRED_COLUMNS + [TARGET_COLUMN]
    df = df.dropna(subset=[c for c in needed if c in df.columns])
    y = df[TARGET_COLUMN].map({"No":0, "Yes":1})
    X = normalize_dataframe_cols(df[REQUIRED_COLUMNS])

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    preds = model_rf.predict(X_test)
    cm = confusion_matrix(y_test, preds)

    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Random Forest Confusion Matrix (20% holdout)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha="center", va="center")
    fig.colorbar(im, ax=ax, shrink=0.8)
    return fig_to_base64(fig)

# -----------------------------
# Predict helpers
# -----------------------------
def dict_to_frame(form_dict):
    row = {}
    # Numeric (cast to float)
    for c in REQUIRED_NUMERIC:
        v = form_dict.get(c, "")
        row[c] = float(v) if v not in (None, "",) else np.nan
    # Categorical
    row["RainToday"] = (form_dict.get("RainToday","No") or "No").strip().title()
    row["Location"] = form_dict.get("Location","") or ""
    return pd.DataFrame([row], columns=REQUIRED_COLUMNS)

def predict_single(row_df, which="Both"):
    row_df = normalize_dataframe_cols(row_df.copy())
    out = {}
    if which in ("Both","Logistic Regression"):
        p = model_lr.predict(row_df)[0]
        pr = model_lr.predict_proba(row_df)[0,1]
        out["lr_pred"] = "Yes" if p == 1 else "No"
        out["lr_prob"] = float(pr)
    if which in ("Both","Random Forest"):
        p = model_rf.predict(row_df)[0]
        pr = model_rf.predict_proba(row_df)[0,1]
        out["rf_pred"] = "Yes" if p == 1 else "No"
        out["rf_prob"] = float(pr)
    return out

def predict_batch(df_in, which="Both"):
    df_in = normalize_dataframe_cols(df_in.copy())
    res = pd.DataFrame(index=df_in.index)
    if which in ("Both","Random Forest"):
        rf_pred = model_rf.predict(df_in)
        rf_prob = model_rf.predict_proba(df_in)[:,1]
        res["RF_Pred"] = np.where(rf_pred == 1, "Yes", "No")
        res["RF_Prob"] = rf_prob
    if which in ("Both","Logistic Regression"):
        lr_pred = model_lr.predict(df_in)
        lr_prob = model_lr.predict_proba(df_in)[:,1]
        res["LR_Pred"] = np.where(lr_pred == 1, "Yes", "No")
        res["LR_Prob"] = lr_prob
    return pd.concat([df_in.reset_index(drop=True), res.reset_index(drop=True)], axis=1)

# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        locations=locations,
        metrics_lr=metrics_lr,
        metrics_rf=metrics_rf,
        better_model_name=better_model_name
    )

@app.route("/predict", methods=["POST"])
def predict():
    model_choice = request.form.get("model_choice", "Both")
    file = request.files.get("file")

    if file and file.filename.lower().endswith(".csv"):
        try:
            df = pd.read_csv(file)
            missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
            if missing:
                flash(f"Uploaded CSV is missing required columns: {missing}", "danger")
                return redirect(url_for("index"))
            out = predict_batch(df[REQUIRED_COLUMNS], which=model_choice)
            # Save for download
            csv_buf = io.StringIO()
            out.to_csv(csv_buf, index=False)
            session["last_csv"] = csv_buf.getvalue()
            # Summary
            summary = {}
            if "RF_Pred" in out.columns:
                summary["rf_yes"] = int((out["RF_Pred"] == "Yes").sum())
                summary["rf_no"] = int((out["RF_Pred"] == "No").sum())
            if "LR_Pred" in out.columns:
                summary["lr_yes"] = int((out["LR_Pred"] == "Yes").sum())
                summary["lr_no"] = int((out["LR_Pred"] == "No").sum())
            return render_template(
                "result.html",
                is_batch=True,
                out_preview=out.head(20).to_html(index=False, classes="table table-striped table-sm"),
                summary=summary,
                model_choice=model_choice,
                metrics_lr=metrics_lr,
                metrics_rf=metrics_rf,
                better_model_name=better_model_name
            )
        except Exception as e:
            flash(f"CSV processing error: {e}", "danger")
            return redirect(url_for("index"))

    # Manual form
    row_df = dict_to_frame(request.form)
    if row_df[REQUIRED_COLUMNS].isna().any().any():
        flash("Please fill all fields for manual prediction.", "danger")
        return redirect(url_for("index"))

    out = predict_single(row_df, which=model_choice)

    # Simple bar for probabilities (matplotlib)
    labels, probs = [], []
    if "rf_prob" in out:
        labels.append("Random Forest"); probs.append(out["rf_prob"])
    if "lr_prob" in out:
        labels.append("Logistic Regression"); probs.append(out["lr_prob"])
    if probs:
        fig, ax = plt.subplots(figsize=(4,3))
        ax.bar(labels, probs, color=["#3b82f6","#10b981"][:len(labels)])
        ax.set_ylim(0,1.0); ax.set_ylabel("Probability of Rain (Yes)")
        prob_png = fig_to_base64(fig)
    else:
        prob_png = None

    return render_template(
        "result.html",
        is_batch=False,
        single_out=out,
        prob_png=prob_png,
        model_choice=model_choice,
        metrics_lr=metrics_lr,
        metrics_rf=metrics_rf,
        better_model_name=better_model_name
    )

@app.route("/download", methods=["GET"])
def download():
    csv_text = session.get("last_csv")
    if not csv_text:
        flash("No predictions to download. Upload a CSV and predict first.", "warning")
        return redirect(url_for("index"))
    return send_file(
        io.BytesIO(csv_text.encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="predictions.csv"
    )

@app.route("/visualize", methods=["GET"])
def visualize():
    trend_png = make_trend_plot_base64()
    feat_png = make_feature_importance_plot_base64()
    cm_png = make_confusion_matrix_plot_base64()
    return render_template(
        "visualize.html",
        trend_png=trend_png,
        feat_png=feat_png,
        cm_png=cm_png
    )

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)