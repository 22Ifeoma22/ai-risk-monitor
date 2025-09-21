import streamlit as st
import pandas as pd
import joblib
import os
from pathlib import Path
import numpy as np
import datetime as dt

st.set_page_config(page_title="AI Risk Monitor", page_icon="🛡️", layout="wide")
st.title("🛡️ AI Risk Monitor — Continuous Model Assurance")

# ---------- load model ----------
bundle = joblib.load("artifacts/model.joblib")
model = bundle["model"]; cfg = bundle["cfg"]
FEATURES = cfg["num_cols"] + cfg["cat_cols"]

# ---------- drift helpers ----------
TRAIN_FEATS = Path("artifacts/train_features.parquet")
LIVE_FEATS  = Path("artifacts/live_window.parquet")

def psi(ref, cur, bins=10):
    bins_ = np.histogram_bin_edges(ref[~np.isnan(ref)], bins=bins)
    ar,_ = np.histogram(ref, bins_); br,_ = np.histogram(cur, bins_)
    ar = np.clip(ar/ar.sum(), 1e-6, 1); br = np.clip(br/br.sum(), 1e-6, 1)
    return float(((br-ar)*np.log(br/ar)).sum())

def append_live_row(row_df: pd.DataFrame):
    LIVE_FEATS.parent.mkdir(exist_ok=True, parents=True)
    if LIVE_FEATS.exists():
        cur = pd.read_parquet(LIVE_FEATS)
        pd.concat([cur, row_df], ignore_index=True).to_parquet(LIVE_FEATS, index=False)
    else:
        row_df.to_parquet(LIVE_FEATS, index=False)

def run_drift_check(num_cols, threshold=0.25):
    if not TRAIN_FEATS.exists() or not LIVE_FEATS.exists():
        return [], pd.DataFrame()
    ref = pd.read_parquet(TRAIN_FEATS)
    cur = pd.read_parquet(LIVE_FEATS)
    recs, alerts = [], []
    for col in num_cols:
        s = psi(ref[col].astype(float), cur[col].astype(float))
        status = "OK" if s <= threshold else "ALERT"
        recs.append({"feature": col, "psi": round(s,3), "status": status})
        if status == "ALERT":
            alerts.append(f"{col}: PSI={s:.3f}")
    table = pd.DataFrame(recs)

    if alerts:
        rr_path = Path("risk_register.csv")
        row = {
            "timestamp": dt.datetime.utcnow().isoformat(),
            "risk_id": "DRIFT",
            "description": "; ".join(alerts),
            "severity": "Medium",
            "owner": "Model Risk",
            "status": "Open",
        }
        if rr_path.exists():
            prev = pd.read_csv(rr_path)
            pd.concat([prev, pd.DataFrame([row])], ignore_index=True).to_csv(rr_path, index=False)
        else:
            pd.DataFrame([row]).to_csv(rr_path, index=False)
    return alerts, table

# ---------- UI ----------
tabs = st.tabs(["Predict", "Risk Register & Drift", "Compliance Mapping"])

with tabs[0]:
    st.subheader(" Predict Risk (Demo)")
    c = st.columns(3)
    num1 = c[0].number_input("num1", value=50.0)
    num2 = c[1].number_input("num2", value=40.0)
    num3 = c[2].number_input("num3", value=0.5, min_value=0.0, max_value=1.0, step=0.01)
    c2 = st.columns(2)
    cat1 = c2[0].selectbox("cat1", ["A","B","C"])
    cat2 = c2[1].selectbox("cat2", ["X","Y"])

    if st.button("Predict"):
        row = pd.DataFrame([{"num1":num1,"num2":num2,"num3":num3,"cat1":cat1,"cat2":cat2}])[FEATURES]
        append_live_row(row[[c for c in cfg["num_cols"]]])   # <-- capture for drift
        proba = model.predict_proba(row)[:,1] if hasattr(model,"predict_proba") else None
        pred = int(model.predict(row)[0])
        st.metric("Prediction", " risk" if pred==1 else " no_risk")
        if proba is not None:
            st.metric("Probability of risk", f"{proba[0]:.2%}")

    st.markdown("---")
    up = st.file_uploader("Upload CSV with columns: " + ", ".join(FEATURES), type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            if hasattr(model,"predict_proba"):
                df["prob_risk"] = model.predict_proba(df[FEATURES])[:,1]
            df["prediction"] = model.predict(df[FEATURES])
            # capture numeric features from uploaded rows too
            append_live_row(df[[c for c in cfg["num_cols"]] ])
            st.dataframe(df.head(50), use_container_width=True)
            st.download_button("Download predictions", df.to_csv(index=False).encode("utf-8"), "predictions.csv")

with tabs[1]:
    st.subheader(" Risk Register & Drift")
    alerts, table = run_drift_check(cfg["num_cols"], threshold=0.25)
    rag = "🟢 OK"
    if not table.empty and (table["status"] == "ALERT").any():
        rag = "🔴 DRIFT"
    st.metric("Drift status", rag)
    if not table.empty:
        st.dataframe(table, use_container_width=True)

    rr_path = "risk_register.csv"
    if os.path.exists(rr_path):
        rr = pd.read_csv(rr_path)
        st.dataframe(rr, use_container_width=True)
        st.download_button("Download risk register", rr.to_csv(index=False).encode("utf-8"), "risk_register.csv")
    else:
        st.info("No risk entries yet. Generate predictions and re-check.")

with tabs[2]:
    st.subheader("📑 Compliance Mapping (Read-only)")
    st.markdown("""
**ISO/IEC 42001**
- AIMS 6–8: Risk assessment & controls → *Drift monitoring, performance thresholds*
- AIMS 9–10: Monitoring & improvement → *Scheduled checks, alerting & review*

**NIST AI RMF 100-1**
- Map → *Context, limitations, stakeholders*
- Measure → *F1/AUC, drift (PSI), fairness tests*
- Manage → *Risk register, owners, SLAs*

**GDPR Article 22**
- Human oversight & meaningful information for automated decisions
""")
