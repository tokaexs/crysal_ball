"""
Layer 6: SERVING LAYER (FastAPI)
  POST /predict_churn     -> risk score + top-3 SHAP drivers
  POST /predict_revenue   -> point forecast + [low, high] interval
  POST /recommend_actions -> ranked action plan with rupee impact
  GET  /drift_status      -> PSI, KS, alert level
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llm import explain_customer_risk, draft_retention_message, ask_copilot, competitive_briefing
from competitive import full_report, load_benchmark

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

app = FastAPI(title="Crystal Ball API", description="Churn + Revenue prediction with prescriptive actions")

_cache = {}


def _get_artifacts():
    if not _cache:
        _cache["churn"] = joblib.load(MODEL_DIR / "churn_model.joblib")
        _cache["revenue"] = joblib.load(MODEL_DIR / "revenue_model.joblib")
        _cache["feats"] = pd.read_parquet(DATA_DIR / "feature_store.parquet")
        _cache["customers"] = pd.read_parquet(DATA_DIR / "customers.parquet")
        try:
            _cache["prescriptive"] = pd.read_parquet(DATA_DIR / "prescriptive_results.parquet")
        except FileNotFoundError:
            _cache["prescriptive"] = pd.DataFrame()
    return _cache


class CustomerRequest(BaseModel):
    customer_id: str


class DraftMessageRequest(BaseModel):
    customer_id: str
    action: str
    tone: str = "warm and professional"


class CopilotRequest(BaseModel):
    question: str


class CompetitiveRequest(BaseModel):
    company_key: str  # e.g. "netflix", "jiohotstar", "amazon_prime_video", "youtube_premium"
    tier: str = "premium"


def _latest_row(customer_id: str):
    feats = _get_artifacts()["feats"]
    rows = feats[feats.customer_id == customer_id].sort_values("month_idx")
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"customer_id {customer_id} not found")
    return rows.tail(1).iloc[0]


@app.get("/")
def root():
    return {"service": "Crystal Ball", "status": "ok",
            "endpoints": ["/predict_churn", "/predict_revenue", "/recommend_actions", "/drift_status"]}


@app.post("/predict_churn")
def predict_churn(req: CustomerRequest):
    art = _get_artifacts()
    model, feature_cols = art["churn"]["model"], art["churn"]["feature_cols"]
    row = _latest_row(req.customer_id)
    X = row[feature_cols].fillna(0).values.reshape(1, -1).astype(float)
    risk = float(model.predict_proba(X)[0, 1])

    # SHAP explainability - explain the underlying calibrated estimator's base learner
    try:
        base_est = model.calibrated_classifiers_[0].estimator
        explainer = shap.TreeExplainer(base_est)
        shap_vals = explainer.shap_values(X)
        sv = shap_vals[0] if isinstance(shap_vals, list) else shap_vals[0]
        top_idx = np.argsort(-np.abs(sv))[:3]
        top_drivers = [{"feature": feature_cols[i], "shap_value": round(float(sv[i]), 4)} for i in top_idx]
    except Exception:
        top_drivers = []

    return {"customer_id": req.customer_id, "churn_risk": round(risk, 4), "top_3_drivers": top_drivers}


@app.post("/predict_revenue")
def predict_revenue(req: CustomerRequest):
    art = _get_artifacts()
    rev = art["revenue"]
    row = _latest_row(req.customer_id)
    feature_cols = rev["feature_cols"]
    X = row[feature_cols].fillna(0).values.reshape(1, -1).astype(float)
    point = float(rev["point_model"].predict(X)[0])
    lo = float(rev["lo_model"].predict(X)[0])
    hi = float(rev["hi_model"].predict(X)[0])
    return {"customer_id": req.customer_id, "point_forecast_inr": round(point, 2),
            "interval_90": [round(min(lo, hi), 2), round(max(lo, hi), 2)]}


@app.post("/recommend_actions")
def recommend_actions(req: CustomerRequest):
    presc = _get_artifacts()["prescriptive"]
    if presc.empty:
        raise HTTPException(status_code=503, detail="Prescriptive results not yet computed. Run src/prescriptive.py")
    rows = presc[presc.customer_id == req.customer_id]
    if rows.empty:
        raise HTTPException(status_code=404,
                             detail="Customer not in current at-risk scoring batch (top-N by risk)")
    rows = rows.sort_values("roi", ascending=False).drop(columns=["customer_id"])
    # ensure plain python types (numpy scalars aren't JSON serializable)
    records = [{k: (v.item() if hasattr(v, "item") else v) for k, v in rec.items()}
               for rec in rows.to_dict("records")]
    return {"customer_id": req.customer_id, "ranked_actions": records}


@app.post("/explain_customer")
def explain_customer(req: CustomerRequest):
    """Natural-language version of /predict_churn + /recommend_actions, for a human to read
    in 10 seconds instead of parsing JSON."""
    churn_resp = predict_churn(req)
    presc = _get_artifacts()["prescriptive"]
    rec_action, rec_save = None, None
    if not presc.empty:
        rows = presc[(presc.customer_id == req.customer_id) & (presc.action != "Do nothing")]
        if not rows.empty:
            top = rows.sort_values("roi", ascending=False).iloc[0]
            rec_action, rec_save = top["action"], float(top["expected_save_inr"])

    narrative = explain_customer_risk(req.customer_id, churn_resp["churn_risk"],
                                       churn_resp["top_3_drivers"], rec_action, rec_save)
    return {"customer_id": req.customer_id, "churn_risk": churn_resp["churn_risk"],
            "recommended_action": rec_action, "narrative": narrative}


@app.post("/draft_retention_message")
def draft_message(req: DraftMessageRequest):
    art = _get_artifacts()
    feats = art["feats"]
    rows = feats[feats.customer_id == req.customer_id]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"customer_id {req.customer_id} not found")
    seg_cols = [c for c in feats.columns if c.startswith("seg_")]
    row = rows.iloc[-1]
    seg_matches = [c.replace("seg_", "") for c in seg_cols if row.get(c, 0) == 1]
    segment = seg_matches[0] if seg_matches else "unknown"

    draft = draft_retention_message(req.customer_id, segment, req.action, req.tone)
    return {"customer_id": req.customer_id, "segment": segment, "action": req.action, "draft": draft}


@app.post("/ask_copilot")
def copilot(req: CopilotRequest):
    """Freeform Q&A grounded in current aggregates (trust card + action ranking + top
    at-risk summary) -- not a raw dump of every row, to keep the context small and relevant."""
    art = _get_artifacts()
    trust_path = MODEL_DIR / "trust_card.json"
    trust = json.load(open(trust_path)) if trust_path.exists() else {}
    try:
        ranking = pd.read_parquet(DATA_DIR / "action_ranking.parquet").to_dict("records")
    except FileNotFoundError:
        ranking = []
    feats = art["feats"]
    latest = feats.sort_values("month_idx").groupby("customer_id").tail(1)
    context = {
        "trust_card": trust,
        "action_ranking": ranking,
        "n_customers": int(latest.customer_id.nunique()),
        "avg_revenue_last_month": float(latest["revenue"].mean()),
    }
    answer = ask_copilot(req.question, context)
    return {"question": req.question, "answer": answer}


@app.get("/competitive_companies")
def competitive_companies():
    """Lists which companies are available in the competitive benchmark dataset."""
    data = load_benchmark()
    return {"companies": {k: v["display_name"] for k, v in data["companies"].items()},
            "meta": data["_meta"]}


@app.post("/competitive_benchmark")
def competitive_benchmark(req: CompetitiveRequest):
    try:
        report = full_report(req.company_key, req.tier)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    narrative = competitive_briefing(
        report["gap_analysis"]["company"], report["gap_analysis"], report["pricing_position"],
        report["retention_cross_reference"]["ranked_by_priority"]
    )
    report["narrative"] = narrative
    return report


@app.get("/drift_status")
def drift_status():
    trust_path = MODEL_DIR / "trust_card.json"
    if not trust_path.exists():
        raise HTTPException(status_code=503, detail="Trust card not yet computed. Run src/trust.py")
    with open(trust_path) as f:
        card = json.load(f)
    return card["drift"]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)