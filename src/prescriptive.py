"""
Layer 4: PRESCRIPTIVE ENGINE (the unique twist) -- causal uplift version
For each at-risk customer, scores every action's Individual Treatment Effect (ITE) using
the T-learner uplift models trained in uplift.py (learned from a randomized historical
action log, not a heuristic guess), converts ITE -> expected rupee save -> ROI, and ranks
actions globally by total impact.
"""
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path

from uplift import score_ite, train_uplift_models  # noqa: F401 (train_uplift_models used by callers)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

ACTION_COSTS = {
    "discount_10pct": 150.0,
    "free_upgrade_3mo": 400.0,
    "account_manager_call": 250.0,
    "loyalty_bonus": 80.0,
}
ACTION_DISPLAY_NAMES = {
    "discount_10pct": "10% discount offer",
    "free_upgrade_3mo": "Free upgrade (3 months)",
    "account_manager_call": "Personal account manager call",
    "loyalty_bonus": "Loyalty points bonus",
}


def _load_artifacts():
    churn_art = joblib.load(MODEL_DIR / "churn_model.joblib")
    uplift_art = joblib.load(MODEL_DIR / "uplift_models.joblib")
    feats = pd.read_parquet(DATA_DIR / "feature_store.parquet")
    customers = pd.read_parquet(DATA_DIR / "customers.parquet")
    return churn_art, uplift_art, feats, customers


def score_and_recommend(top_n_customers=200, ltv_horizon_months=12):
    churn_art, uplift_art, feats, customers = _load_artifacts()
    churn_model, feature_cols = churn_art["model"], churn_art["feature_cols"]
    control_model, treated_models = uplift_art["control_model"], uplift_art["treated_models"]
    uplift_feature_cols = uplift_art["feature_cols"]

    latest = feats.sort_values("month_idx").groupby("customer_id").tail(1).copy()
    X_latest = latest[feature_cols].fillna(0)
    latest["risk_before"] = churn_model.predict_proba(X_latest.values)[:, 1]

    latest = latest.merge(customers[["customer_id", "ltv_multiplier", "base_monthly_price"]],
                           on="customer_id", suffixes=("", "_cust"))
    latest["ltv"] = latest["base_monthly_price"] * ltv_horizon_months * latest["ltv_multiplier"]

    at_risk = latest.sort_values("risk_before", ascending=False).head(top_n_customers).copy()

    results = []
    for _, row in at_risk.iterrows():
        ites, p_control = score_ite(row, uplift_feature_cols, control_model, treated_models)

        seg_matches = [c.replace("seg_", "") for c in feature_cols
                       if c.startswith("seg_") and row.get(c, 0) == 1]
        segment = seg_matches[0] if seg_matches else "unknown"

        for action, ite in ites.items():
            expected_save = ite * row["ltv"]
            cost = ACTION_COSTS[action]
            roi = (expected_save - cost) / cost
            results.append({
                "customer_id": row["customer_id"],
                "segment": segment,
                "action": ACTION_DISPLAY_NAMES[action],
                "risk_before": round(float(row["risk_before"]), 4),
                "ite_risk_reduction": round(float(ite), 4),
                "risk_after": round(max(float(row["risk_before"]) - float(ite), 0.0), 4),
                "ltv": round(float(row["ltv"]), 2),
                "expected_save_inr": round(float(expected_save), 2),
                "cost_inr": cost,
                "roi": round(float(roi), 2),
            })
        # "Do nothing" baseline row for comparison
        results.append({
            "customer_id": row["customer_id"], "segment": segment, "action": "Do nothing",
            "risk_before": round(float(row["risk_before"]), 4), "ite_risk_reduction": 0.0,
            "risk_after": round(float(row["risk_before"]), 4), "ltv": round(float(row["ltv"]), 2),
            "expected_save_inr": 0.0, "cost_inr": 0.0, "roi": 0.0,
        })

    results_df = pd.DataFrame(results)
    results_df.to_parquet(DATA_DIR / "prescriptive_results.parquet", index=False)

    actionable = results_df[results_df["action"] != "Do nothing"]
    best_per_customer = actionable.loc[actionable.groupby("customer_id")["roi"].idxmax()]
    global_rank = (best_per_customer.groupby("action")
                   .agg(customers_targeted=("customer_id", "count"),
                        total_expected_save_inr=("expected_save_inr", "sum"),
                        total_cost_inr=("cost_inr", "sum"),
                        avg_roi=("roi", "mean"))
                   .sort_values("total_expected_save_inr", ascending=False)
                   .reset_index())
    global_rank.to_parquet(DATA_DIR / "action_ranking.parquet", index=False)

    return results_df, global_rank, best_per_customer


if __name__ == "__main__":
    # ensure uplift models exist / are fresh
    with open(MODEL_DIR / "metrics.json") as f:
        metrics = json.load(f)
    train_uplift_models(metrics["churn_features"])

    results_df, global_rank, best_per_customer = score_and_recommend()
    print("Top recommended actions by total rupee impact (causal uplift, not heuristic):")
    print(global_rank.to_string(index=False))
    print(f"\nTotal customers evaluated: {results_df.customer_id.nunique()}")
    print(f"Total expected save if best-action-per-customer adopted: "
          f"₹{best_per_customer.expected_save_inr.sum():,.0f}")
