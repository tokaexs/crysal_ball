"""
Layer 9: COMPETITIVE INTELLIGENCE MODULE
Answers: "what are competitors doing better, and what does that cost us?"

Structured, cited comparison framework over real public data (pricing pages, earnings reports,
press releases) + VADER sentiment analysis over customer feedback + cross-reference with
our own trained LightGBM churn model feature importances.
"""
from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np

from sentiment import analyze_sentiment

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
BENCHMARK_PATH = DATA_DIR / "competitive_benchmark.json"

ALL_FEATURE_FLAGS = [
    "live_sports", "ad_tier_available", "offline_downloads", "bundled_with_other_services",
]

# Bridge connecting churn model features to competitive aspects
FEATURE_TO_ASPECT_MAP = {
    "revenue": "pricing",
    "revenue_lag_1": "pricing",
    "revenue_lag_2": "pricing",
    "revenue_lag_3": "pricing",
    "contract_month-to-month": "pricing",
    "contract_one-year": "pricing",
    "contract_two-year": "pricing",
    "tickets_opened": "support_quality",
    "ticket_velocity": "support_quality",
    "tenure_months": "engagement",
    "active_campaigns_past_6m": "engagement",
}


def load_benchmark():
    with open(BENCHMARK_PATH) as f:
        return json.load(f)


def pricing_position(data: dict, tier: str = "premium") -> list[dict]:
    rows = []
    for key, company in data["companies"].items():
        monthly = None
        comparability_note = None
        if "india_pricing_inr_per_month" in company:
            monthly = company["india_pricing_inr_per_month"].get(tier)
        elif "india_pricing_inr" in company:
            tier_data = company["india_pricing_inr"].get(tier)
            if isinstance(tier_data, dict):
                monthly = tier_data.get("monthly")
            elif tier == "premium" and "prime_membership_annual" in company["india_pricing_inr"]:
                monthly = round(company["india_pricing_inr"]["prime_membership_annual"] / 12, 0)
                comparability_note = "Derived from annual bundle price (video+music+shipping), not a video-only monthly price"
        rows.append({
            "company": company["display_name"],
            "approx_monthly_inr": monthly,
            "comparability_note": comparability_note,
        })
    rows = [r for r in rows if r["approx_monthly_inr"] is not None]
    return sorted(rows, key=lambda r: r["approx_monthly_inr"])


def feature_gap_matrix(data: dict) -> dict:
    matrix = {}
    for flag in ALL_FEATURE_FLAGS:
        matrix[flag] = {
            company["display_name"]: company["feature_flags"].get(flag)
            for company in data["companies"].values()
        }
    return matrix


def what_is_company_missing(data: dict, company_key: str) -> dict:
    if company_key not in data["companies"]:
        raise KeyError(f"Unknown company '{company_key}'. Options: {list(data['companies'])}")
    target = data["companies"][company_key]
    others = {k: v for k, v in data["companies"].items() if k != company_key}

    missing, advantages = [], []
    for flag in ALL_FEATURE_FLAGS:
        target_has = bool(target["feature_flags"].get(flag))
        competitors_with_it = [c["display_name"] for c in others.values() if c["feature_flags"].get(flag)]
        if not target_has and competitors_with_it:
            missing.append({"feature": flag, "competitors_with_it": competitors_with_it})
        elif target_has and not competitors_with_it:
            advantages.append({"feature": flag, "unique_to": target["display_name"]})

    return {
        "company": target["display_name"],
        "missing_vs_competitors": missing,
        "unique_advantages": advantages,
    }


def retention_relevance_ranking(company_key: str, sentiment_data: dict) -> dict:
    """Combines real LightGBM feature importances with competitor sentiment intensity."""
    churn_model_path = MODEL_DIR / "churn_model.joblib"
    aspect_churn_weights = {}

    if churn_model_path.exists():
        model_obj = joblib.load(churn_model_path)
        calibrated = model_obj.get("model", model_obj)
        feature_names = model_obj.get("feature_cols", [])
        if hasattr(calibrated, "calibrated_classifiers_") and calibrated.calibrated_classifiers_:
            base_est = calibrated.calibrated_classifiers_[0].estimator
            importances = getattr(base_est, "feature_importances_", [])
            if len(importances) == len(feature_names):
                total_imp = max(1, sum(importances))
                normalized_imp = {f: imp / total_imp for f, imp in zip(feature_names, importances)}
                for feat, aspect in FEATURE_TO_ASPECT_MAP.items():
                    if feat in normalized_imp:
                        aspect_churn_weights[aspect] = aspect_churn_weights.get(aspect, 0.0) + normalized_imp[feat]

    company_sentiment = sentiment_data.get(company_key, {}).get("aspects", {})
    all_aspects = set(list(company_sentiment.keys()) + list(aspect_churn_weights.keys()))

    ranked = []
    for aspect in all_aspects:
        churn_weight = round(aspect_churn_weights.get(aspect, 0.0), 3)
        sentiment_info = company_sentiment.get(aspect, {})
        compound = sentiment_info.get("compound", 0.0)
        # Priority score: competitor sentiment intensity * our churn importance
        priority_score = round(abs(compound) * churn_weight, 4)

        ranked.append({
            "aspect": aspect,
            "our_churn_model_weight": churn_weight,
            "sentiment_compound": compound,
            "priority_score": priority_score,
            "interpretation": (
                "High model signal & sentiment focus" if priority_score > 0.1 else
                ("Driven by churn model weight" if churn_weight > 0.05 else "Unmodeled aspect in synthetic churn data")
            )
        })

    ranked.sort(key=lambda x: x["priority_score"], reverse=True)
    return {
        "ranked_by_priority": ranked,
        "methodology": "priority_score = |sentiment_compound| * our_churn_model_weight",
    }


def full_report(company_key: str, tier: str = "premium") -> dict:
    data = load_benchmark()
    sentiment_data = analyze_sentiment()
    cross_ref = retention_relevance_ranking(company_key, sentiment_data)

    return {
        "pricing_position": pricing_position(data, tier),
        "feature_gap_matrix": feature_gap_matrix(data),
        "gap_analysis": what_is_company_missing(data, company_key),
        "sentiment": sentiment_data,
        "retention_cross_reference": cross_ref,
        "meta": data["_meta"],
    }


if __name__ == "__main__":
    report = full_report("netflix")
    print(json.dumps(report, indent=2))
