"""
Layer 5: BACKTESTING & TRUST LAYER
Consolidates CV metrics already computed in models.py into a trust "metrics card",
and layer 8's drift check: PSI + KS test on features/predictions, nightly-simulated.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


def population_stability_index(expected: np.ndarray, actual: np.ndarray, bins=10):
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    exp_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    act_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    exp_pct = np.clip(exp_pct, 1e-4, None)
    act_pct = np.clip(act_pct, 1e-4, None)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def drift_status(feature_col="revenue", split_frac=0.7):
    feats = pd.read_parquet(DATA_DIR / "feature_store.parquet")
    feats = feats.sort_values("month_idx")
    cut = int(len(feats) * split_frac)
    baseline = feats[feature_col].values[:cut]
    recent = feats[feature_col].values[cut:]

    psi = population_stability_index(baseline, recent)
    ks_stat, ks_p = ks_2samp(baseline, recent)

    if psi < 0.1:
        alert = "green"
    elif psi < 0.2:
        alert = "yellow"
    else:
        alert = "red"

    return {
        "feature_monitored": feature_col,
        "psi": round(psi, 4),
        "ks_statistic": round(float(ks_stat), 4),
        "ks_p_value": round(float(ks_p), 4),
        "alert_level": alert,
        "retrain_triggered": alert == "red",
    }


def trust_card():
    with open(MODEL_DIR / "metrics.json") as f:
        metrics = json.load(f)
    drift = drift_status()
    card = {
        "churn_auc": metrics["churn"]["auc_cv_mean"],
        "churn_base_rate": metrics["churn"]["base_rate"],
        "revenue_mape": metrics["revenue"]["mape_cv_mean"],
        "revenue_rmse": metrics["revenue"]["rmse_cv_mean"],
        "pi_coverage_90": metrics["revenue"]["pi_coverage_90_cv_mean"],
        "drift": drift,
    }
    with open(MODEL_DIR / "trust_card.json", "w") as f:
        json.dump(card, f, indent=2)
    return card


if __name__ == "__main__":
    print(json.dumps(trust_card(), indent=2))
