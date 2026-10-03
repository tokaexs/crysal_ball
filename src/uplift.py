"""
Layer 4 upgrade: CAUSAL UPLIFT MODELING (replaces the heuristic action-efficacy blend)

Uses a T-learner: for each retention action, train a churn classifier on customers who
were randomly assigned that action (the treated arm), and a separate churn classifier on
customers randomly assigned to the control arm -- both drawn ONLY from the properly
randomized experiment pool (`in_action_experiment == True`), never from the general
population, which is confounded (treated customers were pre-selected as at-risk).

Individual Treatment Effect (ITE) for a customer x, action a:
    ITE(x, a) = P_control(churn | x) - P_treated_a(churn | x)

A positive ITE means the action is predicted to reduce this customer's churn probability.
This is what the prescriptive engine should rank on -- an estimate of the causal effect,
not a correlational proxy.

Caveat (worth stating plainly): a T-learner is simple and interpretable but is known to
underperform an X-learner or doubly-robust estimator when the treated/control groups are
small or imbalanced -- which they are here (~2,600 experiment rows total, split 5 ways).
Treat these ITEs as directionally useful, not as tight point estimates.
"""
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

ACTIONS = ["discount_10pct", "free_upgrade_3mo", "account_manager_call", "loyalty_bonus"]
LGBM_UPLIFT = dict(n_estimators=80, num_leaves=7, min_child_samples=15,
                    learning_rate=0.08, random_state=42, verbosity=-1, class_weight="balanced")


def _experiment_pool():
    feats = pd.read_parquet(DATA_DIR / "feature_store.parquet")
    pool = feats[feats["in_action_experiment"] == True].copy()  # noqa: E712
    return pool


def train_uplift_models(feature_cols: list[str]):
    pool = _experiment_pool()
    if pool.empty:
        raise RuntimeError("No rows found with in_action_experiment=True -- regenerate data first.")

    control = pool[pool.action_taken == "control"]
    Xc = control[feature_cols].fillna(0).values
    yc = control["churned_this_month"].astype(int).values

    control_model = LGBMClassifier(**LGBM_UPLIFT)
    control_model.fit(Xc, yc)

    treated_models = {}
    diagnostics = {}
    for action in ACTIONS:
        arm = pool[pool.action_taken == action]
        if len(arm) < 30:
            diagnostics[action] = {"n": len(arm), "status": "skipped -- too few samples"}
            continue
        Xt = arm[feature_cols].fillna(0).values
        yt = arm["churned_this_month"].astype(int).values

        model = LGBMClassifier(**LGBM_UPLIFT)
        model.fit(Xt, yt)
        treated_models[action] = model

        # quick honesty check: does the treated model, scored on the control arm's own
        # customers, predict systematically lower risk than the control model does?
        # (a sanity check that the models learned something directionally real, not just
        # arm-specific noise)
        pred_treated_on_control_X = model.predict_proba(Xc)[:, 1]
        pred_control_on_control_X = control_model.predict_proba(Xc)[:, 1]
        mean_ite_on_control_pop = float(np.mean(pred_control_on_control_X - pred_treated_on_control_X))
        diagnostics[action] = {
            "n_treated_arm": len(arm),
            "raw_churn_rate_treated_arm": float(yt.mean()),
            "raw_churn_rate_control_arm": float(yc.mean()),
            "mean_ite_estimate_on_control_population": round(mean_ite_on_control_pop, 4),
        }

    joblib.dump({"control_model": control_model, "treated_models": treated_models,
                 "feature_cols": feature_cols}, MODEL_DIR / "uplift_models.joblib")
    with open(MODEL_DIR / "uplift_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)
    return control_model, treated_models, diagnostics


def score_ite(customer_row: pd.Series, feature_cols: list[str], control_model, treated_models: dict):
    """Returns {action: predicted_ite} for one customer's feature row."""
    x = customer_row[feature_cols].fillna(0).values.reshape(1, -1).astype(float)
    p_control = control_model.predict_proba(x)[0, 1]
    ites = {}
    for action, model in treated_models.items():
        p_treated = model.predict_proba(x)[0, 1]
        ites[action] = max(p_control - p_treated, 0.0)  # clip negative ITEs to 0 (no worse-than-doing-nothing plays)
    return ites, float(p_control)


if __name__ == "__main__":
    with open(MODEL_DIR / "metrics.json") as f:
        metrics = json.load(f)
    churn_cols = metrics["churn_features"]
    control_model, treated_models, diagnostics = train_uplift_models(churn_cols)
    print("Uplift model diagnostics (mean ITE should be positive if the action genuinely helps):")
    print(json.dumps(diagnostics, indent=2))
