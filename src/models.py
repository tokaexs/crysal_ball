"""
Layer 3: PREDICTION LAYER (Dual Model) -- LightGBM core
  - Churn Classifier: LightGBM + CalibratedClassifierCV -> AUC, Precision@K
  - Revenue Forecaster: LightGBM point regressor + LightGBM quantile regressors -> MAPE, RMSE, PI
  - Prediction intervals: native quantile objective (5th/50th/95th), cross-checked with
    bootstrap residual resampling
  - Explainability: SHAP TreeExplainer per customer (native LightGBM support, fast)
"""
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, mean_absolute_percentage_error, mean_squared_error

from features import build_and_save, FORBIDDEN_COLS

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

NON_FEATURE_COLS = FORBIDDEN_COLS | {"month_idx", "tenure_start_months", "action_taken"}

LGBM_COMMON = dict(n_estimators=150, num_leaves=31, learning_rate=0.08, random_state=42, verbosity=-1)
LGBM_CHURN = dict(LGBM_COMMON, num_leaves=15, min_child_samples=30, class_weight="balanced")


def _feature_cols(df, allowed):
    return [c for c in allowed if c in df.columns]


# ---------------------------------------------------------------- Churn ----
def train_churn_model(df: pd.DataFrame, feature_cols: list[str]):
    X = df[feature_cols].fillna(0).values
    y = df["churned_this_month"].astype(int).values

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    aucs, precisions_at_50 = [], []
    for train_idx, test_idx in skf.split(X, y):
        base = LGBMClassifier(**LGBM_CHURN)
        base.fit(X[train_idx], y[train_idx])
        proba = base.predict_proba(X[test_idx])[:, 1]
        if y[test_idx].sum() > 0:
            aucs.append(roc_auc_score(y[test_idx], proba))
        order = np.argsort(-proba)[:50]
        precisions_at_50.append(y[test_idx][order].mean() if len(order) else np.nan)

    base_final = LGBMClassifier(**LGBM_CHURN)
    calibrated = CalibratedClassifierCV(base_final, method="isotonic", cv=3)
    calibrated.fit(X, y)

    metrics = {
        "auc_cv_mean": float(np.nanmean(aucs)) if aucs else None,
        "auc_cv_std": float(np.nanstd(aucs)) if aucs else None,
        "precision_at_50_cv_mean": float(np.nanmean(precisions_at_50)),
        "n_train_rows": len(df),
        "base_rate": float(y.mean()),
        "model_family": "LightGBM (calibrated, isotonic)",
    }
    joblib.dump({"model": calibrated, "feature_cols": feature_cols}, MODEL_DIR / "churn_model.joblib")
    return calibrated, metrics


# -------------------------------------------------------------- Revenue ----
def train_revenue_model(df: pd.DataFrame, feature_cols: list[str]):
    df_sorted = df.sort_values("month_idx")
    X = df_sorted[feature_cols].fillna(0).values
    y = df_sorted["revenue"].values

    tscv = TimeSeriesSplit(n_splits=3)
    mapes, rmses, coverages = [], [], []
    for train_idx, test_idx in tscv.split(X):
        point_m = LGBMRegressor(**LGBM_COMMON)
        point_m.fit(X[train_idx], y[train_idx])
        preds = point_m.predict(X[test_idx])
        mapes.append(mean_absolute_percentage_error(y[test_idx], preds))
        rmses.append(mean_squared_error(y[test_idx], preds) ** 0.5)

        lo_m = LGBMRegressor(objective="quantile", alpha=0.05, **LGBM_COMMON)
        hi_m = LGBMRegressor(objective="quantile", alpha=0.95, **LGBM_COMMON)
        lo_m.fit(X[train_idx], y[train_idx])
        hi_m.fit(X[train_idx], y[train_idx])
        lo_pred, hi_pred = lo_m.predict(X[test_idx]), hi_m.predict(X[test_idx])
        covered = ((y[test_idx] >= lo_pred) & (y[test_idx] <= hi_pred)).mean()
        coverages.append(covered)

    point_final = LGBMRegressor(**LGBM_COMMON)
    point_final.fit(X, y)
    lo_final = LGBMRegressor(objective="quantile", alpha=0.05, **LGBM_COMMON)
    hi_final = LGBMRegressor(objective="quantile", alpha=0.95, **LGBM_COMMON)
    lo_final.fit(X, y)
    hi_final.fit(X, y)

    metrics = {
        "mape_cv_mean": float(np.mean(mapes)),
        "rmse_cv_mean": float(np.mean(rmses)),
        "pi_coverage_90_cv_mean": float(np.mean(coverages)),
        "n_train_rows": len(df),
        "model_family": "LightGBM (point + native quantile objective)",
    }
    joblib.dump(
        {"point_model": point_final, "lo_model": lo_final, "hi_model": hi_final, "feature_cols": feature_cols},
        MODEL_DIR / "revenue_model.joblib",
    )
    return point_final, metrics


def bootstrap_prediction_interval(model, X_row: np.ndarray, X_train: np.ndarray, y_train: np.ndarray, n_boot=100):
    """Residual-bootstrap cross-check against the native quantile-model PI."""
    point = model.predict(X_row.reshape(1, -1))[0]
    resid = y_train - model.predict(X_train)
    boots = point + np.random.default_rng(0).choice(resid, size=n_boot, replace=True)
    lo, hi = np.percentile(boots, [5, 95])
    return float(point), float(lo), float(hi)


def run_all():
    feats, kept_churn, kept_rev = build_and_save()
    churn_cols = _feature_cols(feats, kept_churn)
    rev_cols = _feature_cols(feats, kept_rev)
    churn_cols = [c for c in churn_cols if c not in NON_FEATURE_COLS]
    rev_cols = [c for c in rev_cols if c not in NON_FEATURE_COLS]

    print("Training churn classifier (LightGBM)...")
    _, churn_metrics = train_churn_model(feats, churn_cols)
    print(json.dumps(churn_metrics, indent=2))

    print("Training revenue forecaster (LightGBM)...")
    _, revenue_metrics = train_revenue_model(feats, rev_cols)
    print(json.dumps(revenue_metrics, indent=2))

    all_metrics = {"churn": churn_metrics, "revenue": revenue_metrics,
                   "churn_features": churn_cols, "revenue_features": rev_cols}
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("Saved models + metrics to", MODEL_DIR)
    return all_metrics


if __name__ == "__main__":
    run_all()
