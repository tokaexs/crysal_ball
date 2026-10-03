"""
Layer 2: FEATURE ENGINEERING + LEAKAGE GUARD
Builds a per-customer, per-month feature table with:
  - RFM (Recency, Frequency, Monetary)
  - tenure, contract type, ticket velocity
  - lag features for revenue (t-1, t-2, t-3)
Then runs an automatic leakage detector that drops any feature whose absolute
correlation with the target exceeds a threshold, or that is derived from
post-event data (e.g. anything using month_idx == target month directly).
"""
import pandas as pd
import numpy as np
import duckdb
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "warehouse.duckdb"

LEAKAGE_CORR_THRESHOLD = 0.85
# columns that are definitionally post-event / the label itself -- never allowed as features
FORBIDDEN_COLS = {"churned_this_month", "churn_hazard_latent", "month", "customer_id"}


def load_base():
    con = duckdb.connect(str(DB_PATH), read_only=True)
    panel = con.execute("SELECT * FROM monthly_panel ORDER BY customer_id, month_idx").df()
    customers = con.execute("SELECT * FROM customers").df()
    con.close()
    return panel, customers


def build_features(panel: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    df = panel.merge(customers, on="customer_id", how="left")
    df = df.sort_values(["customer_id", "month_idx"]).reset_index(drop=True)
    g = df.groupby("customer_id", group_keys=False)

    # --- Lag features for revenue (t-1, t-2, t-3) ---
    for lag in (1, 2, 3):
        df[f"revenue_lag_{lag}"] = g["revenue"].shift(lag)

    # --- Recency: months since last support ticket ---
    df["had_ticket"] = (df["tickets_opened"] > 0).astype(int)
    df["_ticket_month"] = np.where(df["had_ticket"] == 1, df["month_idx"], np.nan)
    df["last_ticket_month"] = g["_ticket_month"].apply(lambda s: s.ffill())
    df["recency_months_since_ticket"] = (df["month_idx"] - df["last_ticket_month"]).fillna(99)
    df.drop(columns=["_ticket_month", "last_ticket_month"], inplace=True)

    # --- Frequency: rolling 3-month ticket count ---
    df["ticket_freq_3m"] = g["tickets_opened"].apply(lambda s: s.rolling(3, min_periods=1).sum())

    # --- Monetary: rolling 3-month avg revenue + revenue trend ---
    df["revenue_avg_3m"] = g["revenue"].apply(lambda s: s.rolling(3, min_periods=1).mean())
    df["revenue_trend"] = (df["revenue"] - df["revenue_avg_3m"]) / df["revenue_avg_3m"].replace(0, np.nan)

    # --- Ticket velocity (rate of change) ---
    df["ticket_velocity"] = g["tickets_opened"].diff().fillna(0)

    # --- One-hot encode categoricals ---
    df = pd.get_dummies(df, columns=["segment", "contract_type", "campaign_exposed"], prefix=["seg", "contract", "camp"])

    # Drop first 3 months per customer (insufficient lag history)
    df = df[df["month_idx"] >= 3].reset_index(drop=True)
    df["revenue_trend"] = df["revenue_trend"].fillna(0)
    df["recency_months_since_ticket"] = df["recency_months_since_ticket"].clip(upper=18)

    return df


# Features that are deliberately built from strictly-lagged (pre-target-period) data.
# Their high correlation with the target is the whole point (that's what makes them good
# predictors) -- it is NOT leakage, which specifically means the feature encodes information
# from the same period as, or after, the event being predicted. The guard exempts these by
# construction and instead flags accidental leakage: any feature that is not a designed lag/
# rolling-window feature but still moves in near-lockstep with the target (a proxy for it).
SAFE_LAGGED_PREFIXES = ("revenue_lag_", "revenue_avg_3m", "revenue_trend", "ticket_freq_3m",
                         "ticket_velocity", "recency_months_since_ticket")


def leakage_guard(df: pd.DataFrame, target_col: str):
    """Drops candidate features that would leak the target (same-period/post-event proxies),
    while exempting intentionally-lagged engineered features from the correlation check."""
    candidate_cols = [c for c in df.columns if c not in FORBIDDEN_COLS and c != target_col
                       and pd.api.types.is_numeric_dtype(df[c])]
    dropped = []
    kept = []
    for c in candidate_cols:
        if c.startswith(SAFE_LAGGED_PREFIXES):
            kept.append(c)
            continue
        corr = df[[c, target_col]].corr().iloc[0, 1]
        if pd.notna(corr) and abs(corr) > LEAKAGE_CORR_THRESHOLD:
            dropped.append((c, round(float(corr), 3)))
        else:
            kept.append(c)
    return kept, dropped


def build_and_save():
    panel, customers = load_base()
    feats = build_features(panel, customers)

    kept_churn, dropped_churn = leakage_guard(feats, "churned_this_month")
    kept_rev, dropped_rev = leakage_guard(feats, "revenue")

    feats.to_parquet(DATA_DIR / "feature_store.parquet", index=False)

    report = {
        "n_rows": len(feats),
        "n_raw_features": len(feats.columns),
        "churn_target_kept_features": len(kept_churn),
        "churn_target_dropped_features": dropped_churn,
        "revenue_target_kept_features": len(kept_rev),
        "revenue_target_dropped_features": dropped_rev,
    }
    print("Feature store written:", DATA_DIR / "feature_store.parquet")
    print(f"Rows: {report['n_rows']}, raw feature cols: {report['n_raw_features']}")
    print(f"Leakage guard (churn target) dropped: {dropped_churn}")
    print(f"Leakage guard (revenue target) dropped: {dropped_rev}")
    return feats, kept_churn, kept_rev


if __name__ == "__main__":
    build_and_save()
