"""
Layer 1: DATA LAYER
Generates realistic synthetic Telco Churn + SaaS Revenue + Support Tickets +
Campaign data with actual signal (not pure noise), loads it into a single-file
DuckDB warehouse, and materializes a Parquet feature store.
"""
import numpy as np
import pandas as pd
import duckdb
from pathlib import Path
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "warehouse.duckdb"

N_CUSTOMERS = 6000
N_MONTHS = 18  # 18 months of history -> last month = prediction target month
START_DATE = datetime(2025, 4, 1)

CONTRACT_TYPES = ["month-to-month", "one-year", "two-year"]
CONTRACT_CHURN_MULT = {"month-to-month": 1.9, "one-year": 0.75, "two-year": 0.35}
SEGMENTS = ["A_enterprise", "B_smb", "C_individual"]
CAMPAIGN_TYPES = ["email_promo", "discount_push", "feature_announce", "none"]

# Historical retention actions: randomly assigned (like a real experiment) to a slice of
# at-risk customer-months, each with a TRUE heterogeneous treatment effect on churn hazard
# that varies by segment/tenure. This is what makes genuine uplift modeling possible later,
# instead of guessing at "action efficacy" with no data to learn it from.
ACTIONS = ["discount_10pct", "free_upgrade_3mo", "account_manager_call", "loyalty_bonus", "control"]
ASSIGNMENT_RATE = 0.18  # fraction of at-risk-month rows that get randomly assigned a real action

# true multiplicative effect on hazard, by (action, segment) -- this is the ground truth an
# uplift model has to recover from the logged outcomes. "control" is always 1.0 (no effect).
TRUE_UPLIFT_EFFECT = {
    "discount_10pct":        {"A_enterprise": 0.55, "B_smb": 0.65, "C_individual": 0.80},
    "free_upgrade_3mo":      {"A_enterprise": 0.60, "B_smb": 0.75, "C_individual": 0.90},
    "account_manager_call":  {"A_enterprise": 0.45, "B_smb": 0.85, "C_individual": 0.95},
    "loyalty_bonus":         {"A_enterprise": 0.90, "B_smb": 0.80, "C_individual": 0.70},
    "control":               {"A_enterprise": 1.00, "B_smb": 1.00, "C_individual": 1.00},
}


def _gen_customers():
    seg = RNG.choice(SEGMENTS, size=N_CUSTOMERS, p=[0.15, 0.35, 0.5])
    contract = RNG.choice(CONTRACT_TYPES, size=N_CUSTOMERS, p=[0.5, 0.3, 0.2])
    tenure_start = RNG.integers(0, 48, size=N_CUSTOMERS)  # months tenure at data start
    base_price = np.where(seg == "A_enterprise", RNG.normal(4200, 800, N_CUSTOMERS),
                  np.where(seg == "B_smb", RNG.normal(1400, 300, N_CUSTOMERS),
                           RNG.normal(450, 120, N_CUSTOMERS)))
    base_price = np.clip(base_price, 100, None)
    ltv_mult = RNG.normal(1.0, 0.15, N_CUSTOMERS)
    df = pd.DataFrame({
        "customer_id": [f"C{i:06d}" for i in range(N_CUSTOMERS)],
        "segment": seg,
        "contract_type": contract,
        "tenure_start_months": tenure_start,
        "base_monthly_price": base_price.round(2),
        "ltv_multiplier": ltv_mult.clip(0.5, 1.8),
    })
    return df


def _gen_monthly_panel(customers: pd.DataFrame):
    """Simulate month-by-month revenue, tickets, campaigns, and a latent churn hazard
    that actually depends on features, so models have real signal to learn."""
    rows = []
    active = {cid: True for cid in customers.customer_id}
    cust_idx = customers.set_index("customer_id")

    for cid, crow in cust_idx.iterrows():
        tenure = crow.tenure_start_months
        price = crow.base_monthly_price
        contract = crow.contract_type
        churned_month = None
        ticket_velocity_state = RNG.poisson(0.4)  # rolling baseline

        for m in range(N_MONTHS):
            if churned_month is not None:
                break
            month_date = START_DATE + pd.DateOffset(months=m)
            tenure += 1

            # support tickets this month (mean-reverting w/ occasional spikes)
            spike = RNG.random() < 0.06
            tickets = RNG.poisson(1.2 + ticket_velocity_state + (3 if spike else 0))
            ticket_velocity_state = 0.6 * ticket_velocity_state + 0.4 * tickets * 0.3

            # campaign exposure
            campaign = RNG.choice(CAMPAIGN_TYPES, p=[0.25, 0.15, 0.2, 0.4])

            # revenue this month: base price + small drift + noise, discount campaigns cut revenue slightly
            drift = 1 + 0.002 * m + RNG.normal(0, 0.04)
            camp_discount = 0.9 if campaign == "discount_push" else 1.0
            revenue = max(0, price * drift * camp_discount + RNG.normal(0, price * 0.05))

            # ---- latent churn hazard (this is the ground-truth signal) ----
            hazard = 0.02  # base monthly hazard
            hazard *= CONTRACT_CHURN_MULT[contract]
            hazard *= np.exp(-tenure / 60)  # longer tenure -> stickier
            hazard *= 1 + 0.18 * tickets  # ticket velocity raises risk
            hazard *= 0.75 if campaign in ("discount_push", "feature_announce") else 1.0
            hazard = np.clip(hazard, 0.001, 0.9)

            # ---- randomized action assignment (the "historical experiment" log) ----
            # Mimic a real retention program: only customers already flagged as somewhat at
            # risk get offered a play, but WHICH play they get is randomly assigned -- this
            # is exactly the randomization an uplift model needs to isolate a causal effect
            # (as opposed to "customers who happened to churn less were the ones who acted
            # healthier anyway", which a naive correlation would conflate).
            action_taken = "control"
            in_action_experiment = False
            if hazard > 0.03 and RNG.random() < ASSIGNMENT_RATE:
                action_taken = RNG.choice(ACTIONS)
                in_action_experiment = True  # this row is part of the randomized pool
                # (action_taken=="control" here means "randomized into the control ARM",
                # distinct from the 91%+ of rows never entered into the experiment at all)
            true_mult = TRUE_UPLIFT_EFFECT[action_taken][crow.segment]
            hazard_after_action = np.clip(hazard * true_mult, 0.001, 0.9)

            will_churn_this_month = RNG.random() < hazard_after_action

            rows.append({
                "customer_id": cid,
                "month": month_date.strftime("%Y-%m-01"),
                "month_idx": m,
                "tenure_months": tenure,
                "tickets_opened": int(tickets),
                "campaign_exposed": campaign,
                "revenue": round(float(revenue), 2),
                "churn_hazard_latent": round(float(hazard), 5),
                "action_taken": action_taken,
                "in_action_experiment": in_action_experiment,
                "churned_this_month": bool(will_churn_this_month),
            })

            if will_churn_this_month:
                churned_month = m

    panel = pd.DataFrame(rows)
    return panel


def build():
    customers = _gen_customers()
    panel = _gen_monthly_panel(customers)

    con = duckdb.connect(str(DB_PATH))
    con.execute("CREATE OR REPLACE TABLE customers AS SELECT * FROM customers")
    con.execute("CREATE OR REPLACE TABLE monthly_panel AS SELECT * FROM panel")
    con.close()

    customers.to_parquet(DATA_DIR / "customers.parquet", index=False)
    panel.to_parquet(DATA_DIR / "monthly_panel.parquet", index=False)

    print(f"customers: {len(customers)} rows")
    print(f"monthly_panel: {len(panel)} rows across {N_MONTHS} months")
    print(f"overall churn rate: {panel.churned_this_month.mean():.3%}")
    print(f"warehouse written to {DB_PATH}")


if __name__ == "__main__":
    build()
