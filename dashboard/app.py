"""
Layer 7: DASHBOARD LAYER (Streamlit, 5 tabs)
Run with: streamlit run dashboard/app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

import importlib
import llm
import competitive

importlib.reload(llm)
importlib.reload(competitive)

from llm import explain_customer_risk, draft_retention_message, ask_copilot, competitive_briefing
from competitive import load_benchmark, full_report

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(page_title="Crystal Ball", page_icon="🔮", layout="wide")


@st.cache_data
def load_data():
    feats = pd.read_parquet(DATA_DIR / "feature_store.parquet")
    customers = pd.read_parquet(DATA_DIR / "customers.parquet")
    presc = pd.read_parquet(DATA_DIR / "prescriptive_results.parquet")
    ranking = pd.read_parquet(DATA_DIR / "action_ranking.parquet")
    with open(MODEL_DIR / "trust_card.json") as f:
        trust = json.load(f)
    with open(MODEL_DIR / "metrics.json") as f:
        metrics = json.load(f)
    return feats, customers, presc, ranking, trust, metrics


@st.cache_resource
def load_models():
    churn_art = joblib.load(MODEL_DIR / "churn_model.joblib")
    revenue_art = joblib.load(MODEL_DIR / "revenue_model.joblib")
    return churn_art, revenue_art


feats, customers, presc, ranking, trust, metrics = load_data()
churn_art, revenue_art = load_models()
churn_model, churn_cols = churn_art["model"], churn_art["feature_cols"]

latest = feats.sort_values("month_idx").groupby("customer_id").tail(1).copy()
latest_X = latest[churn_cols].fillna(0)
latest["churn_risk"] = churn_model.predict_proba(latest_X.values)[:, 1]
latest = latest.merge(customers[["customer_id", "segment", "contract_type", "base_monthly_price", "ltv_multiplier"]],
                       on="customer_id", suffixes=("", "_c"))
latest["ltv"] = latest["base_monthly_price"] * 12 * latest["ltv_multiplier"]

st.title("🔮 Crystal Ball — Churn, Revenue & Prescriptive Retention")
st.caption("Predict who leaves, forecast what you'll earn, and know exactly what to do about it.")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Executive Summary", "🔥 Churn Heatmap", "📈 Revenue Forecast",
    "🎮 What-If Simulator", "🛡️ Trust Center", "🤖 AI Copilot", "🏆 Competitive Intel"
])

# ---------------------------------------------------------------- TAB 1 ----
with tab1:
    at_risk_count = (latest.churn_risk > 0.05).sum()
    at_risk_value = latest.loc[latest.churn_risk > 0.05, "ltv"].sum()
    next_month_revenue = latest["base_monthly_price"].sum()
    top_action = ranking.iloc[0] if len(ranking) else None

    c1, c2, c3 = st.columns(3)
    c1.metric("Next month revenue (run-rate)", f"₹{next_month_revenue:,.0f}")
    c2.metric("At-risk customers (>5% churn risk)", f"{at_risk_count:,}", f"₹{at_risk_value:,.0f} LTV exposed")
    if top_action is not None:
        c3.metric("Top recommended play", top_action["action"],
                  f"₹{top_action['total_expected_save_inr']:,.0f} save potential")

    st.divider()
    st.subheader("💡 Recommended plan")
    if len(ranking):
        for _, r in ranking.head(3).iterrows():
            with st.container(border=True):
                cols = st.columns([3, 1, 1, 1])
                cols[0].markdown(f"**{r['action']}** → {int(r['customers_targeted'])} customers")
                cols[1].metric("Save", f"₹{r['total_expected_save_inr']:,.0f}")
                cols[2].metric("Cost", f"₹{r['total_cost_inr']:,.0f}")
                cols[3].metric("Avg ROI", f"{r['avg_roi']:.1f}x")
    st.button("✅ Accept Plan", type="primary")
    st.caption("(Demo action — wire this to a workflow/CRM trigger in production.)")

# ---------------------------------------------------------------- TAB 2 ----
with tab2:
    st.subheader("Customer risk table")
    seg_filter = st.multiselect("Segment", options=sorted(latest.segment.unique()),
                                 default=sorted(latest.segment.unique()))
    filtered = latest[latest.segment.isin(seg_filter)].sort_values("churn_risk", ascending=False)

    def risk_color(v):
        if v > 0.15:
            return "background-color:#ffb3b3"
        elif v > 0.05:
            return "background-color:#ffe0b3"
        return "background-color:#c9f2c9"

    show_cols = ["customer_id", "segment", "contract_type", "tenure_months", "tickets_opened",
                 "churn_risk", "ltv"]
    styler = filtered[show_cols].head(300).style
    if hasattr(styler, "map"):
        styled = styler.map(risk_color, subset=["churn_risk"])
    else:
        styled = styler.applymap(risk_color, subset=["churn_risk"])
    styled = styled.format({"churn_risk": "{:.1%}", "ltv": "₹{:,.0f}"})
    st.dataframe(styled, height=450, use_container_width=True)

    st.subheader("SHAP explanation for a selected customer")
    sel_cid = st.selectbox("Customer", filtered.customer_id.head(50).tolist())
    shap_df = None
    if sel_cid:
        row = filtered[filtered.customer_id == sel_cid].iloc[0]
        X = row[churn_cols].fillna(0).values.reshape(1, -1).astype(float)
        try:
            import shap
            base_est = churn_model.calibrated_classifiers_[0].estimator
            explainer = shap.TreeExplainer(base_est)
            sv = explainer.shap_values(X)
            sv = sv[0] if isinstance(sv, list) else sv[0]
            order = np.argsort(-np.abs(sv))[:8]
            shap_df = pd.DataFrame({"feature": [churn_cols[i] for i in order],
                                     "impact": [sv[i] for i in order]}).sort_values("impact")
            fig = px.bar(shap_df, x="impact", y="feature", orientation="h",
                         color="impact", color_continuous_scale="RdBu_r",
                         title=f"Top drivers of churn risk for {sel_cid}")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"SHAP explanation unavailable: {e}")

        st.divider()
        colX, colY = st.columns(2)
        with colX:
            if st.button("🤖 Explain this customer in plain English", key="explain_btn"):
                cust_presc = presc[(presc.customer_id == sel_cid) & (presc.action != "Do nothing")]
                rec_action, rec_save = None, None
                if not cust_presc.empty:
                    top = cust_presc.sort_values("roi", ascending=False).iloc[0]
                    rec_action, rec_save = top["action"], float(top["expected_save_inr"])
                driver_dicts = (shap_df.rename(columns={"impact": "shap_value"}).to_dict("records")
                                 if shap_df is not None else [])
                with st.spinner("Asking Claude..."):
                    narrative = explain_customer_risk(sel_cid, float(row["churn_risk"]),
                                                       driver_dicts, rec_action, rec_save)
                st.info(narrative)
        with colY:
            action_options = presc[presc.customer_id == sel_cid]["action"].unique().tolist() or ["10% discount offer"]
            chosen_action = st.selectbox("Action to draft a message for", action_options, key="draft_action_select")
            if st.button("✉️ Draft retention message", key="draft_btn"):
                with st.spinner("Drafting..."):
                    draft = draft_retention_message(sel_cid, row["segment"], chosen_action)
                st.text_area("Draft", draft, height=150)

# ---------------------------------------------------------------- TAB 3 ----
with tab3:
    st.subheader("Actual vs predicted revenue with 90% prediction interval")
    seg_pick = st.selectbox("Segment to forecast", sorted(customers.segment.unique()))
    seg_ids = customers[customers.segment == seg_pick].customer_id
    hist = feats[feats.customer_id.isin(seg_ids)].groupby("month_idx")["revenue"].sum().reset_index()

    rev_cols = revenue_art["feature_cols"]
    seg_feats = feats[feats.customer_id.isin(seg_ids)].copy()
    Xr = seg_feats[rev_cols].fillna(0).values
    seg_feats["pred"] = revenue_art["point_model"].predict(Xr)
    seg_feats["pred_lo"] = revenue_art["lo_model"].predict(Xr)
    seg_feats["pred_hi"] = revenue_art["hi_model"].predict(Xr)
    agg = seg_feats.groupby("month_idx").agg(actual=("revenue", "sum"), pred=("pred", "sum"),
                                              lo=("pred_lo", "sum"), hi=("pred_hi", "sum")).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=agg.month_idx, y=agg.hi, line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=agg.month_idx, y=agg.lo, line=dict(width=0), fill="tonexty",
                              fillcolor="rgba(100,150,250,0.2)", name="90% PI"))
    fig.add_trace(go.Scatter(x=agg.month_idx, y=agg.actual, mode="lines+markers", name="Actual",
                              line=dict(color="black")))
    fig.add_trace(go.Scatter(x=agg.month_idx, y=agg.pred, mode="lines+markers", name="Predicted",
                              line=dict(color="orange", dash="dash")))
    fig.update_layout(title=f"{seg_pick} — monthly revenue", xaxis_title="Month index", yaxis_title="Revenue (₹)")
    st.plotly_chart(fig, use_container_width=True)

    conf_badge = "🟢 High confidence" if metrics["revenue"]["mape_cv_mean"] < 0.15 else "🟡 Moderate confidence"
    st.info(f"{conf_badge} — CV MAPE {metrics['revenue']['mape_cv_mean']:.1%}, "
            f"PI coverage {metrics['revenue']['pi_coverage_90_cv_mean']:.0%} (target 90%)")

# ---------------------------------------------------------------- TAB 4 ----
with tab4:
    st.subheader("🎮 What-if simulator")
    st.caption("Adjust the levers below and watch churn risk, rupee impact, and ROI recalculate live.")

    sim_cid = st.selectbox("Pick a customer to simulate", latest.sort_values("churn_risk", ascending=False)
                            .customer_id.head(100).tolist(), key="sim_select")
    row = latest[latest.customer_id == sim_cid].iloc[0]

    col_a, col_b, col_c = st.columns(3)
    discount_pct = col_a.slider("Discount %", 0, 30, 0, step=5)
    contract_upgrade = col_b.checkbox("Upgrade to 1-year contract")
    support_sla = col_c.slider("Support SLA boost (tickets resolved faster, -N recent tickets)", 0, 3, 0)

    sim_row = row[churn_cols].fillna(0).copy()
    if discount_pct > 0:
        for c in ["camp_discount_push"]:
            if c in sim_row.index:
                sim_row[c] = 1
        if "camp_none" in sim_row.index:
            sim_row["camp_none"] = 0
        if "revenue_trend" in sim_row.index:
            sim_row["revenue_trend"] = sim_row["revenue_trend"] - discount_pct / 100 * 0.6
    if contract_upgrade:
        if "contract_month-to-month" in sim_row.index:
            sim_row["contract_month-to-month"] = 0
        if "contract_one-year" in sim_row.index:
            sim_row["contract_one-year"] = 1
    if support_sla > 0:
        if "ticket_velocity" in sim_row.index:
            sim_row["ticket_velocity"] = max(sim_row["ticket_velocity"] - support_sla, -2)
        if "recency_months_since_ticket" in sim_row.index:
            sim_row["recency_months_since_ticket"] = 0

    x_sim = sim_row.values.reshape(1, -1).astype(float)
    new_risk = float(churn_model.predict_proba(x_sim)[0, 1])
    old_risk = float(row["churn_risk"])
    ltv = float(row["ltv"])
    saved_inr = max(old_risk - new_risk, 0) * ltv
    est_cost = (discount_pct * 15) + (400 if contract_upgrade else 0) + (support_sla * 80)
    roi = (saved_inr - est_cost) / est_cost if est_cost > 0 else 0.0

    g1, g2, g3 = st.columns(3)
    with g1:
        fig = go.Figure(go.Indicator(mode="gauge+number+delta", value=new_risk * 100,
                                      delta={"reference": old_risk * 100, "decreasing": {"color": "green"}},
                                      title={"text": "Churn risk %"},
                                      gauge={"axis": {"range": [0, 100]},
                                             "bar": {"color": "darkred"},
                                             "steps": [{"range": [0, 5], "color": "#c9f2c9"},
                                                       {"range": [5, 15], "color": "#ffe0b3"},
                                                       {"range": [15, 100], "color": "#ffb3b3"}]}))
        fig.update_layout(height=280, margin=dict(t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
    g2.metric("₹ Expected save", f"₹{saved_inr:,.0f}", f"vs baseline risk {old_risk:.1%}")
    with g3:
        fig2 = go.Figure(go.Indicator(mode="gauge+number", value=roi,
                                       title={"text": "ROI (x)"},
                                       gauge={"axis": {"range": [-2, 10]}, "bar": {"color": "green" if roi > 0 else "red"}}))
        fig2.update_layout(height=280, margin=dict(t=40, b=0))
        st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------- TAB 5 ----
with tab5:
    st.subheader("🛡️ Trust center")
    c1, c2, c3 = st.columns(3)
    c1.metric("Churn AUC (5-fold CV)", f"{metrics['churn']['auc_cv_mean']:.2f}")
    c2.metric("Revenue MAPE (CV)", f"{metrics['revenue']['mape_cv_mean']:.1%}")
    c3.metric("90% PI coverage", f"{metrics['revenue']['pi_coverage_90_cv_mean']:.0%}",
              "target 90%")

    drift = trust["drift"]
    color = {"green": "🟢", "yellow": "🟡", "red": "🔴"}[drift["alert_level"]]
    st.markdown(f"### {color} Drift status: **{drift['alert_level'].upper()}**")
    st.write(f"PSI: `{drift['psi']}` · KS statistic: `{drift['ks_statistic']}` "
             f"(p={drift['ks_p_value']}) · Retrain triggered: `{drift['retrain_triggered']}`")
    st.caption("Monitoring the `revenue` feature distribution, first 70% of history vs. most recent 30% "
               "(simulated nightly check — wire to a real scheduler for production).")

    st.divider()
    st.subheader("Model card")
    st.json(metrics)

# ---------------------------------------------------------------- TAB 6 ----
with tab6:
    st.subheader("🤖 Ask Crystal Ball")
    st.caption("Grounded in the current trust card, action ranking, and portfolio aggregates — "
               "not free-floating opinions. Set ANTHROPIC_API_KEY to enable real answers.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(msg)

    example_qs = ["Which action has the best ROI right now?", "How healthy is the revenue forecast?",
                  "What's driving churn risk most broadly?"]
    cols = st.columns(len(example_qs))
    picked = None
    for c, q in zip(cols, example_qs):
        if c.button(q, key=f"ex_{q}"):
            picked = q

    typed = st.chat_input("Ask about churn risk, revenue, or retention economics...")
    question = picked or typed

    if question:
        st.session_state.chat_history.append(("user", question))
        with st.chat_message("user"):
            st.write(question)
        context = {
            "trust_card": trust,
            "action_ranking": ranking.to_dict("records"),
            "n_customers": int(customers.customer_id.nunique()),
            "avg_revenue_last_month": float(latest["base_monthly_price"].mean()),
            "at_risk_count": int((latest.churn_risk > 0.05).sum()),
        }
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ask_copilot(question, context)
            st.write(answer)
        st.session_state.chat_history.append(("assistant", answer))

# ---------------------------------------------------------------- TAB 7 ----
with tab7:
    st.subheader("🏆 Competitive Intelligence")
    bench_data = load_benchmark()
    meta = bench_data["_meta"]
    st.caption(f"Vertical: {meta['vertical']} · Compiled {meta['compiled']}")
    with st.expander("⚠️ Methodology (read before presenting this)"):
        st.write(meta["methodology"])
        st.warning(meta["caveat"])

    company_options = {v["display_name"]: k for k, v in bench_data["companies"].items()}
    picked_name = st.selectbox("Benchmark which company?", list(company_options.keys()))
    picked_key = company_options[picked_name]

    report = full_report(picked_key)

    st.markdown("#### 💰 Pricing position (approx. monthly, India, cheapest → most expensive)")
    price_df = pd.DataFrame(report["pricing_position"])
    st.dataframe(price_df, use_container_width=True, hide_index=True)
    for row in report["pricing_position"]:
        if row["comparability_note"]:
            st.caption(f"⚠️ {row['company']}: {row['comparability_note']}")

    st.markdown("#### ✅ Feature matrix")
    matrix_df = pd.DataFrame(report["feature_gap_matrix"]).T
    st.dataframe(matrix_df, use_container_width=True)

    st.markdown(f"#### 🔍 What {picked_name} is missing vs. competitors")
    gap = report["gap_analysis"]
    if gap["missing_vs_competitors"]:
        for g in gap["missing_vs_competitors"]:
            st.write(f"❌ **{g['feature'].replace('_', ' ')}** — has it: "
                      f"{', '.join(g['competitors_with_it'])}")
    else:
        st.write("No gaps found in this feature set (doesn't mean no gaps exist — see methodology above).")
    if gap["unique_advantages"]:
        for a in gap["unique_advantages"]:
            st.write(f"✅ **{a['feature'].replace('_', ' ')}** — unique to {a['unique_to']}")

    st.markdown("#### 🎯 ML: priority-ranked gaps (real sentiment × our own model's real feature importance)")
    st.caption("priority_score = |competitor sentiment| × how much that aspect actually drives churn in "
               "OUR trained model (not a guess — pulled from the fitted LightGBM model's feature_importances_). "
               "A score of 0 means our data has no signal there, not that it doesn't matter.")
    cross_ref = report.get("retention_cross_reference", {})
    ranked_priority = cross_ref.get("ranked_by_priority", [])
    if ranked_priority:
        priority_df = pd.DataFrame(ranked_priority)
        st.dataframe(priority_df, use_container_width=True, hide_index=True)
    else:
        st.write("Priority ranking not available for this profile.")

    sentiment_dict = report.get("sentiment", {})
    if sentiment_dict:
        with st.expander("Sentiment detail (VADER, real model, sourced+paraphrased statements)"):
            for comp_key, s in sentiment_dict.items():
                st.write(f"**{s['company']}** — overall: {s['overall_compound']} "
                          f"({s['n_statements']} statements, {s['confidence_note']})")
                st.json(s["aspects"])

    if st.button("🤖 Generate competitive briefing", key="comp_briefing_btn"):
        with st.spinner("Generating briefing..."):
            try:
                narrative = competitive_briefing(picked_name, gap, report.get("pricing_position", []),
                                                  ranked_priority)
            except TypeError:
                narrative = competitive_briefing(picked_name, gap, report.get("pricing_position", []))
        st.info(narrative)

st.sidebar.title("🔮 Crystal Ball")
st.sidebar.markdown(
    "**Pipeline:** Data → Features (+leakage guard) → Dual models → "
    "Prescriptive engine → Backtesting → FastAPI → this dashboard."
)
st.sidebar.caption(f"{len(customers):,} customers · {feats.month_idx.nunique()} months of history")