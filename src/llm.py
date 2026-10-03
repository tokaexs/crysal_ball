"""
LLM layer -- turns model outputs into natural language a human can act on without
reading a metrics table: plain-English risk explanations, ready-to-send retention
messages, and a freeform Q&A copilot grounded in the current book of business.

Requires an ANTHROPIC_API_KEY environment variable. If it isn't set, every function
degrades gracefully with a clear message instead of crashing -- so the rest of the app
(models, API, dashboard) works with or without an LLM configured.
"""
from __future__ import annotations

import os
import json
from pathlib import Path
import requests
from dotenv import find_dotenv, load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def _get_api_info():
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH, override=True)
    else:
        load_dotenv(find_dotenv(), override=True)

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()

    if anthropic_key and anthropic_key.startswith("sk-ant-"):
        model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
        return "anthropic", anthropic_key, model
    elif openrouter_key or (anthropic_key and anthropic_key.startswith("sk-or-")):
        key = openrouter_key or anthropic_key
        model = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        return "openrouter", key, model
    elif anthropic_key:
        model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
        return "anthropic", anthropic_key, model
    return None, None, None


def _call_claude(system: str, user_message: str, max_tokens: int = 500) -> str:
    provider, key, model = _get_api_info()
    if not key:
        return ("[LLM layer inactive: set ANTHROPIC_API_KEY or OPENROUTER_API_KEY in your .env file to "
                "enable natural-language insights, retention message drafts, and the copilot. "
                "Everything else in Crystal Ball works without it.]")

    try:
        if provider == "openrouter":
            resp = requests.post(
                OPENROUTER_API_URL,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "X-Title": "Crystal Ball Churn Analytics",
                },
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_message},
                    ],
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices") or []
            if choices:
                return choices[0].get("message", {}).get("content", "").strip()
            return "[LLM returned no content]"
        else:
            resp = requests.post(
                ANTHROPIC_API_URL,
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "system": system,
                    "messages": [{"role": "user", "content": user_message}],
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            text_blocks = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
            return "\n".join(text_blocks).strip() or "[LLM returned no text content]"
    except requests.exceptions.RequestException as e:
        return f"[LLM call failed: {e}]"


# ---------------------------------------------------------- explain a customer ----
def explain_customer_risk(customer_id: str, risk: float, top_drivers: list[dict],
                           recommended_action: str | None, expected_save: float | None) -> str:
    system = (
        "You are a churn-analytics copilot writing for a Customer Success manager who has "
        "60 seconds before a call. Be direct and concrete. No hedging, no restating the "
        "numbers back verbatim -- interpret them. 3-4 sentences max."
    )
    drivers_str = "; ".join(f"{d['feature']} (impact {d['shap_value']:+.3f})" for d in top_drivers)
    action_str = (f"The system recommends: {recommended_action}, with an estimated ₹{expected_save:,.0f} save."
                  if recommended_action else "No specific action has been scored for this customer yet.")
    user_message = (
        f"Customer {customer_id} has a predicted churn risk of {risk:.1%}.\n"
        f"Top SHAP drivers: {drivers_str}\n"
        f"{action_str}\n\n"
        "In plain English: why is this customer at risk, and what should the CS manager "
        "actually say or do on the call?"
    )
    return _call_claude(system, user_message, max_tokens=300)


# ---------------------------------------------------------- draft a retention message ----
def draft_retention_message(customer_id: str, segment: str, action: str, tone: str = "warm and professional") -> str:
    system = (
        f"You write short, {tone} customer retention emails. No corporate filler, no "
        "over-apologizing, no fake urgency. Sound like a person who actually knows this "
        "account, not a mail-merge template. Keep it under 120 words. Output only the "
        "email body (no subject line, no signature block)."
    )
    user_message = (
        f"Draft a retention email for customer {customer_id}, a {segment.replace('_', ' ')} account. "
        f"The offer being extended is: {action}."
    )
    return _call_claude(system, user_message, max_tokens=300)


# ---------------------------------------------------------- freeform copilot ----
def ask_copilot(question: str, context: dict) -> str:
    system = (
        "You are Crystal Ball's data copilot. You answer questions about churn risk, revenue "
        "forecasts, and retention economics using ONLY the JSON context provided below -- never "
        "invent numbers that aren't in it. If the context doesn't contain what's needed to answer, "
        "say so plainly rather than guessing. Be concise: a few sentences or a short list, not an essay."
    )
    user_message = (
        f"Context (current model outputs and aggregates):\n{json.dumps(context, indent=2, default=str)}\n\n"
        f"Question: {question}"
    )
    return _call_claude(system, user_message, max_tokens=500)


# ---------------------------------------------------------- competitive briefing ----
def competitive_briefing(company_name: str, gap_analysis: dict, pricing_position: list,
                          priority_ranking: list | None = None) -> str:
    system = (
        "You are a competitive-strategy analyst. Use ONLY the structured facts given -- don't "
        "invent numbers or claims not in the data. Be direct about what's a real gap vs. what's "
        "a modeling limitation (e.g. a priority_score of 0 means our own model has no signal on "
        "that aspect, not that the aspect doesn't matter). Lead with the highest priority_score "
        "item if given -- that's grounded in our own trained churn model's feature importance, "
        "not a guess. 4-6 sentences."
    )
    priority_str = (f"\nPriority-ranked gaps (combines competitor sentiment with OUR OWN trained "
                     f"churn model's real feature importance): {json.dumps(priority_ranking[:4], indent=2)}"
                     if priority_ranking else "")
    user_message = (
        f"Company: {company_name}\n"
        f"Pricing position (cheapest to most expensive): {json.dumps(pricing_position, indent=2)}\n"
        f"Feature gap analysis: {json.dumps(gap_analysis, indent=2)}"
        f"{priority_str}\n\n"
        "Write a short competitive briefing: where does this company stand, what are competitors "
        "doing that it isn't, and what's the single highest-leverage gap to close?"
    )
    return _call_claude(system, user_message, max_tokens=400)


if __name__ == "__main__":
    # smoke test -- works with or without a real key
    print(explain_customer_risk("C000123", 0.14,
                                 [{"feature": "contract_month-to-month", "shap_value": 0.6},
                                  {"feature": "ticket_velocity", "shap_value": 0.3}],
                                 "10% discount offer", 1200.0))