"""
Layer 9b: SENTIMENT ANALYSIS (VADER)
Runs lexicon-based sentiment analysis on curated customer feedback statements
per competitor across key customer-experience aspects.
"""
from __future__ import annotations

import json
from pathlib import Path

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SENTIMENT_PATH = DATA_DIR / "sentiment_sources.json"


def analyze_sentiment() -> dict:
    if not SENTIMENT_PATH.exists():
        return {}

    with open(SENTIMENT_PATH) as f:
        data = json.load(f)

    analyzer = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer is not None else None
    results = {}

    for comp_key, statements in data["companies"].items():
        aspect_scores = {}
        compounds = []
        for s in statements:
            aspect = s["aspect"]
            if analyzer is not None:
                vs = analyzer.polarity_scores(s["text"])
            else:
                vs = {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
            compounds.append(vs["compound"])
            aspect_scores[aspect] = {
                "compound": round(vs["compound"], 3),
                "pos": vs["pos"],
                "neg": vs["neg"],
                "neu": vs["neu"],
                "text_summary": s["text"],
            }

        avg_compound = round(sum(compounds) / len(compounds), 3) if compounds else 0.0
        results[comp_key] = {
            "company": comp_key,
            "overall_compound": avg_compound,
            "n_statements": len(statements),
            "confidence_note": "Sampled qualitative statements",
            "aspects": aspect_scores,
        }

    return results


if __name__ == "__main__":
    res = analyze_sentiment()
    print(json.dumps(res, indent=2))
