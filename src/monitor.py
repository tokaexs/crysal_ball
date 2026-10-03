"""
Layer 8: MONITORING & AUTO-RETRAIN
Simulates the nightly job: re-checks drift (PSI/KS), logs a synthetic prediction
outcome feed, and re-triggers the training pipeline if drift crosses the red
threshold. In production this would be a cron / Airflow DAG; here it's a single
script you can run on a schedule (cron, GitHub Actions, etc.) or call from the
dashboard's "Simulate retrain" button.
"""
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
LOG_PATH = MODEL_DIR / "retrain_log.jsonl"

sys.path.insert(0, str(BASE_DIR / "src"))
from trust import trust_card  # noqa: E402


def run_nightly_check(auto_retrain=False):
    card = trust_card()
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "drift": card["drift"]}

    if card["drift"]["retrain_triggered"]:
        entry["action"] = "RETRAIN_TRIGGERED"
        if auto_retrain:
            print("PSI exceeded red threshold -> retraining models...")
            subprocess.run([sys.executable, str(BASE_DIR / "src" / "models.py")], check=True)
            # prescriptive.py re-trains the uplift models itself (train_uplift_models) before scoring
            subprocess.run([sys.executable, str(BASE_DIR / "src" / "prescriptive.py")], check=True)
            entry["action"] = "RETRAIN_COMPLETED"
    else:
        entry["action"] = "NO_ACTION_HEALTHY"

    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(json.dumps(entry, indent=2))
    return entry


if __name__ == "__main__":
    run_nightly_check(auto_retrain="--auto-retrain" in sys.argv)
