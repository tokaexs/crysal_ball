"""
Runs the entire Crystal Ball pipeline end to end, in order:
  1. Generate synthetic data + warehouse
  2. Feature engineering + leakage guard   (folded into models.py, run separately for report)
  3. Train churn + revenue models
  4. Score customers + rank prescriptive actions
  5. Compute trust card (backtests + drift)

Usage:
    python3 run_pipeline.py
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
STEPS = ["generate_data.py", "models.py", "prescriptive.py", "trust.py"]
# note: prescriptive.py trains the uplift models itself before scoring, so a separate
# uplift.py step isn't needed here -- run it standalone only if you want its diagnostics.


def main():
    for step in STEPS:
        print(f"\n{'=' * 60}\n>>> Running {step}\n{'=' * 60}")
        result = subprocess.run([sys.executable, str(SRC / step)])
        if result.returncode != 0:
            print(f"Step {step} failed, stopping pipeline.")
            sys.exit(1)
    print("\nPipeline complete. Start the API with:\n"
          "  uvicorn api.main:app --reload\n"
          "Start the dashboard with:\n"
          "  streamlit run dashboard/app.py\n"
          "\n"
          "To enable the LLM layer (natural-language explanations, retention drafts, the "
          "copilot), set your API key first:\n"
          "  export ANTHROPIC_API_KEY=sk-ant-...")


if __name__ == "__main__":
    main()
