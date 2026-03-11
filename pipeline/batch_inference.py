

import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "logs"


def run_script(script_module: str, args=None) -> None:
    """Run a pipeline module as a subprocess and stream results."""
    if args is None:
        args = []

    command = [sys.executable, "-m", script_module] + args
    print(f"\n[INFO] Running: {' '.join(command)}")

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(result.stdout)

    if result.returncode != 0:
        if result.stderr:
            print(result.stderr)
        raise RuntimeError(f"{script_module} failed with exit code {result.returncode}")

    if result.stderr:
        print(result.stderr)



def write_run_log(status: str, started_at: datetime, finished_at: datetime) -> None:
    """Persist a simple run log for batch job observability."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "batch_inference.log"

    duration_seconds = round((finished_at - started_at).total_seconds(), 2)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"status={status} | "
            f"started_at={started_at.isoformat()} | "
            f"finished_at={finished_at.isoformat()} | "
            f"duration_seconds={duration_seconds}\n"
        )



def main() -> None:
    started_at = datetime.utcnow()
    print("\n[INFO] Starting SEC risk batch inference pipeline")
    print(f"[INFO] Start time (UTC): {started_at.isoformat()}")

    try:
        # Step 1 — Refresh corpus from SEC filings.
        run_script(
            "pipeline.build_corpus",
            [
                "AAPL",
                "MSFT",
                "--forms",
                "10-K",
                "10-Q",
                "--start-date",
                "2023-01-01",
                "--end-date",
                "2024-12-31",
                "--max-filings",
                "3",
                "--output",
                "data/sec_risk_corpus.json",
            ],
        )

        # Step 2 — Load refreshed corpus into Postgres.
        run_script("pipeline.load_corpus_to_db")

        # Step 3 — Run full-corpus FinBERT inference.
        run_script("training.predict_full_corpus")

        # Step 4 — Build filing-level reports and top-risk extracts.
        run_script("training.summarize_predictions")

        finished_at = datetime.utcnow()
        print("\n[INFO] Batch inference pipeline completed successfully")
        print(f"[INFO] Finish time (UTC): {finished_at.isoformat()}")
        write_run_log("success", started_at, finished_at)

    except Exception:
        finished_at = datetime.utcnow()
        print("\n[ERROR] Batch inference pipeline failed")
        print(f"[ERROR] Failure time (UTC): {finished_at.isoformat()}")
        write_run_log("failure", started_at, finished_at)
        raise


if __name__ == "__main__":
    main()