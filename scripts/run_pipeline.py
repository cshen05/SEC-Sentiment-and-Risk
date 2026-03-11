

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"

DEFAULT_TICKERS = ["AAPL", "MSFT"]
DEFAULT_FORMS = ["10-K", "10-Q"]
DEFAULT_START_DATE = "2022-01-01"
DEFAULT_END_DATE = "2024-12-31"
DEFAULT_MAX_FILINGS = 5


class PipelineStepError(RuntimeError):
    """Raised when a pipeline step fails."""



def run_module(module_name: str, args: list[str] | None = None) -> None:
    """Run a Python module as a subprocess and stream its output."""
    if args is None:
        args = []

    command = [sys.executable, "-m", module_name] + args
    print(f"\n[INFO] Running: {' '.join(command)}")

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise PipelineStepError(
            f"Step failed: {module_name} (exit code {result.returncode})"
        )



def write_run_log(mode: str, status: str, started_at: datetime, finished_at: datetime) -> None:
    """Append a simple run log entry for pipeline observability."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "run_pipeline.log"
    duration_seconds = round((finished_at - started_at).total_seconds(), 2)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(
            f"mode={mode} | status={status} | "
            f"started_at={started_at.isoformat()} | "
            f"finished_at={finished_at.isoformat()} | "
            f"duration_seconds={duration_seconds}\n"
        )



def run_full_pipeline(args: argparse.Namespace) -> None:
    """Run the full end-to-end SEC risk pipeline."""
    run_module(
        "pipeline.build_corpus",
        args.tickers
        + [
            "--forms",
            *args.forms,
            "--start-date",
            args.start_date,
            "--end-date",
            args.end_date,
            "--max-filings",
            str(args.max_filings),
            "--output",
            "data/sec_risk_corpus.json",
        ],
    )
    run_module("pipeline.load_corpus_to_db")
    run_module("training.predict_full_corpus")
    run_module("training.summarize_predictions")
    run_module("training.build_portfolio_signals")
    run_module("training.build_risk_monitoring_report")



def run_report_pipeline() -> None:
    """Rebuild downstream reports from existing predictions."""
    run_module("training.summarize_predictions")
    run_module("training.build_portfolio_signals")
    run_module("training.build_risk_monitoring_report")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SEC sentiment and risk pipeline from a single command."
    )
    parser.add_argument(
        "--mode",
        choices=["full", "reports"],
        default="full",
        help="Pipeline mode to run.",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help="List of ticker symbols to ingest when running in full mode.",
    )
    parser.add_argument(
        "--forms",
        nargs="+",
        default=DEFAULT_FORMS,
        help="List of SEC forms to request when running in full mode.",
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help="Start date for SEC filing retrieval in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        default=DEFAULT_END_DATE,
        help="End date for SEC filing retrieval in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--max-filings",
        type=int,
        default=DEFAULT_MAX_FILINGS,
        help="Maximum number of filings per ticker/form combination in full mode.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    started_at = datetime.utcnow()

    print("\n[INFO] Starting SEC sentiment and risk pipeline")
    print(f"[INFO] Mode: {args.mode}")
    print(f"[INFO] Start time (UTC): {started_at.isoformat()}")

    try:
        if args.mode == "full":
            run_full_pipeline(args)
        elif args.mode == "reports":
            run_report_pipeline()
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")

        finished_at = datetime.utcnow()
        print("\n[INFO] Pipeline completed successfully")
        print(f"[INFO] Finish time (UTC): {finished_at.isoformat()}")
        write_run_log(args.mode, "success", started_at, finished_at)

    except Exception:
        finished_at = datetime.utcnow()
        print("\n[ERROR] Pipeline failed")
        print(f"[ERROR] Failure time (UTC): {finished_at.isoformat()}")
        write_run_log(args.mode, "failure", started_at, finished_at)
        raise


if __name__ == "__main__":
    main()