"""
Per-state training runner.

Trains all 5 model types for every state in the dataset (43 states),
selects the best model (lowest CV MAPE) as champion for each state,
then re-fits the champion on the full state dataset for production.

Usage
-----
# Sequential (safe, ~2–3 h total):
    python scripts/train_all_states.py

# Parallel (faster, uses N workers):
    python scripts/train_all_states.py --workers 4

# Single state (quick smoke-test):
    python scripts/train_all_states.py --state California

# Specific model types only:
    python scripts/train_all_states.py --models sarima prophet xgboost

# Skip cross-validation (much faster, worse model selection):
    python scripts/train_all_states.py --skip-cv

Output
------
- Model files saved to models/  (one per model type × state × version)
- Registry updated at models/registry.json with champion flags
- Progress log written to logs/train_states_<timestamp>.log
- Summary table printed to stdout when complete

Background use
--------------
Windows PowerShell:
    Start-Job { python scripts/train_all_states.py } | Out-Null
    Receive-Job -Id <id> -Keep

    # or with nohup-equivalent:
    python scripts/train_all_states.py > logs/train.log 2>&1

Linux/macOS:
    nohup python scripts/train_all_states.py > logs/train.log 2>&1 &
    tail -f logs/train.log
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor as ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402

from src.pipeline.train import run_training  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_PATH    = str(ROOT / "data.csv")
CONFIG_PATH  = str(ROOT / "config" / "training_config.yaml")
OUTPUT_DIR   = str(ROOT / "models")
LOG_DIR      = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


# ── State discovery ───────────────────────────────────────────────────────────
def discover_states(data_path: str) -> list[str]:
    df = pd.read_csv(data_path)
    col = next((c for c in df.columns if c.strip().lower() == "state"), df.columns[0])
    states = sorted(df[col].dropna().str.strip().unique().tolist())
    logger.info(f"Discovered {len(states)} states from '{col}' column")
    return states


# ── Single-state worker (called in subprocess for parallel mode) ──────────────
def train_one_state(
    state: str,
    models_to_run: list[str] | None,
    cv_splits: int,
    skip_cv: bool,
) -> dict:
    """Train all models for one state, return result dict."""
    t0 = time.perf_counter()
    try:
        result = run_training(
            data_path=DATA_PATH,
            config_path=CONFIG_PATH,
            models_to_run=models_to_run,
            state_filter=state,
            output_dir=OUTPUT_DIR,
            horizon=8,
            cv_splits=cv_splits,
            skip_cv=skip_cv,
        )
        elapsed = round(time.perf_counter() - t0, 1)
        return {
            "state":    state,
            "status":   "ok",
            "champion": result["champion"],
            "version":  result["version"],
            "cv_mape":  result["cv_results"].get(result["champion"], {}).get("mape"),
            "test_mape": result["cv_results"]
            .get(result["champion"], {})
            .get("test_mape"),
            "elapsed":  elapsed,
        }
    except Exception as exc:
        return {
            "state":   state,
            "status":  "error",
            "error":   str(exc),
            "elapsed": round(time.perf_counter() - t0, 1),
        }


# ── Main orchestrator ─────────────────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"train_states_{ts}.log"

    logger.add(str(log_path), level="INFO", rotation="50 MB")
    logger.info("=" * 60)
    logger.info("Per-state training run started")
    logger.info(f"  Timestamp  : {ts}")
    logger.info(f"  Data       : {DATA_PATH}")
    logger.info(f"  Config     : {CONFIG_PATH}")
    logger.info(f"  Models     : {args.models or 'all'}")
    logger.info(f"  CV splits  : {args.cv_splits}")
    logger.info(f"  Skip CV    : {args.skip_cv}")
    logger.info(f"  Workers    : {args.workers}")
    logger.info(f"  Log        : {log_path}")
    logger.info("=" * 60)

    # Determine states to train
    if args.state:
        states = [args.state.strip()]
    elif args.states:
        states = [s.strip() for s in args.states]
    else:
        states = discover_states(DATA_PATH)

    total = len(states)
    results: list[dict] = []

    if args.workers == 1:
        # ── Sequential ────────────────────────────────────────────────────────
        for i, state in enumerate(states, 1):
            logger.info(f"[{i:>3}/{total}] Training {state} ...")
            r = train_one_state(state, args.models, args.cv_splits, args.skip_cv)
            results.append(r)
            if r["status"] == "ok":
                logger.info(
                    f"[{i:>3}/{total}] {state:20s}  champion={r['champion']:10s}"
                    f"  CV MAPE={r['cv_mape']:.3f}%"
                    f"  ({r['elapsed']:.0f}s)"
                )
            else:
                logger.error(f"[{i:>3}/{total}] {state}: FAILED — {r['error']}")
            _print_progress(i, total, results)
    else:
        # ── Parallel ──────────────────────────────────────────────────────────
        logger.info(f"Running {args.workers} workers in parallel")
        futures = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for state in states:
                f = pool.submit(
                    train_one_state, state, args.models, args.cv_splits, args.skip_cv
                )
                futures[f] = state

            completed = 0
            for future in as_completed(futures):
                completed += 1
                r = future.result()
                results.append(r)
                if r["status"] == "ok":
                    logger.info(
                        f"[{completed:>3}/{total}] {r['state']:20s}"
                        f"  champion={r['champion']:10s}"
                        f"  CV MAPE={r['cv_mape']:.3f}%"
                        f"  ({r['elapsed']:.0f}s)"
                    )
                else:
                    logger.error(
                        f"[{completed:>3}/{total}] {r['state']}: FAILED — {r['error']}"
                    )
                _print_progress(completed, total, results)

    # ── Summary ───────────────────────────────────────────────────────────────
    ok      = [r for r in results if r["status"] == "ok"]
    failed  = [r for r in results if r["status"] == "error"]
    total_t = sum(r["elapsed"] for r in results)

    logger.info("=" * 60)
    logger.info(f"Training complete  —  {len(ok)}/{total} states succeeded")
    logger.info(f"Total wall time: {total_t/60:.1f} min")
    if failed:
        logger.warning(f"Failed states: {[r['state'] for r in failed]}")

    # Print leaderboard
    print("\n" + "=" * 72)
    print(
        f"{'STATE':<22} {'CHAMPION':<12} "
        f"{'CV MAPE':>9} {'TEST MAPE':>10} {'TIME':>7}"
    )
    print("-" * 72)
    for r in sorted(ok, key=lambda x: x.get("cv_mape") or 999):
        cv   = f"{r['cv_mape']:.3f}%" if r.get("cv_mape") is not None else "  —"
        test = f"{r['test_mape']:.3f}%" if r.get("test_mape") is not None else "  —"
        print(
            f"{r['state']:<22} {r['champion']:<12} "
            f"{cv:>9} {test:>10} {r['elapsed']:>6.0f}s"
        )
    if failed:
        print("-" * 72)
        for r in failed:
            print(f"{r['state']:<22} FAILED: {r['error'][:40]}")
    print("=" * 72)

    # Save summary JSON
    summary_path = LOG_DIR / f"summary_{ts}.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    sys.exit(0 if not failed else 1)


def _print_progress(done: int, total: int, results: list[dict]) -> None:
    ok  = sum(1 for r in results if r["status"] == "ok")
    err = sum(1 for r in results if r["status"] == "error")
    pct = done / total * 100
    bar_len = 30
    filled  = int(bar_len * done / total)
    bar     = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  [{bar}] {pct:5.1f}%  {ok} ok, {err} failed", end="", flush=True)
    if done == total:
        print()


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train all 5 models for every state and select state champions."
    )
    p.add_argument("--state", default=None, help="Train a single state only")
    p.add_argument(
        "--states", nargs="+", default=None,
        help="Train specific states only (space-separated)",
    )
    p.add_argument(
        "--models", nargs="+", help="Model types to train (default: all)"
    )
    p.add_argument(
        "--cv-splits", type=int, default=3, help="CV folds per model (default 3)"
    )
    p.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip CV — faster but no model selection",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Parallel workers. Use 1 for sequential (safest). "
            "Higher = faster but more RAM. Recommended max: CPU cores / 2"
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())
