#!/usr/bin/env python3
"""
WandB Training Monitor

Monitor WandB for training run status with stall detection.
Can be used standalone or imported by training_controller.py.

Usage:
    # Check training status
    python wandb_monitor.py --run-id bs4-qwen3-8b

    # Watch mode (continuous monitoring)
    python wandb_monitor.py --run-id bs4-qwen3-8b --watch

    # JSON output for scripting
    python wandb_monitor.py --run-id bs4-qwen3-8b --json

Environment variables:
    WANDB_API_KEY - WandB API key
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TrainingStatus:
    """Training run status."""

    HEALTHY = "healthy"
    FAILED = "failed"
    STALLED = "stalled"
    FINISHED = "finished"
    NOT_FOUND = "not_found"
    UNKNOWN = "unknown"
    ERROR = "error"


def get_wandb_api():
    """Get authenticated WandB API client."""
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        logger.error("WANDB_API_KEY environment variable not set")
        sys.exit(1)

    try:
        import wandb

        return wandb.Api()
    except ImportError:
        logger.error("wandb package not installed. Run: pip install wandb")
        sys.exit(1)


def find_runs(api, entity: str, project: str, run_id: str) -> tuple:
    """
    Find trainer and orchestrator runs for a given run_id.

    Returns (trainer_run, orchestrator_run) tuple.
    """
    try:
        runs = api.runs(
            f"{entity}/{project}",
            filters={"display_name": {"$regex": f"^{run_id}"}},
            order="-created_at",
        )

        trainer_run = None
        orchestrator_run = None

        for run in runs:
            if run.name.endswith("-trainer") and trainer_run is None:
                trainer_run = run
            elif run.name.endswith("-orchestrator") and orchestrator_run is None:
                orchestrator_run = run

            if trainer_run and orchestrator_run:
                break

        return trainer_run, orchestrator_run

    except Exception as e:
        logger.error(f"Error finding runs: {e}")
        return None, None


def check_run_health(run, stale_threshold_minutes: int = 20) -> dict:
    """
    Check health of a single WandB run.

    Returns dict with status info.
    """
    if run is None:
        return {
            "status": TrainingStatus.NOT_FOUND,
            "run_state": None,
            "last_step": None,
            "last_step_time": None,
            "message": "Run not found",
        }

    run_state = run.state

    # Check for terminal states
    if run_state in ["failed", "crashed"]:
        return {
            "status": TrainingStatus.FAILED,
            "run_state": run_state,
            "last_step": run.lastHistoryStep,
            "last_step_time": None,
            "message": f"Run {run_state} at step {run.lastHistoryStep}",
        }

    if run_state == "finished":
        return {
            "status": TrainingStatus.FINISHED,
            "run_state": run_state,
            "last_step": run.lastHistoryStep,
            "last_step_time": None,
            "message": f"Run finished at step {run.lastHistoryStep}",
        }

    if run_state != "running":
        return {
            "status": TrainingStatus.UNKNOWN,
            "run_state": run_state,
            "last_step": run.lastHistoryStep,
            "last_step_time": None,
            "message": f"Unknown state: {run_state}",
        }

    # Check for stall (running but no progress)
    # NOTE: run.history(samples=N) returns evenly-spaced samples across ALL history,
    # NOT the last N entries. Use a high sample count to get recent data with better
    # resolution. The max() will find the actual latest timestamp.
    try:
        # Use high sample count for better resolution on recent data
        history = run.history(samples=500)
        if not history.empty and "_timestamp" in history.columns:
            last_timestamp = history["_timestamp"].max()
            if last_timestamp:
                last_step_time = datetime.fromtimestamp(last_timestamp, tz=UTC)
                stale_threshold = datetime.now(UTC) - timedelta(minutes=stale_threshold_minutes)

                if last_step_time < stale_threshold:
                    minutes_stale = int((datetime.now(UTC) - last_step_time).total_seconds() / 60)
                    return {
                        "status": TrainingStatus.STALLED,
                        "run_state": run_state,
                        "last_step": run.lastHistoryStep,
                        "last_step_time": last_step_time.isoformat(),
                        "message": f"Stalled for {minutes_stale} minutes at step {run.lastHistoryStep}",
                    }

                return {
                    "status": TrainingStatus.HEALTHY,
                    "run_state": run_state,
                    "last_step": run.lastHistoryStep,
                    "last_step_time": last_step_time.isoformat(),
                    "message": f"Healthy at step {run.lastHistoryStep}",
                }
    except Exception as e:
        logger.warning(f"Could not check history: {e}")

    return {
        "status": TrainingStatus.HEALTHY,
        "run_state": run_state,
        "last_step": run.lastHistoryStep,
        "last_step_time": None,
        "message": f"Healthy at step {run.lastHistoryStep} (no timestamp available)",
    }


def get_metrics(run, samples: int = 10) -> dict:
    """Get recent metrics from a run."""
    try:
        history = run.history(samples=samples)
        if history.empty:
            return {}

        metrics = {}

        # Get latest values for key metrics
        metric_names = [
            "reward/mean",
            "entropy/mean",
            "mismatch_kl/mean",
            "optim/grad_norm",
            "time/step",
            "perf/throughput",
        ]

        for name in metric_names:
            if name in history.columns:
                values = history[name].dropna()
                if len(values) > 0:
                    metrics[name] = {
                        "latest": float(values.iloc[-1]),
                        "mean": float(values.mean()),
                        "min": float(values.min()),
                        "max": float(values.max()),
                    }

        return metrics

    except Exception as e:
        logger.warning(f"Could not get metrics: {e}")
        return {}


def check_training_status(
    entity: str,
    project: str,
    run_id: str,
    stale_threshold_minutes: int = 20,
    include_metrics: bool = False,
) -> dict:
    """
    Check overall training status.

    Returns combined status from trainer and orchestrator runs.
    """
    api = get_wandb_api()
    trainer_run, orchestrator_run = find_runs(api, entity, project, run_id)

    trainer_status = check_run_health(trainer_run, stale_threshold_minutes)
    orchestrator_status = check_run_health(orchestrator_run, stale_threshold_minutes)

    # Combine statuses (worst case wins)
    priority = [
        TrainingStatus.ERROR,
        TrainingStatus.FAILED,
        TrainingStatus.STALLED,
        TrainingStatus.NOT_FOUND,
        TrainingStatus.UNKNOWN,
        TrainingStatus.HEALTHY,
        TrainingStatus.FINISHED,
    ]

    combined_status = TrainingStatus.HEALTHY

    for status in [trainer_status["status"], orchestrator_status["status"]]:
        if priority.index(status) < priority.index(combined_status):
            combined_status = status

    result = {
        "status": combined_status,
        "trainer": trainer_status,
        "orchestrator": orchestrator_status,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    # Add metrics if requested
    if include_metrics:
        result["metrics"] = {}
        if trainer_run:
            result["metrics"]["trainer"] = get_metrics(trainer_run)
        if orchestrator_run:
            result["metrics"]["orchestrator"] = get_metrics(orchestrator_run)

    return result


def format_status(status: dict) -> str:
    """Format status dict for human-readable output."""
    lines = []

    # Overall status
    emoji = {
        TrainingStatus.HEALTHY: "✅",
        TrainingStatus.FAILED: "❌",
        TrainingStatus.STALLED: "⚠️",
        TrainingStatus.FINISHED: "🎉",
        TrainingStatus.NOT_FOUND: "❓",
        TrainingStatus.UNKNOWN: "❓",
        TrainingStatus.ERROR: "💥",
    }.get(status["status"], "❓")

    lines.append(f"\n{emoji} Training Status: {status['status'].upper()}")
    lines.append(f"   Timestamp: {status['timestamp']}")

    # Trainer status
    t = status["trainer"]
    lines.append(f"\n   Trainer:      {t['status']} - {t['message']}")

    # Orchestrator status
    o = status["orchestrator"]
    lines.append(f"   Orchestrator: {o['status']} - {o['message']}")

    # Metrics if available
    if "metrics" in status and status["metrics"]:
        lines.append("\n   Recent Metrics:")
        for run_type, metrics in status["metrics"].items():
            if metrics:
                lines.append(f"     {run_type}:")
                for name, values in metrics.items():
                    lines.append(f"       {name}: {values['latest']:.4f}")

    return "\n".join(lines)


def watch_training(
    entity: str, project: str, run_id: str, interval: int = 60, stale_threshold: int = 20
):
    """Continuously monitor training status."""
    logger.info(f"Watching training run: {run_id}")
    logger.info(f"Check interval: {interval}s, Stale threshold: {stale_threshold}min")

    try:
        while True:
            status = check_training_status(
                entity,
                project,
                run_id,
                stale_threshold_minutes=stale_threshold,
                include_metrics=True,
            )

            print(format_status(status))

            if status["status"] in [TrainingStatus.FINISHED, TrainingStatus.FAILED]:
                logger.info("Training ended, stopping watch")
                break

            time.sleep(interval)

    except KeyboardInterrupt:
        logger.info("Watch interrupted")


def main():
    parser = argparse.ArgumentParser(description="WandB Training Monitor")
    parser.add_argument("--run-id", required=True, help="Training run ID")
    parser.add_argument("--entity", default="seconds-0-domus-magna-inc", help="WandB entity")
    parser.add_argument("--project", default="beautiful-soup-env", help="WandB project")
    parser.add_argument(
        "--stale-threshold",
        type=int,
        default=20,
        help="Minutes without progress to consider stalled",
    )
    parser.add_argument("--watch", action="store_true", help="Continuous monitoring mode")
    parser.add_argument("--interval", type=int, default=60, help="Watch interval in seconds")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--metrics", action="store_true", help="Include recent metrics")

    args = parser.parse_args()

    if args.watch:
        watch_training(
            args.entity,
            args.project,
            args.run_id,
            interval=args.interval,
            stale_threshold=args.stale_threshold,
        )
    else:
        status = check_training_status(
            args.entity,
            args.project,
            args.run_id,
            stale_threshold_minutes=args.stale_threshold,
            include_metrics=args.metrics,
        )

        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(format_status(status))

        # Exit with non-zero code if unhealthy
        if status["status"] not in [TrainingStatus.HEALTHY, TrainingStatus.FINISHED]:
            sys.exit(1)


if __name__ == "__main__":
    main()
