#!/usr/bin/env python3
"""Check WandB training progress for the BeautifulSoup RL run."""
import sys

try:
    import wandb
except ImportError:
    print("Installing wandb...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "wandb"])
    import wandb

PROJECT = "seconds-0-domus-magna-inc/beautiful-soup-env"
TRAINER_RUN_ID = "r4rv5keq"
ORCHESTRATOR_RUN_ID = "yek034oe"


def check_run(run_id: str, name: str):
    """Check a single run's status and metrics."""
    api = wandb.Api()
    run = api.run(f"{PROJECT}/{run_id}")

    print(f"\n{'='*60}")
    print(f"{name}: {run.name}")
    print(f"{'='*60}")
    print(f"State: {run.state}")
    print(f"URL: {run.url}")

    # Get summary metrics
    summary = dict(run.summary) if run.summary else {}

    # Filter to useful metrics
    key_metrics = [
        "train/reward", "train/loss", "train/policy_loss", "train/kl",
        "rollout/mean_reward", "rollout/success_rate",
        "step", "_step", "_runtime"
    ]

    relevant = {k: v for k, v in summary.items()
                if any(m in k for m in key_metrics) and isinstance(v, (int, float))}

    if relevant:
        print("\nKey Metrics:")
        for key, value in sorted(relevant.items()):
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    else:
        print("\nNo metrics logged yet (training may still be initializing)")

    # Runtime
    if "_runtime" in summary:
        runtime_sec = summary["_runtime"]
        hours = int(runtime_sec // 3600)
        mins = int((runtime_sec % 3600) // 60)
        print(f"\nRuntime: {hours}h {mins}m")

    return run.state


def main():
    print("BeautifulSoup RL Training Monitor")
    print(f"Project: {PROJECT}")

    trainer_state = check_run(TRAINER_RUN_ID, "Trainer")
    orchestrator_state = check_run(ORCHESTRATOR_RUN_ID, "Orchestrator")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Trainer: {trainer_state}")
    print(f"Orchestrator: {orchestrator_state}")

    if trainer_state == "running" and orchestrator_state == "running":
        print("\n✅ Training is running normally")
    elif trainer_state == "finished" and orchestrator_state == "finished":
        print("\n🎉 Training completed!")
    else:
        print(f"\n⚠️  Check run status - may need attention")


if __name__ == "__main__":
    main()
