#!/usr/bin/env python3
"""
Training Controller for GitHub Actions

Monitors WandB for training status and auto-provisions Vast.ai instances
when training fails or stalls. Designed to be idempotent and safe to run
repeatedly.

Usage:
    python training_controller.py --run-id bs4-qwen3-8b --action check
    python training_controller.py --run-id bs4-qwen3-8b --action provision
    python training_controller.py --run-id bs4-qwen3-8b --action terminate

Environment variables:
    WANDB_API_KEY - WandB API key
    VAST_API_KEY - Vast.ai API key
    B2_APPLICATION_KEY_ID - Backblaze B2 key ID
    B2_APPLICATION_KEY - Backblaze B2 application key
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/training_controller.log')
    ]
)
logger = logging.getLogger(__name__)


def check_wandb_status(
    entity: str,
    project: str,
    run_id: str,
    stale_threshold_minutes: int = 20
) -> dict:
    """
    Check WandB for training run status.

    Returns:
        dict with keys:
            - status: 'healthy' | 'failed' | 'stalled' | 'not_found'
            - run_state: WandB run state
            - last_step: Last step number
            - last_step_time: Timestamp of last step
            - message: Human-readable status message
    """
    try:
        import wandb
        api = wandb.Api()

        # Find runs matching our run_id pattern
        # Prime-RL creates two runs: {name}-trainer and {name}-orchestrator
        runs = api.runs(
            f"{entity}/{project}",
            filters={"display_name": {"$regex": f"^{run_id}"}},
            order="-created_at"
        )

        trainer_run = None
        orchestrator_run = None

        for run in runs:
            if run.name.endswith('-trainer') and trainer_run is None:
                trainer_run = run
            elif run.name.endswith('-orchestrator') and orchestrator_run is None:
                orchestrator_run = run

            if trainer_run and orchestrator_run:
                break

        if not trainer_run and not orchestrator_run:
            return {
                'status': 'not_found',
                'run_state': None,
                'last_step': None,
                'last_step_time': None,
                'message': f'No runs found matching {run_id}'
            }

        # Check trainer run state (primary indicator)
        primary_run = trainer_run or orchestrator_run
        run_state = primary_run.state

        if run_state in ['failed', 'crashed']:
            return {
                'status': 'failed',
                'run_state': run_state,
                'last_step': primary_run.lastHistoryStep,
                'last_step_time': None,
                'message': f'Training {run_state} at step {primary_run.lastHistoryStep}'
            }

        if run_state == 'finished':
            return {
                'status': 'finished',
                'run_state': run_state,
                'last_step': primary_run.lastHistoryStep,
                'last_step_time': None,
                'message': f'Training finished at step {primary_run.lastHistoryStep}'
            }

        if run_state != 'running':
            return {
                'status': 'unknown',
                'run_state': run_state,
                'last_step': primary_run.lastHistoryStep,
                'last_step_time': None,
                'message': f'Unknown run state: {run_state}'
            }

        # Check for stalled runs (running but no progress)
        try:
            history = primary_run.history(samples=5)
            if not history.empty and '_timestamp' in history.columns:
                last_timestamp = history['_timestamp'].max()
                if last_timestamp:
                    last_step_time = datetime.fromtimestamp(last_timestamp, tz=timezone.utc)
                    stale_threshold = datetime.now(timezone.utc) - timedelta(minutes=stale_threshold_minutes)

                    if last_step_time < stale_threshold:
                        return {
                            'status': 'stalled',
                            'run_state': run_state,
                            'last_step': primary_run.lastHistoryStep,
                            'last_step_time': last_step_time.isoformat(),
                            'message': f'Training stalled - no progress for {stale_threshold_minutes} minutes'
                        }
        except Exception as e:
            logger.warning(f"Could not check history for stall detection: {e}")

        return {
            'status': 'healthy',
            'run_state': run_state,
            'last_step': primary_run.lastHistoryStep,
            'last_step_time': None,
            'message': f'Training healthy at step {primary_run.lastHistoryStep}'
        }

    except Exception as e:
        logger.error(f"Error checking WandB: {e}")
        return {
            'status': 'error',
            'run_state': None,
            'last_step': None,
            'last_step_time': None,
            'message': f'Error checking WandB: {e}'
        }


def check_existing_instances(run_id: str) -> list:
    """
    Check for existing Vast.ai instances with our label.

    Returns list of instance dicts.
    """
    try:
        # Import from our provision_vast module
        from provision_vast import list_instances, VastAPIError
        return list_instances(run_id)
    except (ImportError, VastAPIError) as e:
        if isinstance(e, ImportError):
            pass  # Fall through to fallback
        else:
            logger.warning(f"Vast API error: {e}, trying fallback")
    except Exception as e:
        logger.warning(f"Error using provision_vast: {e}, trying fallback")

    # Fallback to direct CLI call
    try:
        import subprocess
        import json

        api_key = os.environ.get('VAST_API_KEY')
        if not api_key:
            logger.error("VAST_API_KEY not set")
            return []

        # Set API key
        subprocess.run(['vastai', 'set', 'api-key', api_key],
                      capture_output=True, timeout=30)

        # List instances
        result = subprocess.run(['vastai', 'show', 'instances', '--raw'],
                               capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            if '403' in result.stderr:
                return []  # No instances
            logger.error(f"Error listing instances: {result.stderr}")
            return []

        try:
            instances = json.loads(result.stdout)
        except json.JSONDecodeError:
            return []

        # Filter by our label
        label = f"bs4-training-{run_id}"
        return [inst for inst in instances if (inst.get('label') or '').startswith(label)]

    except Exception as e:
        logger.error(f"Error checking Vast.ai instances: {e}")
        return []


def provision_instance(
    run_id: str,
    gpu_type: str,
    gpu_count: int,
    max_bid: float
) -> Optional[dict]:
    """
    Provision a new Vast.ai instance for training.

    Returns instance info dict or None on failure.
    """
    try:
        # Import from our provision_vast module
        from provision_vast import create_instance, VastAPIError
        return create_instance(
            run_id=run_id,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            max_price=max_bid
        )
    except (ImportError, VastAPIError) as e:
        if isinstance(e, VastAPIError):
            logger.warning(f"Vast API error: {e}, trying fallback")
        # Fallback to direct CLI call
        import subprocess
        import json

        api_key = os.environ.get('VAST_API_KEY')
        if not api_key:
            logger.error("VAST_API_KEY not set")
            return None

        # Set API key
        subprocess.run(['vastai', 'set', 'api-key', api_key],
                      capture_output=True, timeout=30)

        # Search for offers
        logger.info(f"Searching for {gpu_count}x {gpu_type} instances under ${max_bid}/hr")
        query = f"num_gpus>={gpu_count} gpu_name={gpu_type} rentable=true"

        result = subprocess.run([
            'vastai', 'search', 'offers', '--raw',
            '--limit', '10', '--order', 'dph_total', query
        ], capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            logger.error(f"Search failed: {result.stderr}")
            return None

        try:
            offers = json.loads(result.stdout)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON: {result.stdout[:200]}")
            return None

        if not offers:
            logger.error("No offers found")
            return None

        # Filter by price
        max_total = max_bid * gpu_count
        suitable = [o for o in offers if o.get('dph_total', float('inf')) <= max_total]

        if not suitable:
            logger.error(f"No offers under ${max_total:.2f}/hr")
            return None

        best = suitable[0]
        offer_id = best['id']
        logger.info(f"Selected offer {offer_id} at ${best['dph_total']:.2f}/hr")

        # Build create command
        onstart_url = "https://raw.githubusercontent.com/seconds-0/beautifulsoup-rl/main/scripts/onstart.sh"

        # Build env string with docker-style -e flags
        env_vars = [f'-e RUN_ID={run_id}', '-e B2_BUCKET=beautifulsoup-rl']
        for var in ['B2_APPLICATION_KEY_ID', 'B2_APPLICATION_KEY', 'WANDB_API_KEY']:
            if os.environ.get(var):
                env_vars.append(f'-e {var}={os.environ[var]}')
        env_string = ' '.join(env_vars)

        cmd = [
            'vastai', 'create', 'instance', str(offer_id),
            '--image', 'nvidia/cuda:12.1-devel-ubuntu22.04',
            '--disk', '100',
            '--label', f'bs4-training-{run_id}',
            '--onstart-cmd', f'curl -sSL {onstart_url} | bash',
            '--ssh',
            '--env', env_string,
            '--raw',
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            logger.error(f"Create failed: {result.stderr}")
            return None

        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            return {'status': 'created', 'raw': result.stdout}

    except Exception as e:
        logger.error(f"Error provisioning instance: {e}")
        return None


def terminate_instances(run_id: str) -> int:
    """
    Terminate all instances with our label.

    Returns number of instances terminated.
    """
    try:
        # Import from our provision_vast module
        from provision_vast import terminate_instances as vast_terminate, VastAPIError
        return vast_terminate(run_id, force=True)
    except (ImportError, VastAPIError) as e:
        if isinstance(e, VastAPIError):
            logger.warning(f"Vast API error: {e}, trying fallback")
        # Fallback to direct CLI call
        import subprocess

        instances = check_existing_instances(run_id)
        terminated = 0

        for inst in instances:
            inst_id = inst['id']
            logger.info(f"Terminating instance {inst_id}")

            result = subprocess.run(
                ['vastai', 'destroy', 'instance', str(inst_id)],
                capture_output=True, text=True, timeout=60
            )

            if result.returncode == 0:
                terminated += 1
            else:
                logger.error(f"Failed to terminate {inst_id}: {result.stderr}")

        return terminated

    except Exception as e:
        logger.error(f"Error terminating instances: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description='Training Controller')
    parser.add_argument('--run-id', required=True, help='Training run ID')
    parser.add_argument('--wandb-project', default='beautiful-soup-env', help='WandB project name')
    parser.add_argument('--wandb-entity', default='seconds-0-domus-magna-inc', help='WandB entity')
    parser.add_argument('--gpu-type', default='H100_PCIE', help='GPU type to request')
    parser.add_argument('--gpu-count', type=int, default=2, help='Number of GPUs')
    parser.add_argument('--max-bid', type=float, default=2.50, help='Max bid per GPU per hour')
    parser.add_argument('--action', choices=['check', 'provision', 'terminate'], default='check',
                        help='Action to perform')
    parser.add_argument('--stale-threshold', type=int, default=20,
                        help='Minutes without progress before considering stalled')

    args = parser.parse_args()

    logger.info(f"Training Controller starting - action: {args.action}, run_id: {args.run_id}")

    # Handle explicit actions
    if args.action == 'terminate':
        count = terminate_instances(args.run_id)
        logger.info(f"Terminated {count} instance(s)")
        return

    if args.action == 'provision':
        # Check for existing instances first
        existing = check_existing_instances(args.run_id)
        if existing:
            logger.info(f"Found {len(existing)} existing instance(s), skipping provision")
            for inst in existing:
                logger.info(f"  Instance {inst['id']}: {inst.get('actual_status', 'unknown')}")
            return

        instance = provision_instance(args.run_id, args.gpu_type, args.gpu_count, args.max_bid)
        if instance:
            logger.info("Successfully provisioned new instance")
        else:
            logger.error("Failed to provision instance")
            sys.exit(1)
        return

    # Default: check and auto-recover
    status = check_wandb_status(
        args.wandb_entity,
        args.wandb_project,
        args.run_id,
        args.stale_threshold
    )

    logger.info(f"WandB status: {status['status']} - {status['message']}")

    if status['status'] == 'healthy':
        logger.info("Training is healthy, no action needed")
        return

    if status['status'] == 'finished':
        logger.info("Training completed!")
        # Optionally terminate instances
        terminate_instances(args.run_id)
        return

    if status['status'] in ['failed', 'stalled', 'not_found']:
        logger.warning(f"Training needs recovery: {status['message']}")

        # Check for existing instances
        existing = check_existing_instances(args.run_id)
        running_instances = [i for i in existing if i.get('actual_status') == 'running']

        if running_instances:
            logger.info(f"Found {len(running_instances)} running instance(s), assuming recovery in progress")
            return

        # No running instances, provision new one
        logger.info("No running instances found, provisioning new instance")
        instance = provision_instance(args.run_id, args.gpu_type, args.gpu_count, args.max_bid)

        if instance:
            logger.info("Successfully provisioned recovery instance")
        else:
            logger.error("Failed to provision recovery instance")
            sys.exit(1)

    else:
        logger.warning(f"Unknown status: {status['status']}")


if __name__ == '__main__':
    main()
