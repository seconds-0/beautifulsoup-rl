#!/usr/bin/env python3
"""
Vast.ai Instance Provisioning Script

Provision GPU instances on Vast.ai for training. Uses the vastai CLI
(not Python API which doesn't have a VastAI class).

Usage:
    # Search for available instances
    python provision_vast.py search --gpu H100 --count 2 --max-price 2.50

    # Provision a new instance
    python provision_vast.py create --run-id bs4-qwen3-8b --gpu H100 --count 2

    # List our running instances
    python provision_vast.py list --run-id bs4-qwen3-8b

    # Terminate instances
    python provision_vast.py terminate --run-id bs4-qwen3-8b

Environment variables:
    VAST_API_KEY - Vast.ai API key (required)
    B2_APPLICATION_KEY_ID - Backblaze B2 key ID
    B2_APPLICATION_KEY - Backblaze B2 application key
    WANDB_API_KEY - WandB API key
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# GPU type mapping for Vast.ai query syntax
GPU_TYPES = {
    'H100': 'H100',
    'H100_PCIE': 'H100_PCIE',
    'H100_SXM': 'H100_SXM5',
    'A100': 'A100',
    'A100_80GB': 'A100_SXM4',
    '4090': 'RTX_4090',
    'RTX4090': 'RTX_4090',
}


class VastAPIError(Exception):
    """Error interacting with Vast.ai API."""
    pass


def setup_api_key():
    """Ensure API key is configured for vastai CLI.

    Raises:
        VastAPIError: If API key is not set or cannot be configured.
    """
    api_key = os.environ.get('VAST_API_KEY')
    if not api_key:
        raise VastAPIError("VAST_API_KEY environment variable not set")

    # Set API key via CLI
    try:
        result = subprocess.run(
            ['vastai', 'set', 'api-key', api_key],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            raise VastAPIError(f"Failed to set API key: {result.stderr}")
    except FileNotFoundError:
        raise VastAPIError("vastai CLI not installed. Run: pip install vastai")


def run_vastai_command(args: list, timeout: int = 60) -> tuple[int, str, str]:
    """Run a vastai CLI command and return (returncode, stdout, stderr).

    Raises:
        VastAPIError: If vastai CLI is not installed.
    """
    try:
        result = subprocess.run(
            ['vastai'] + args,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except FileNotFoundError:
        raise VastAPIError("vastai CLI not installed. Run: pip install vastai")


def search_offers(gpu_type: str, gpu_count: int, max_price: float, limit: int = 20) -> list:
    """Search for available GPU offers using vastai CLI."""
    setup_api_key()

    # Normalize GPU type
    gpu_name = GPU_TYPES.get(gpu_type.upper(), gpu_type)

    # Build query string
    # Vast.ai query syntax: field=value field>=value etc
    query = f"num_gpus>={gpu_count} gpu_name={gpu_name} rentable=true"

    logger.info(f"Searching for {gpu_count}x {gpu_name} under ${max_price}/GPU/hr")
    logger.info(f"Query: {query}")

    returncode, stdout, stderr = run_vastai_command([
        'search', 'offers',
        '--raw',
        '--limit', str(limit),
        '--order', 'dph_total',
        query
    ])

    if returncode != 0:
        logger.error(f"Search failed: {stderr}")
        return []

    try:
        offers = json.loads(stdout)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON response: {stdout[:200]}")
        return []

    if not offers:
        logger.warning("No offers found")
        return []

    # Filter by max price (per GPU)
    max_total = max_price * gpu_count
    filtered = [o for o in offers if o.get('dph_total', float('inf')) <= max_total]

    logger.info(f"Found {len(filtered)} offers under ${max_total:.2f}/hr total")

    return filtered


def list_instances(run_id: Optional[str] = None) -> list:
    """List instances, optionally filtered by run_id label."""
    setup_api_key()

    returncode, stdout, stderr = run_vastai_command([
        'show', 'instances', '--raw'
    ])

    if returncode != 0:
        # 403 means no instances (not an error)
        if '403' in stderr:
            return []
        logger.error(f"List failed: {stderr}")
        return []

    try:
        instances = json.loads(stdout)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON response: {stdout[:200]}")
        return []

    if run_id:
        label_prefix = f"bs4-training-{run_id}"
        instances = [i for i in instances if i.get('label', '').startswith(label_prefix)]

    return instances


def create_instance(
    run_id: str,
    gpu_type: str,
    gpu_count: int,
    max_price: float,
    docker_image: str = "nvidia/cuda:12.1-devel-ubuntu22.04",
    disk_gb: int = 100
) -> Optional[dict]:
    """Create a new Vast.ai instance for training."""
    setup_api_key()

    # Search for offers
    offers = search_offers(gpu_type, gpu_count, max_price)
    if not offers:
        logger.error("No suitable offers found")
        return None

    best_offer = offers[0]
    offer_id = best_offer['id']
    logger.info(f"Selected offer {offer_id}: "
                f"{best_offer.get('gpu_name', 'GPU')} x{best_offer.get('num_gpus', '?')} "
                f"at ${best_offer['dph_total']:.2f}/hr")

    # Build onstart script URL
    onstart_url = "https://raw.githubusercontent.com/seconds-0/beautifulsoup-rl/main/scripts/onstart.sh"
    onstart_cmd = f"curl -sSL {onstart_url} | bash"

    # Build environment variables string
    env_vars = {
        'RUN_ID': run_id,
        'B2_BUCKET': 'beautifulsoup-rl',
    }

    # Add secrets from environment
    for var in ['B2_APPLICATION_KEY_ID', 'B2_APPLICATION_KEY', 'WANDB_API_KEY']:
        if os.environ.get(var):
            env_vars[var] = os.environ[var]

    # Format as single --env string with docker-style -e flags
    env_string = ' '.join(f'-e {key}={value}' for key, value in env_vars.items())

    logger.info("Creating instance...")

    # Create instance using vastai CLI
    cmd = [
        'create', 'instance', str(offer_id),
        '--image', docker_image,
        '--disk', str(disk_gb),
        '--label', f'bs4-training-{run_id}',
        '--onstart-cmd', onstart_cmd,
        '--ssh',  # Enable SSH access
        '--env', env_string,
        '--raw'
    ]

    returncode, stdout, stderr = run_vastai_command(cmd, timeout=120)

    if returncode != 0:
        logger.error(f"Create failed: {stderr}")
        return None

    try:
        result = json.loads(stdout)
        logger.info(f"Instance created: {result}")
        return result
    except json.JSONDecodeError:
        # Sometimes vastai returns just success message
        logger.info(f"Instance created (non-JSON response): {stdout}")
        return {'status': 'created', 'raw': stdout}


def terminate_instances(run_id: str, force: bool = False) -> int:
    """Terminate instances matching run_id."""
    setup_api_key()

    instances = list_instances(run_id)
    if not instances:
        logger.info("No instances found to terminate")
        return 0

    terminated = 0
    for inst in instances:
        inst_id = inst['id']
        status = inst.get('actual_status', 'unknown')

        if not force and status == 'running':
            logger.warning(f"Instance {inst_id} is running, use --force to terminate")
            continue

        logger.info(f"Terminating instance {inst_id} (status: {status})")

        returncode, stdout, stderr = run_vastai_command([
            'destroy', 'instance', str(inst_id)
        ])

        if returncode == 0:
            terminated += 1
        else:
            logger.error(f"Failed to terminate {inst_id}: {stderr}")

    return terminated


def print_offers(offers: list):
    """Pretty-print offer list."""
    if not offers:
        print("No offers found")
        return

    print(f"\n{'ID':<10} {'GPU':<15} {'Count':<6} {'$/hr':<8} {'VRAM':<8} {'Location':<15}")
    print("-" * 70)

    for o in offers:
        print(f"{o.get('id', 'N/A'):<10} "
              f"{o.get('gpu_name', 'N/A'):<15} "
              f"{o.get('num_gpus', '?'):<6} "
              f"${o.get('dph_total', 0):<7.2f} "
              f"{o.get('gpu_ram', 0):<8} "
              f"{o.get('geolocation', 'N/A'):<15}")


def print_instances(instances: list):
    """Pretty-print instance list."""
    if not instances:
        print("No instances found")
        return

    print(f"\n{'ID':<10} {'Label':<30} {'Status':<12} {'GPU':<15} {'$/hr':<8}")
    print("-" * 80)

    for i in instances:
        print(f"{i.get('id', 'N/A'):<10} "
              f"{i.get('label', 'N/A')[:30]:<30} "
              f"{i.get('actual_status', 'N/A'):<12} "
              f"{i.get('gpu_name', 'N/A'):<15} "
              f"${i.get('dph_total', 0):<7.2f}")


def main():
    parser = argparse.ArgumentParser(description='Vast.ai Instance Provisioning')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Search command
    search_parser = subparsers.add_parser('search', help='Search for available offers')
    search_parser.add_argument('--gpu', default='H100', help='GPU type (e.g., H100, A100, 4090)')
    search_parser.add_argument('--count', type=int, default=2, help='Number of GPUs')
    search_parser.add_argument('--max-price', type=float, default=2.50, help='Max price per GPU per hour')
    search_parser.add_argument('--limit', type=int, default=20, help='Max results to show')

    # Create command
    create_parser = subparsers.add_parser('create', help='Create a new instance')
    create_parser.add_argument('--run-id', required=True, help='Training run ID')
    create_parser.add_argument('--gpu', default='H100', help='GPU type')
    create_parser.add_argument('--count', type=int, default=2, help='Number of GPUs')
    create_parser.add_argument('--max-price', type=float, default=2.50, help='Max price per GPU per hour')
    create_parser.add_argument('--disk', type=int, default=100, help='Disk size in GB')

    # List command
    list_parser = subparsers.add_parser('list', help='List instances')
    list_parser.add_argument('--run-id', help='Filter by run ID')

    # Terminate command
    term_parser = subparsers.add_parser('terminate', help='Terminate instances')
    term_parser.add_argument('--run-id', required=True, help='Run ID to terminate')
    term_parser.add_argument('--force', action='store_true', help='Force terminate running instances')

    args = parser.parse_args()

    if args.command == 'search':
        offers = search_offers(args.gpu, args.count, args.max_price, args.limit)
        print_offers(offers)

    elif args.command == 'create':
        instance = create_instance(
            run_id=args.run_id,
            gpu_type=args.gpu,
            gpu_count=args.count,
            max_price=args.max_price,
            disk_gb=args.disk
        )
        if instance:
            print(f"\nInstance created successfully!")
            if isinstance(instance, dict) and 'new_contract' in instance:
                print(f"  ID: {instance['new_contract']}")
                print(f"  SSH: vastai ssh {instance['new_contract']}")
            else:
                print(f"  Result: {instance}")
        else:
            sys.exit(1)

    elif args.command == 'list':
        instances = list_instances(args.run_id)
        print_instances(instances)

    elif args.command == 'terminate':
        count = terminate_instances(args.run_id, args.force)
        print(f"\nTerminated {count} instance(s)")

    else:
        parser.print_help()


if __name__ == '__main__':
    try:
        main()
    except VastAPIError as e:
        logger.error(str(e))
        sys.exit(1)
