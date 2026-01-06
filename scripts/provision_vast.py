#!/usr/bin/env python3
"""
Vast.ai Instance Provisioning Script

Provision GPU instances on Vast.ai for training. Can be used manually
or called by the training controller.

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
import sys
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# GPU type mapping for Vast.ai
GPU_TYPES = {
    'H100': 'H100_PCIE',
    'H100_PCIE': 'H100_PCIE',
    'H100_SXM': 'H100_SXM5',
    'A100': 'A100_PCIE',
    'A100_80GB': 'A100_SXM4',
    '4090': 'RTX_4090',
    'RTX4090': 'RTX_4090',
}


def get_vast_client():
    """Get authenticated Vast.ai client."""
    api_key = os.environ.get('VAST_API_KEY')
    if not api_key:
        logger.error("VAST_API_KEY environment variable not set")
        sys.exit(1)

    try:
        import vastai
        return vastai.VastAI(api_key=api_key)
    except ImportError:
        logger.error("vastai package not installed. Run: pip install vastai")
        sys.exit(1)


def search_offers(gpu_type: str, gpu_count: int, max_price: float, limit: int = 20):
    """Search for available GPU offers."""
    vast = get_vast_client()

    # Normalize GPU type
    gpu_name = GPU_TYPES.get(gpu_type.upper(), gpu_type)

    logger.info(f"Searching for {gpu_count}x {gpu_name} under ${max_price}/GPU/hr")

    try:
        offers = vast.search_offers(
            gpu_name=gpu_name,
            num_gpus=gpu_count,
            order="dph_total",
            limit=limit
        )

        if not offers:
            logger.warning("No offers found")
            return []

        # Filter by price
        max_total = max_price * gpu_count
        filtered = [o for o in offers if o.get('dph_total', float('inf')) <= max_total]

        logger.info(f"Found {len(filtered)} offers under ${max_total:.2f}/hr total")

        return filtered

    except Exception as e:
        logger.error(f"Error searching offers: {e}")
        return []


def create_instance(
    run_id: str,
    gpu_type: str,
    gpu_count: int,
    max_price: float,
    docker_image: str = "nvidia/cuda:12.1-devel-ubuntu22.04",
    disk_gb: int = 100
) -> Optional[dict]:
    """Create a new Vast.ai instance for training."""
    vast = get_vast_client()

    # Search for offers
    offers = search_offers(gpu_type, gpu_count, max_price)
    if not offers:
        logger.error("No suitable offers found")
        return None

    best_offer = offers[0]
    logger.info(f"Selected offer {best_offer['id']}: "
                f"{best_offer.get('gpu_name', 'GPU')} x{best_offer.get('num_gpus', '?')} "
                f"at ${best_offer['dph_total']:.2f}/hr")

    # Build onstart script URL
    onstart_url = "https://raw.githubusercontent.com/seconds-0/beautifulsoup-rl/main/scripts/onstart.sh"

    # Environment variables to pass to instance
    env_vars = {
        'RUN_ID': run_id,
        'B2_BUCKET': 'beautifulsoup-rl',
    }

    # Add secrets from environment
    for var in ['B2_APPLICATION_KEY_ID', 'B2_APPLICATION_KEY', 'WANDB_API_KEY']:
        if os.environ.get(var):
            env_vars[var] = os.environ[var]

    try:
        logger.info("Creating instance...")

        instance = vast.create_instance(
            offer_id=best_offer['id'],
            image=docker_image,
            label=f"bs4-training-{run_id}",
            disk=disk_gb,
            onstart=f"curl -sSL {onstart_url} | bash",
            env=env_vars
        )

        logger.info(f"Instance created: {instance}")
        return instance

    except Exception as e:
        logger.error(f"Error creating instance: {e}")
        return None


def list_instances(run_id: Optional[str] = None) -> list:
    """List instances, optionally filtered by run_id label."""
    vast = get_vast_client()

    try:
        instances = vast.show_instances()

        if run_id:
            label_prefix = f"bs4-training-{run_id}"
            instances = [i for i in instances if i.get('label', '').startswith(label_prefix)]

        return instances

    except Exception as e:
        logger.error(f"Error listing instances: {e}")
        return []


def terminate_instances(run_id: str, force: bool = False) -> int:
    """Terminate instances matching run_id."""
    vast = get_vast_client()

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

        try:
            logger.info(f"Terminating instance {inst_id} (status: {status})")
            vast.destroy_instance(inst_id)
            terminated += 1
        except Exception as e:
            logger.error(f"Failed to terminate {inst_id}: {e}")

    return terminated


def print_offers(offers: list):
    """Pretty-print offer list."""
    if not offers:
        print("No offers found")
        return

    print(f"\n{'ID':<10} {'GPU':<15} {'Count':<6} {'$/hr':<8} {'VRAM':<8} {'Location':<15}")
    print("-" * 70)

    for o in offers:
        print(f"{o['id']:<10} "
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
        print(f"{i['id']:<10} "
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
            print(f"  ID: {instance.get('id')}")
            print(f"  SSH: vast ssh {instance.get('id')}")
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
    main()
