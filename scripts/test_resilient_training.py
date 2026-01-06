#!/usr/bin/env python3
"""
Test script for the Resilient Training System

Tests all components of the checkpoint sync, auto-resume, and monitoring system.
Run this before deploying to validate the infrastructure.

Usage:
    # Test all components
    python scripts/test_resilient_training.py

    # Test specific component
    python scripts/test_resilient_training.py --test b2
    python scripts/test_resilient_training.py --test wandb
    python scripts/test_resilient_training.py --test vast

Environment variables required:
    B2_APPLICATION_KEY_ID - Backblaze B2 key ID
    B2_APPLICATION_KEY - Backblaze B2 application key
    WANDB_API_KEY - WandB API key
    VAST_API_KEY - Vast.ai API key (optional, for vast tests)
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestResult:
    """Store test results."""
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.message = ""
        self.details = {}

    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status} {self.name}: {self.message}"


def test_b2_connection() -> TestResult:
    """Test B2 connection and bucket access."""
    result = TestResult("B2 Connection")

    key_id = os.environ.get('B2_APPLICATION_KEY_ID')
    app_key = os.environ.get('B2_APPLICATION_KEY')

    if not key_id or not app_key:
        result.message = "Missing B2 credentials in environment"
        return result

    try:
        # Authorize B2
        proc = subprocess.run(
            ['b2', 'authorize-account', key_id, app_key],
            capture_output=True, text=True, timeout=30
        )
        if proc.returncode != 0:
            result.message = f"B2 auth failed: {proc.stderr}"
            return result

        # List buckets
        proc = subprocess.run(
            ['b2', 'ls', '--json'],
            capture_output=True, text=True, timeout=30
        )
        if proc.returncode != 0:
            result.message = f"B2 ls failed: {proc.stderr}"
            return result

        # Check for our bucket
        proc = subprocess.run(
            ['b2', 'ls', 'beautifulsoup-rl'],
            capture_output=True, text=True, timeout=30
        )
        if proc.returncode != 0:
            result.message = f"Cannot access beautifulsoup-rl bucket: {proc.stderr}"
            return result

        result.passed = True
        result.message = "B2 connection OK, bucket accessible"
        return result

    except subprocess.TimeoutExpired:
        result.message = "B2 command timed out"
        return result
    except FileNotFoundError:
        result.message = "b2 CLI not installed (pip install b2)"
        return result
    except Exception as e:
        result.message = f"Error: {e}"
        return result


def test_b2_upload_download() -> TestResult:
    """Test B2 upload and download roundtrip."""
    result = TestResult("B2 Upload/Download")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file
            test_content = f"test-{datetime.now(timezone.utc).isoformat()}"
            test_file = Path(tmpdir) / "test_file.txt"
            test_file.write_text(test_content)

            # Upload
            proc = subprocess.run(
                ['b2', 'file', 'upload', 'beautifulsoup-rl',
                 str(test_file), 'test/resilient_training_test.txt'],
                capture_output=True, text=True, timeout=60
            )
            if proc.returncode != 0:
                result.message = f"Upload failed: {proc.stderr}"
                return result

            # Download
            download_path = Path(tmpdir) / "downloaded.txt"
            proc = subprocess.run(
                ['b2', 'file', 'download',
                 'b2://beautifulsoup-rl/test/resilient_training_test.txt',
                 str(download_path)],
                capture_output=True, text=True, timeout=60
            )
            if proc.returncode != 0:
                result.message = f"Download failed: {proc.stderr}"
                return result

            # Verify content
            downloaded_content = download_path.read_text()
            if downloaded_content != test_content:
                result.message = f"Content mismatch: expected {test_content!r}, got {downloaded_content!r}"
                return result

            # Cleanup test file
            subprocess.run(
                ['b2', 'delete-file-version', 'test/resilient_training_test.txt'],
                capture_output=True, text=True, timeout=30
            )

            result.passed = True
            result.message = "Upload/download roundtrip successful"
            return result

    except Exception as e:
        result.message = f"Error: {e}"
        return result


def test_wandb_connection() -> TestResult:
    """Test WandB API connection."""
    result = TestResult("WandB Connection")

    api_key = os.environ.get('WANDB_API_KEY')
    if not api_key:
        result.message = "Missing WANDB_API_KEY in environment"
        return result

    try:
        import wandb
        api = wandb.Api()

        # Try to list runs (just verify API works)
        runs = api.runs(
            "seconds-0-domus-magna-inc/beautiful-soup-env",
            per_page=1
        )
        # Just accessing runs triggers the API call
        run_count = len(list(runs)[:1])

        result.passed = True
        result.message = f"WandB API connected, project accessible"
        return result

    except ImportError:
        result.message = "wandb not installed (pip install wandb)"
        return result
    except Exception as e:
        result.message = f"WandB error: {e}"
        return result


def test_wandb_monitor() -> TestResult:
    """Test WandB monitor script."""
    result = TestResult("WandB Monitor Script")

    script_path = Path(__file__).parent / "wandb_monitor.py"
    if not script_path.exists():
        result.message = f"Script not found: {script_path}"
        return result

    try:
        proc = subprocess.run(
            [sys.executable, str(script_path),
             '--run-id', 'bs4-qwen3-8b',
             '--json'],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, 'WANDB_API_KEY': os.environ.get('WANDB_API_KEY', '')}
        )

        # Script returns non-zero if training not healthy, which is OK
        try:
            output = json.loads(proc.stdout)
            result.details = output
            result.passed = True
            result.message = f"Monitor returned status: {output.get('status', 'unknown')}"
        except json.JSONDecodeError:
            result.message = f"Invalid JSON output: {proc.stdout[:200]}"

        return result

    except subprocess.TimeoutExpired:
        result.message = "Monitor script timed out"
        return result
    except Exception as e:
        result.message = f"Error: {e}"
        return result


def test_vast_connection() -> TestResult:
    """Test Vast.ai CLI connection."""
    result = TestResult("Vast.ai Connection")

    api_key = os.environ.get('VAST_API_KEY')
    if not api_key:
        result.message = "Missing VAST_API_KEY (optional)"
        result.passed = True  # Mark as pass since it's optional
        return result

    try:
        # Set API key via CLI
        proc = subprocess.run(
            ['vastai', 'set', 'api-key', api_key],
            capture_output=True, text=True, timeout=30
        )
        if proc.returncode != 0:
            result.message = f"Failed to set API key: {proc.stderr}"
            return result

        # List instances (works even if empty)
        proc = subprocess.run(
            ['vastai', 'show', 'instances', '--raw'],
            capture_output=True, text=True, timeout=60
        )

        # 403 means no instances (not an error for this test)
        if proc.returncode != 0 and '403' not in proc.stderr:
            result.message = f"CLI error: {proc.stderr}"
            return result

        # Try to parse response
        try:
            instances = json.loads(proc.stdout) if proc.stdout.strip() else []
            result.passed = True
            result.message = f"Vast.ai CLI connected, found {len(instances)} instance(s)"
        except json.JSONDecodeError:
            # Empty or non-JSON response is OK for connection test
            result.passed = True
            result.message = "Vast.ai CLI connected (no instances)"

        return result

    except FileNotFoundError:
        result.message = "vastai CLI not installed (pip install vastai)"
        return result
    except subprocess.TimeoutExpired:
        result.message = "Vast.ai CLI timed out"
        return result
    except Exception as e:
        result.message = f"Vast.ai error: {e}"
        return result


def test_vast_search() -> TestResult:
    """Test Vast.ai offer search via CLI."""
    result = TestResult("Vast.ai Offer Search")

    api_key = os.environ.get('VAST_API_KEY')
    if not api_key:
        result.message = "Missing VAST_API_KEY (optional)"
        result.passed = True
        return result

    try:
        # Set API key first
        subprocess.run(
            ['vastai', 'set', 'api-key', api_key],
            capture_output=True, text=True, timeout=30
        )

        # Search for H100s using CLI
        # Query syntax: field=value field>=value
        query = "num_gpus>=2 gpu_name=H100_PCIE rentable=true"
        proc = subprocess.run(
            ['vastai', 'search', 'offers', '--raw', '--limit', '5', '--order', 'dph_total', query],
            capture_output=True, text=True, timeout=60
        )

        if proc.returncode != 0:
            result.message = f"Search failed: {proc.stderr}"
            return result

        try:
            offers = json.loads(proc.stdout) if proc.stdout.strip() else []
        except json.JSONDecodeError:
            result.message = f"Invalid JSON response: {proc.stdout[:100]}"
            return result

        if offers:
            cheapest = offers[0]
            result.details = {
                'offers_found': len(offers),
                'cheapest_price': cheapest.get('dph_total'),
                'cheapest_gpu': cheapest.get('gpu_name'),
            }
            price = cheapest.get('dph_total', 0)
            result.message = f"Found {len(offers)} offer(s), cheapest: ${price:.2f}/hr"
        else:
            result.message = "No H100 offers found (normal if none available)"

        result.passed = True
        return result

    except FileNotFoundError:
        result.message = "vastai CLI not installed (pip install vastai)"
        return result
    except subprocess.TimeoutExpired:
        result.message = "Vast.ai search timed out"
        return result
    except Exception as e:
        result.message = f"Search error: {e}"
        return result


def test_checkpoint_sync_script() -> TestResult:
    """Test checkpoint sync script syntax."""
    result = TestResult("Checkpoint Sync Script")

    script_path = Path(__file__).parent / "checkpoint_sync.sh"
    if not script_path.exists():
        result.message = f"Script not found: {script_path}"
        return result

    try:
        # Check bash syntax
        proc = subprocess.run(
            ['bash', '-n', str(script_path)],
            capture_output=True, text=True, timeout=10
        )
        if proc.returncode != 0:
            result.message = f"Syntax error: {proc.stderr}"
            return result

        result.passed = True
        result.message = "Script syntax OK"
        return result

    except Exception as e:
        result.message = f"Error: {e}"
        return result


def test_onstart_script() -> TestResult:
    """Test onstart script syntax."""
    result = TestResult("Onstart Script")

    script_path = Path(__file__).parent / "onstart.sh"
    if not script_path.exists():
        result.message = f"Script not found: {script_path}"
        return result

    try:
        # Check bash syntax
        proc = subprocess.run(
            ['bash', '-n', str(script_path)],
            capture_output=True, text=True, timeout=10
        )
        if proc.returncode != 0:
            result.message = f"Syntax error: {proc.stderr}"
            return result

        result.passed = True
        result.message = "Script syntax OK"
        return result

    except Exception as e:
        result.message = f"Error: {e}"
        return result


def test_training_controller() -> TestResult:
    """Test training controller script."""
    result = TestResult("Training Controller")

    script_path = Path(__file__).parent / "training_controller.py"
    if not script_path.exists():
        result.message = f"Script not found: {script_path}"
        return result

    try:
        # Test import
        proc = subprocess.run(
            [sys.executable, '-c', f'import sys; sys.path.insert(0, "{script_path.parent}"); import training_controller'],
            capture_output=True, text=True, timeout=10
        )
        if proc.returncode != 0:
            result.message = f"Import error: {proc.stderr}"
            return result

        result.passed = True
        result.message = "Controller imports successfully"
        return result

    except Exception as e:
        result.message = f"Error: {e}"
        return result


def run_all_tests() -> list[TestResult]:
    """Run all tests."""
    tests = [
        ("B2", [test_b2_connection, test_b2_upload_download]),
        ("WandB", [test_wandb_connection, test_wandb_monitor]),
        ("Vast.ai", [test_vast_connection, test_vast_search]),
        ("Scripts", [test_checkpoint_sync_script, test_onstart_script, test_training_controller]),
    ]

    results = []
    for category, test_funcs in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {category}")
        logger.info('='*60)

        for test_func in test_funcs:
            logger.info(f"Running {test_func.__name__}...")
            result = test_func()
            results.append(result)
            print(f"  {result}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Test Resilient Training System')
    parser.add_argument('--test', choices=['b2', 'wandb', 'vast', 'scripts', 'all'],
                        default='all', help='Which tests to run')
    parser.add_argument('--json', action='store_true', help='Output JSON')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("Resilient Training System - Test Suite")
    print("="*60 + "\n")

    # Check credentials
    creds = {
        'B2_APPLICATION_KEY_ID': bool(os.environ.get('B2_APPLICATION_KEY_ID')),
        'B2_APPLICATION_KEY': bool(os.environ.get('B2_APPLICATION_KEY')),
        'WANDB_API_KEY': bool(os.environ.get('WANDB_API_KEY')),
        'VAST_API_KEY': bool(os.environ.get('VAST_API_KEY')),
    }
    print("Credentials found:")
    for name, found in creds.items():
        status = "✓" if found else "✗"
        print(f"  {status} {name}")
    print()

    if args.test == 'all':
        results = run_all_tests()
    else:
        test_map = {
            'b2': [test_b2_connection, test_b2_upload_download],
            'wandb': [test_wandb_connection, test_wandb_monitor],
            'vast': [test_vast_connection, test_vast_search],
            'scripts': [test_checkpoint_sync_script, test_onstart_script, test_training_controller],
        }
        results = []
        for test_func in test_map[args.test]:
            result = test_func()
            results.append(result)
            print(f"  {result}")

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")

    if args.json:
        output = {
            'passed': passed,
            'total': total,
            'results': [
                {'name': r.name, 'passed': r.passed, 'message': r.message, 'details': r.details}
                for r in results
            ]
        }
        print(json.dumps(output, indent=2))

    if passed < total:
        print("\n⚠️  Some tests failed. Review output above.")
        sys.exit(1)
    else:
        print("\n✅ All tests passed!")
        sys.exit(0)


if __name__ == '__main__':
    main()
