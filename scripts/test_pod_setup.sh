#!/bin/bash
#
# Test Pod Setup for Resilient Training
#
# Run this after pod_setup.sh to verify everything is working.
# Meant to be run on a GPU pod with all dependencies installed.
#
# Usage:
#   ./test_pod_setup.sh
#

set -e

echo "=============================================="
echo "Resilient Training Pod Tests"
echo "=============================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASSED=0
FAILED=0

pass() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
    ((PASSED++))
}

fail() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    ((FAILED++))
}

warn() {
    echo -e "${YELLOW}⚠ WARN${NC}: $1"
}

# Test 1: Environment variables
echo "--- Environment Variables ---"

if [ -n "$B2_APPLICATION_KEY_ID" ]; then
    pass "B2_APPLICATION_KEY_ID is set"
else
    fail "B2_APPLICATION_KEY_ID not set"
fi

if [ -n "$B2_APPLICATION_KEY" ]; then
    pass "B2_APPLICATION_KEY is set"
else
    fail "B2_APPLICATION_KEY not set"
fi

if [ -n "$WANDB_API_KEY" ]; then
    pass "WANDB_API_KEY is set"
else
    fail "WANDB_API_KEY not set"
fi

if [ -n "$RUN_ID" ]; then
    pass "RUN_ID is set: $RUN_ID"
else
    warn "RUN_ID not set (using 'default')"
    export RUN_ID="default"
fi

if [ "$VLLM_USE_V1" = "0" ]; then
    pass "VLLM_USE_V1=0 (V1 engine disabled)"
else
    fail "VLLM_USE_V1 should be 0"
fi

if [ "$VLLM_WORKER_MULTIPROC_METHOD" = "spawn" ]; then
    pass "VLLM_WORKER_MULTIPROC_METHOD=spawn"
else
    fail "VLLM_WORKER_MULTIPROC_METHOD should be 'spawn'"
fi

echo ""

# Test 2: CLI tools
echo "--- CLI Tools ---"

if command -v b2 &> /dev/null; then
    pass "b2 CLI installed: $(b2 version 2>/dev/null | head -1 || echo 'unknown version')"
else
    fail "b2 CLI not installed"
fi

if command -v jq &> /dev/null; then
    pass "jq installed"
else
    fail "jq not installed"
fi

if command -v uv &> /dev/null; then
    pass "uv installed"
else
    fail "uv not installed"
fi

echo ""

# Test 3: B2 connectivity
echo "--- B2 Connectivity ---"

if [ -n "$B2_APPLICATION_KEY_ID" ] && [ -n "$B2_APPLICATION_KEY" ]; then
    if b2 authorize-account "$B2_APPLICATION_KEY_ID" "$B2_APPLICATION_KEY" > /dev/null 2>&1; then
        pass "B2 authorization successful"

        # Test bucket access
        if b2 ls beautifulsoup-rl > /dev/null 2>&1; then
            pass "beautifulsoup-rl bucket accessible"

            # Test upload/download
            TEST_FILE="/tmp/test_$(date +%s).txt"
            echo "test-$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$TEST_FILE"

            if b2 file upload beautifulsoup-rl "$TEST_FILE" "test/pod_test.txt" > /dev/null 2>&1; then
                pass "B2 upload works"

                DOWNLOAD_FILE="/tmp/downloaded_test.txt"
                if b2 file download "b2://beautifulsoup-rl/test/pod_test.txt" "$DOWNLOAD_FILE" > /dev/null 2>&1; then
                    if diff -q "$TEST_FILE" "$DOWNLOAD_FILE" > /dev/null 2>&1; then
                        pass "B2 download roundtrip verified"
                    else
                        fail "B2 download content mismatch"
                    fi
                    rm -f "$DOWNLOAD_FILE"
                else
                    fail "B2 download failed"
                fi

                # Cleanup
                b2 delete-file-version "test/pod_test.txt" > /dev/null 2>&1 || true
            else
                fail "B2 upload failed"
            fi

            rm -f "$TEST_FILE"
        else
            fail "beautifulsoup-rl bucket not accessible"
        fi
    else
        fail "B2 authorization failed"
    fi
else
    warn "Skipping B2 tests (credentials not set)"
fi

echo ""

# Test 4: WandB connectivity
echo "--- WandB Connectivity ---"

if [ -n "$WANDB_API_KEY" ]; then
    # Test via Python
    if python3 -c "import wandb; api = wandb.Api(); print('WandB API OK')" 2>/dev/null; then
        pass "WandB API accessible"
    else
        fail "WandB API connection failed"
    fi
else
    warn "Skipping WandB tests (WANDB_API_KEY not set)"
fi

echo ""

# Test 5: Scripts
echo "--- Scripts ---"

SCRIPT_DIR="$(dirname "$0")"

for script in checkpoint_sync.sh onstart.sh; do
    if [ -f "$SCRIPT_DIR/$script" ]; then
        if bash -n "$SCRIPT_DIR/$script" 2>/dev/null; then
            pass "$script syntax OK"
        else
            fail "$script has syntax errors"
        fi
    else
        fail "$script not found"
    fi
done

for script in training_controller.py wandb_monitor.py provision_vast.py; do
    if [ -f "$SCRIPT_DIR/$script" ]; then
        if python3 -m py_compile "$SCRIPT_DIR/$script" 2>/dev/null; then
            pass "$script syntax OK"
        else
            fail "$script has syntax errors"
        fi
    else
        fail "$script not found"
    fi
done

echo ""

# Test 6: File descriptor limit
echo "--- System Limits ---"

FD_LIMIT=$(ulimit -n 2>/dev/null || echo "0")
if [ "$FD_LIMIT" -ge 65536 ]; then
    pass "File descriptor limit OK: $FD_LIMIT"
else
    fail "File descriptor limit too low: $FD_LIMIT (need 65536+)"
fi

echo ""

# Test 7: prime-rl and environment
echo "--- Prime-RL ---"

if [ -d ~/prime-rl ]; then
    pass "prime-rl directory exists"

    cd ~/prime-rl
    if uv run python -c "import beautiful_soup_env; print('Environment OK')" 2>/dev/null; then
        pass "beautiful_soup_env imports successfully"
    else
        fail "beautiful_soup_env import failed (run: prime env install seconds-0/beautiful-soup-env)"
    fi
else
    warn "prime-rl not installed yet"
fi

echo ""

# Test 8: GPU
echo "--- GPU ---"

if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)

    pass "GPU detected: ${GPU_COUNT}x $GPU_NAME ($GPU_MEM)"
else
    fail "nvidia-smi not available"
fi

echo ""

# Summary
echo "=============================================="
echo "Summary"
echo "=============================================="
echo ""
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed! Pod is ready for training.${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Upload config: b2 file upload beautifulsoup-rl config.toml \$RUN_ID/config.toml"
    echo "  2. Start training: uv run rl @ /root/config.toml --ckpt --ckpt.interval 5"
    exit 0
else
    echo -e "${RED}✗ Some tests failed. Fix issues above before training.${NC}"
    exit 1
fi
