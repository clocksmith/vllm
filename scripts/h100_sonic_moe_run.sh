#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Sonic MoE H100 test, benchmark, and sweep runner.
#
# Prerequisites:
#   - H100 GPU with CUDA 12.9+
#   - Python 3.12+
#   - This repo checked out on feature/sonic-moe-integration-h100-tmp
#
# Usage:
#   # Full run (setup + test + benchmark + sweep + plot):
#   bash scripts/h100_sonic_moe_run.sh
#
#   # Skip setup (already installed):
#   bash scripts/h100_sonic_moe_run.sh --skip-setup
#
#   # Only run specific stages:
#   bash scripts/h100_sonic_moe_run.sh --only test
#   bash scripts/h100_sonic_moe_run.sh --only bench
#   bash scripts/h100_sonic_moe_run.sh --only sweep
#   bash scripts/h100_sonic_moe_run.sh --only plot
#
#   # Quick sweep (fewer shapes, ~5 min):
#   bash scripts/h100_sonic_moe_run.sh --only sweep --quick
#
# Outputs go to results/ directory with timestamps.

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
SKIP_SETUP=false
ONLY=""
QUICK=false
RESULTS_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-setup) SKIP_SETUP=true; shift ;;
        --only) ONLY="$2"; shift 2 ;;
        --quick) QUICK=true; shift ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results/$TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

LOG="$RESULTS_DIR/run.log"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()  { echo -e "\n\033[1;34m>>> $*\033[0m" | tee -a "$LOG"; }
ok()    { echo -e "\033[1;32m    OK: $*\033[0m" | tee -a "$LOG"; }
fail()  { echo -e "\033[1;31m    FAIL: $*\033[0m" | tee -a "$LOG"; }
run()   { echo "+ $*" >> "$LOG"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }

should_run() {
    [[ -z "$ONLY" ]] || [[ "$ONLY" == "$1" ]]
}

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
info "Preflight checks"

if ! command -v nvidia-smi &>/dev/null; then
    fail "nvidia-smi not found"; exit 1
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
if [[ "$GPU_NAME" != *"H100"* ]] && [[ "$GPU_NAME" != *"H200"* ]]; then
    fail "Expected Hopper GPU, got: $GPU_NAME"; exit 1
fi
ok "GPU: $GPU_NAME"

run nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
    --format=csv,noheader | tee "$RESULTS_DIR/gpu_info.txt"

PYTHON="${PYTHON:-python3}"
PY_VERSION="$($PYTHON --version 2>&1)"
ok "Python: $PY_VERSION"

CUDA_VERSION="$($PYTHON -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'unknown')"
ok "CUDA: $CUDA_VERSION"

echo "$GPU_NAME" > "$RESULTS_DIR/gpu_name.txt"

# ---------------------------------------------------------------------------
# Stage 1: Setup
# ---------------------------------------------------------------------------
if ! $SKIP_SETUP && should_run setup; then
    info "Stage 1: Installing dependencies"

    # vLLM (precompiled for speed, Python-only changes don't need rebuild)
    info "Installing vLLM (precompiled editable)"
    run $PYTHON -m pip install --upgrade pip
    run $PYTHON -m pip install uv

    # Install test deps first (platform-agnostic)
    if [[ -f requirements/test.in ]]; then
        run uv pip install -r requirements/test.in
    fi

    # Install vLLM
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
    VLLM_USE_PRECOMPILED=1 run uv pip install -e . --torch-backend=auto

    # Sonic MoE from PyPI (install last to avoid dep clobbering)
    info "Installing sonic-moe from PyPI"
    run $PYTHON -m pip install sonic-moe

    # matplotlib for plotting
    run $PYTHON -m pip install matplotlib

    # Verify imports
    info "Verifying imports"
    $PYTHON -c "
import torch, vllm
print(f'vLLM:    {vllm.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA:    {torch.version.cuda}')
print(f'GPU:     {torch.cuda.get_device_name(0)}')
print(f'SM:      {torch.cuda.get_device_capability(0)}')
" | tee "$RESULTS_DIR/versions.txt"

    $PYTHON -c "
try:
    import sonicmoe; print(f'SonicMoE: available')
except ImportError as e:
    print(f'SonicMoE: MISSING ({e})')
    exit(1)
" | tee -a "$RESULTS_DIR/versions.txt"

    # Package versions for reproducibility
    $PYTHON -m pip show nvidia-cutlass-dsl quack-kernels sonic-moe flashinfer-python 2>/dev/null \
        | grep -E '^(Name|Version):' | tee "$RESULTS_DIR/package_versions.txt"

    ok "Setup complete"
fi

# ---------------------------------------------------------------------------
# Stage 2: Tests
# ---------------------------------------------------------------------------
if should_run test; then
    info "Stage 2: Running Sonic MoE tests"
    TEST_OUT="$RESULTS_DIR/test_results.txt"

    if run $PYTHON -m pytest -v -s tests/kernels/moe/test_sonic_moe.py 2>&1 | tee "$TEST_OUT"; then
        ok "All tests passed"
    else
        fail "Some tests failed (see $TEST_OUT)"
        # Don't exit — continue to benchmarks even if some tests skip/fail
    fi
fi

# ---------------------------------------------------------------------------
# Stage 3: Microbenchmark (Sonic vs Triton, 2 shapes)
# ---------------------------------------------------------------------------
if should_run bench; then
    info "Stage 3: Running Sonic MoE microbenchmark"
    BENCH_JSON="$RESULTS_DIR/benchmark_sonic_moe.json"

    if run $PYTHON benchmarks/kernels/benchmark_sonic_moe.py \
        --warmup 20 --iters 100 \
        --output-json "$BENCH_JSON"; then
        ok "Benchmark complete: $BENCH_JSON"
        # Print summary table
        $PYTHON -c "
import json, sys
with open('$BENCH_JSON') as f:
    data = json.load(f)
print()
print(f\"Device: {data['device']}\")
print(f\"{'dtype':<6} {'E':>3} {'topk':>4} {'M':>5} {'K':>5} {'N':>5} {'triton_us':>10} {'sonic_us':>10} {'speedup':>8} {'rel_err':>8}\")
print('-' * 80)
for r in data['results']:
    print(f\"{r['dtype']:<6} {r['e']:>3} {r['topk']:>4} {r['m']:>5} {r['k']:>5} {r['n']:>5} {r['triton_us']:>10.1f} {r['sonic_us']:>10.1f} {r['speedup']:>8.3f} {r['rel_err']:>8.4f}\")
" | tee "$RESULTS_DIR/benchmark_summary.txt"
    else
        fail "Benchmark failed"
    fi
fi

# ---------------------------------------------------------------------------
# Stage 4: Backend sweep
# ---------------------------------------------------------------------------
if should_run sweep; then
    info "Stage 4: Running MoE backend sweep"
    SWEEP_JSON="$RESULTS_DIR/benchmark_moe_backend_sweep.json"

    SWEEP_ARGS=(
        --warmup 10
        --iters 50
        --output-json "$SWEEP_JSON"
    )

    if $QUICK; then
        info "(Quick mode: reduced shape grid)"
        SWEEP_ARGS+=(
            --prefill-m "256,512"
            --decode-m "1,8,64"
            --k-list "512,1024"
            --i-list "1024,2048"
            --e-list "8,16"
            --topk-list "2,4"
            --dtype bf16
        )
    fi

    if run $PYTHON benchmarks/kernels/benchmark_moe_backend_sweep.py "${SWEEP_ARGS[@]}"; then
        ok "Sweep complete: $SWEEP_JSON"
    else
        fail "Sweep failed"
    fi
fi

# ---------------------------------------------------------------------------
# Stage 5: Plot
# ---------------------------------------------------------------------------
if should_run plot; then
    # Find the most recent sweep JSON if not from this run
    SWEEP_JSON="${SWEEP_JSON:-$(ls -t "$RESULTS_DIR"/benchmark_moe_backend_sweep*.json 2>/dev/null | head -1)}"
    if [[ -z "$SWEEP_JSON" ]] || [[ ! -f "$SWEEP_JSON" ]]; then
        # Try parent results dir
        SWEEP_JSON="$(ls -t "$REPO_ROOT"/results/*/benchmark_moe_backend_sweep*.json 2>/dev/null | head -1)"
    fi

    if [[ -n "$SWEEP_JSON" ]] && [[ -f "$SWEEP_JSON" ]]; then
        info "Stage 5: Generating sweep plots"
        SWEEP_PNG="$RESULTS_DIR/benchmark_moe_backend_sweep_summary.png"

        if run $PYTHON benchmarks/kernels/plot_moe_backend_sweep.py \
            --input-json "$SWEEP_JSON" \
            --output-png "$SWEEP_PNG"; then
            ok "Plot: $SWEEP_PNG"
        else
            fail "Plot generation failed"
        fi
    else
        info "Stage 5: Skipping plot (no sweep JSON found)"
    fi
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
info "All done. Results in: $RESULTS_DIR"
ls -lh "$RESULTS_DIR"/ | tee -a "$LOG"
echo ""
echo "To copy results off this machine:"
echo "  scp -r $(whoami)@\$(hostname):$RESULTS_DIR ."
