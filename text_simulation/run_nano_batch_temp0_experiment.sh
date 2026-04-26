#!/bin/bash
# run_nano_batch_temp0_experiment.sh — Batch API runner for nano temp=0.0 experiment.
#
# Submits all 40 configs (10 settings × 4 reasoning levels) to OpenAI Batch API.
# Each config is one batch job (20 personas). Runs sequentially to avoid rate limits.
#
# Usage (from Digital-Twin-Simulation/):
#   bash text_simulation/run_nano_batch_temp0_experiment.sh              # all 40
#   bash text_simulation/run_nano_batch_temp0_experiment.sh skill_v3     # one setting
#   bash text_simulation/run_nano_batch_temp0_experiment.sh skill_v3 high # one config
#
# Logs: text_simulation/logs/nano_batch_temp0/<config_name>.log

set -euo pipefail

FILTER_SETTING="${1:-}"
FILTER_REASONING="${2:-}"

CONFIG_DIR="text_simulation/configs/nano_temp0"
LOG_DIR="text_simulation/logs/nano_batch_temp0"
mkdir -p "$LOG_DIR"

TOTAL=0
FAILED=0
SUCCEEDED=0

for config_path in "$CONFIG_DIR"/*.yaml; do
    config_name=$(basename "$config_path" .yaml)
    setting="${config_name%%__*}"
    reasoning="${config_name##*__}"

    if [[ -n "$FILTER_SETTING"   && "$setting"   != "$FILTER_SETTING"   ]]; then continue; fi
    if [[ -n "$FILTER_REASONING" && "$reasoning" != "$FILTER_REASONING" ]]; then continue; fi

    TOTAL=$((TOTAL + 1))
    log_path="$LOG_DIR/${config_name}.log"

    echo "[$(date '+%H:%M:%S')] $config_name"

    if python text_simulation/run_nano_batch.py \
          --config "$config_path" \
          > "$log_path" 2>&1; then
        SUCCEEDED=$((SUCCEEDED + 1))
        echo "  ✓ → $log_path"
    else
        FAILED=$((FAILED + 1))
        echo "  ✗ FAILED → $log_path"
    fi
done

echo ""
echo "===== Nano Batch Temp0 Experiment Complete ====="
echo "  Total:     $TOTAL"
echo "  Succeeded: $SUCCEEDED"
echo "  Failed:    $FAILED"
