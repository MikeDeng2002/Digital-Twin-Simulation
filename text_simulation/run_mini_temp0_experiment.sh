#!/bin/bash
# run_mini_temp0_experiment.sh — Runner for gpt-5.4-mini temp=0.0 experiment.
#
# Runs all 40 configs (10 settings × 4 reasoning levels), 1 run each.
# temperature=0.0 → deterministic, no repetitions needed.
#
# Usage (from Digital-Twin-Simulation/):
#   bash text_simulation/run_mini_temp0_experiment.sh              # all 40
#   bash text_simulation/run_mini_temp0_experiment.sh skill_v3     # one setting
#   bash text_simulation/run_mini_temp0_experiment.sh skill_v3 high # one config
#
# Logs: text_simulation/logs/mini_temp0/<config_name>.log

set -euo pipefail

FILTER_SETTING="${1:-}"
FILTER_REASONING="${2:-}"

CONFIG_DIR="text_simulation/configs/mini_temp0"
LOG_DIR="text_simulation/logs/mini_temp0"
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

    if python text_simulation/run_LLM_simulations.py \
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
echo "===== Mini Temp0 Experiment Complete ====="
echo "  Total:     $TOTAL"
echo "  Succeeded: $SUCCEEDED"
echo "  Failed:    $FAILED"
