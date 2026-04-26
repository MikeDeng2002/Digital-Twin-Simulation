#!/bin/bash
# run_o4mini_experiment.sh — Runner for o4-mini experiment.
#
# Runs all 10 configs (one per persona setting), 1 run each.
# No reasoning_effort — o4-mini reasons internally.
#
# Usage (from Digital-Twin-Simulation/):
#   bash text_simulation/run_o4mini_experiment.sh          # all 10
#   bash text_simulation/run_o4mini_experiment.sh skill_v3 # one setting
#
# Logs: text_simulation/logs/o4mini/<setting>.log

set -euo pipefail

FILTER_SETTING="${1:-}"

CONFIG_DIR="text_simulation/configs/o4mini"
LOG_DIR="text_simulation/logs/o4mini"
mkdir -p "$LOG_DIR"

TOTAL=0
FAILED=0
SUCCEEDED=0

for config_path in "$CONFIG_DIR"/*.yaml; do
    setting=$(basename "$config_path" .yaml)

    if [[ -n "$FILTER_SETTING" && "$setting" != "$FILTER_SETTING" ]]; then continue; fi

    TOTAL=$((TOTAL + 1))
    log_path="$LOG_DIR/${setting}.log"

    echo "[$(date '+%H:%M:%S')] $setting"

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
echo "===== o4-mini Experiment Complete ====="
echo "  Total:     $TOTAL"
echo "  Succeeded: $SUCCEEDED"
echo "  Failed:    $FAILED"
