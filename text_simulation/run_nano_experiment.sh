#!/bin/bash
# run_nano_experiment.sh — Master runner for the nano experiment.
#
# Runs all 120 configs (10 settings × 4 reasoning levels × 3 reps).
# Each config is an independent call to run_LLM_simulations.py.
#
# Usage (from Digital-Twin-Simulation/):
#   bash text_simulation/run_nano_experiment.sh                     # all 120
#   bash text_simulation/run_nano_experiment.sh skill_v3            # only skill_v3 setting
#   bash text_simulation/run_nano_experiment.sh skill_v3 high       # only skill_v3 + high reasoning
#   bash text_simulation/run_nano_experiment.sh skill_v3 high rep_1 # single config
#
# Logs are written to text_simulation/logs/nano/<config_name>.log

set -euo pipefail

FILTER_SETTING="${1:-}"
FILTER_REASONING="${2:-}"
FILTER_REP="${3:-}"

CONFIG_DIR="text_simulation/configs/nano"
LOG_DIR="text_simulation/logs/nano"
mkdir -p "$LOG_DIR"

TOTAL=0
SKIPPED=0
FAILED=0
SUCCEEDED=0

for config_path in "$CONFIG_DIR"/*.yaml; do
    config_name=$(basename "$config_path" .yaml)

    # Parse config name: setting__reasoning__rep
    IFS='__' read -r setting reasoning rep <<< "$config_name"

    # Apply filters
    if [[ -n "$FILTER_SETTING"   && "$setting"   != "$FILTER_SETTING"   ]]; then continue; fi
    if [[ -n "$FILTER_REASONING" && "$reasoning" != "$FILTER_REASONING" ]]; then continue; fi
    if [[ -n "$FILTER_REP"       && "$rep"       != "$FILTER_REP"       ]]; then continue; fi

    TOTAL=$((TOTAL + 1))
    log_path="$LOG_DIR/${config_name}.log"

    echo "[$(date '+%H:%M:%S')] Running: $config_name"

    if python text_simulation/run_LLM_simulations.py --config "$config_path" > "$log_path" 2>&1; then
        SUCCEEDED=$((SUCCEEDED + 1))
        echo "  ✓ done → $log_path"
    else
        FAILED=$((FAILED + 1))
        echo "  ✗ FAILED → $log_path"
    fi
done

echo ""
echo "===== Nano Experiment Complete ====="
echo "  Total:     $TOTAL"
echo "  Succeeded: $SUCCEEDED"
echo "  Failed:    $FAILED"
echo "  Skipped:   $SKIPPED"
