#!/bin/bash
# run_nano_experiment.sh — Master runner for the nano experiment.
#
# Runs all 40 configs × 3 repetitions = 120 total runs.
# All runs use temperature=1.0; variance comes from LLM sampling randomness.
# Each rep writes to a separate output subfolder (rep_1, rep_2, rep_3).
#
# Usage (from Digital-Twin-Simulation/):
#   bash text_simulation/run_nano_experiment.sh                     # all 120
#   bash text_simulation/run_nano_experiment.sh skill_v3            # only skill_v3 setting
#   bash text_simulation/run_nano_experiment.sh skill_v3 high       # only skill_v3 + high reasoning
#   bash text_simulation/run_nano_experiment.sh skill_v3 high rep_2 # single rep
#
# Logs: text_simulation/logs/nano/<config_name>__<rep>.log

set -euo pipefail

FILTER_SETTING="${1:-}"
FILTER_REASONING="${2:-}"
FILTER_REP="${3:-}"

CONFIG_DIR="text_simulation/configs/nano"
LOG_DIR="text_simulation/logs/nano"
mkdir -p "$LOG_DIR"

TOTAL=0
FAILED=0
SUCCEEDED=0

for config_path in "$CONFIG_DIR"/*.yaml; do
    config_name=$(basename "$config_path" .yaml)

    # Parse config name: setting__reasoning
    setting="${config_name%%__*}"
    reasoning="${config_name##*__}"

    # Apply setting/reasoning filters
    if [[ -n "$FILTER_SETTING"   && "$setting"   != "$FILTER_SETTING"   ]]; then continue; fi
    if [[ -n "$FILTER_REASONING" && "$reasoning" != "$FILTER_REASONING" ]]; then continue; fi

    for rep in rep_1 rep_2 rep_3; do
        # Apply rep filter
        if [[ -n "$FILTER_REP" && "$rep" != "$FILTER_REP" ]]; then continue; fi

        TOTAL=$((TOTAL + 1))
        run_name="${config_name}__${rep}"
        log_path="$LOG_DIR/${run_name}.log"

        # Inject rep into output_folder_dir by passing as env var; runner reads NANO_REP
        echo "[$(date '+%H:%M:%S')] $run_name"

        if python text_simulation/run_nano_batch.py \
              --config "$config_path" \
              --nano_rep "$rep" \
              > "$log_path" 2>&1; then
            SUCCEEDED=$((SUCCEEDED + 1))
            echo "  ✓ → $log_path"
        else
            FAILED=$((FAILED + 1))
            echo "  ✗ FAILED → $log_path"
        fi
    done
done

echo ""
echo "===== Nano Experiment Complete ====="
echo "  Total:     $TOTAL"
echo "  Succeeded: $SUCCEEDED"
echo "  Failed:    $FAILED"
