#!/bin/bash
# run_skills_ep_ablation_temp0.sh — Batch API runner for skills+ep ablation experiment.
#
# Submits 84 configs (7 conditions × 4 reasoning levels × 3 versions) to OpenAI Batch API.
# Runs sequentially to avoid rate limits. Each job covers 20 personas.
#
# Usage (from Digital-Twin-Simulation/):
#   bash text_simulation/run_skills_ep_ablation_temp0.sh            # all 84
#   bash text_simulation/run_skills_ep_ablation_temp0.sh v2         # one version
#   bash text_simulation/run_skills_ep_ablation_temp0.sh v3 bg_dp_tools_ep  # one condition
#   bash text_simulation/run_skills_ep_ablation_temp0.sh v1 bg high         # single config
#
# Logs: text_simulation/logs/skills_ep_ablation_temp0/<suite>/<config>.log

set -euo pipefail

FILTER_VERSION="${1:-}"
FILTER_CONDITION="${2:-}"
FILTER_REASONING="${3:-}"

LOG_DIR="text_simulation/logs/skills_ep_ablation_temp0"
mkdir -p "$LOG_DIR"

TOTAL=0
FAILED=0
SUCCEEDED=0

for version in v1 v2 v3; do
    if [[ -n "$FILTER_VERSION" && "$version" != "$FILTER_VERSION" ]]; then continue; fi

    SUITE="nano_skills_ep_ablation_${version}_temp0"
    CONFIG_DIR="text_simulation/configs/${SUITE}"
    SUITE_LOG_DIR="${LOG_DIR}/${SUITE}"
    mkdir -p "$SUITE_LOG_DIR"

    for config_path in "$CONFIG_DIR"/*.yaml; do
        config_name=$(basename "$config_path" .yaml)
        condition="${config_name%%__*}"
        reasoning="${config_name##*__}"

        if [[ -n "$FILTER_CONDITION" && "$condition" != "$FILTER_CONDITION" ]]; then continue; fi
        if [[ -n "$FILTER_REASONING" && "$reasoning" != "$FILTER_REASONING" ]]; then continue; fi

        TOTAL=$((TOTAL + 1))
        log_path="${SUITE_LOG_DIR}/${config_name}.log"

        # Skip if output already exists with results
        out_dir="text_simulation/text_simulation_output_nano_skills_ep_ablation_${version}_temp0/${condition}/${reasoning}"
        if compgen -G "${out_dir}/pid_*/pid_*_response.json" > /dev/null 2>&1; then
            SUCCEEDED=$((SUCCEEDED + 1))
            echo "[$(date '+%H:%M:%S')] SKIP (already done): ${SUITE} / ${config_name}"
            continue
        fi

        echo "[$(date '+%H:%M:%S')] ${SUITE} / ${config_name}"

        if python text_simulation/run_nano_batch.py \
              --config "$config_path" \
              > "$log_path" 2>&1; then
            SUCCEEDED=$((SUCCEEDED + 1))
            echo "  OK -> $log_path"
        else
            FAILED=$((FAILED + 1))
            echo "  FAILED -> $log_path"
        fi
    done
done

echo ""
echo "===== Skills+EP Ablation Experiment Complete ====="
echo "  Total:     $TOTAL"
echo "  Succeeded: $SUCCEEDED"
echo "  Failed:    $FAILED"
