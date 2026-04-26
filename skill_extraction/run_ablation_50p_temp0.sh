#!/bin/bash
# run_ablation_50p_temp0.sh — Run 50-persona ablation (v2+v3, high reasoning, 4 conditions).

set -euo pipefail

FILTER_VERSION="${1:-}"
FILTER_CONDITION="${2:-}"

LOG_DIR="text_simulation/logs/ablation_50p_temp0"
mkdir -p "$LOG_DIR"

TOTAL=0; SUCCEEDED=0; FAILED=0

for version in v2 v3; do
    if [[ -n "$FILTER_VERSION" && "$version" != "$FILTER_VERSION" ]]; then continue; fi

    SUITE="nano_v2_ablation_50p_${version}_temp0"
    SUITE_LOG="$LOG_DIR/$SUITE"
    mkdir -p "$SUITE_LOG"

    for config_path in "text_simulation/configs/${SUITE}"/*.yaml; do
        config_name=$(basename "$config_path" .yaml)
        condition="${config_name%%__*}"

        if [[ -n "$FILTER_CONDITION" && "$condition" != "$FILTER_CONDITION" ]]; then continue; fi

        TOTAL=$((TOTAL + 1))
        log_path="$SUITE_LOG/${config_name}.log"

        out_dir="text_simulation/text_simulation_output_${SUITE}/${condition}/high"
        if compgen -G "${out_dir}/pid_*/pid_*_response.json" > /dev/null 2>&1; then
            SUCCEEDED=$((SUCCEEDED + 1))
            echo "[$(date '+%H:%M:%S')] SKIP (done): ${SUITE}/${config_name}"
            continue
        fi

        echo "[$(date '+%H:%M:%S')] ${SUITE}/${config_name}"
        if python text_simulation/run_nano_batch.py --config "$config_path" > "$log_path" 2>&1; then
            SUCCEEDED=$((SUCCEEDED + 1))
            echo "  OK -> $log_path"
        else
            FAILED=$((FAILED + 1))
            echo "  FAILED -> $log_path"
        fi
    done
done

echo ""
echo "===== 50p Ablation Complete ====="
echo "  Total: $TOTAL  Succeeded: $SUCCEEDED  Failed: $FAILED"
