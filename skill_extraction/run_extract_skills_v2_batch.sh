#!/bin/bash
# run_extract_skills_v2_batch.sh — Extract skills_v2 for a range of personas.
#
# Runs extract_skills_v2.py for each pid in the range, v2_inferred and v3_maximum only.
# Skips pids where both versions already exist (no --force).
#
# Usage (from Digital-Twin-Simulation/):
#   bash skill_extraction/run_extract_skills_v2_batch.sh 21 50
#   bash skill_extraction/run_extract_skills_v2_batch.sh 21 50 v2_inferred

set -euo pipefail

START="${1:-21}"
END="${2:-50}"
VERSION_FILTER="${3:-all}"   # "all", "v2_inferred", or "v3_maximum"

LOG_DIR="skill_extraction/logs/extract_v2_batch"
mkdir -p "$LOG_DIR"

TOTAL=0; OK=0; SKIP=0; FAIL=0

for pid in $(seq "$START" "$END"); do
    for version in v2_inferred v3_maximum; do
        if [[ "$VERSION_FILTER" != "all" && "$version" != "$VERSION_FILTER" ]]; then continue; fi

        out_dir="text_simulation/skills_v2/pid_${pid}/${version}"
        if [ -f "${out_dir}/background.txt" ] && [ -f "${out_dir}/decision_procedure.txt" ] && [ -f "${out_dir}/evaluation_profile.txt" ]; then
            SKIP=$((SKIP+1)); TOTAL=$((TOTAL+1))
            echo "SKIP pid_${pid}/${version} (already complete)"
            continue
        fi

        TOTAL=$((TOTAL+1))
        log="${LOG_DIR}/pid_${pid}_${version}.log"
        echo "[$(date '+%H:%M:%S')] Extracting pid_${pid}/${version}..."

        if python skill_extraction/extract_skills_v2.py --pid "$pid" --version "$version" > "$log" 2>&1; then
            OK=$((OK+1))
            echo "  OK -> $log"
        else
            FAIL=$((FAIL+1))
            echo "  FAILED -> $log"
            tail -3 "$log"
        fi
    done
done

echo ""
echo "===== Extraction Complete ====="
echo "  Total: $TOTAL  OK: $OK  Skipped: $SKIP  Failed: $FAIL"
