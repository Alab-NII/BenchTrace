#!/bin/bash
# GPT-4.1 Evolution Evaluation — AlfWorld, 7 baselines, full dataset
# Tasks run in parallel within each baseline; baselines are sequential.
# Usage: nohup bash run_gpt41_evaleval.sh > logs/run_gpt41_evaleval.log 2>&1 &

set -e

MODEL="gpt-4.1"
RUNNER_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$RUNNER_DIR/../.." && pwd)"
OUTPUT_ROOT="$PROJECT_ROOT/main_result/AlfWorld"
TASKS="pick_and_place look_at_obj pick_clean pick_heat pick_cool pick_two"

# OpenAI API (real endpoint, no vLLM)
export OPENAI_API_KEY="$(python3 -c "import json; print(json.load(open('$PROJECT_ROOT/api_key.json'))['openai_api_key'])")"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # prevent load_dotenv() from overriding with .env's localhost value

mkdir -p "$RUNNER_DIR/logs"

run_baseline() {
    local baseline=$1
    local runner=$2
    shift 2
    local extra_args=("$@")

    echo "========================================"
    echo "BASELINE: $baseline"
    echo "========================================"

    local pids=()
    local max_concurrency=6
    for task in $TASKS; do
        local output_dir="$OUTPUT_ROOT/$task/${baseline}_gpt41"
        mkdir -p "$output_dir"
        if [ -f "$output_dir/results.json" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP  $baseline / $task (results.json exists)"
            continue
        fi
        # Wait if at concurrency limit
        while [ "${#pids[@]}" -ge "$max_concurrency" ]; do
            for i in "${!pids[@]}"; do
                if ! kill -0 "${pids[$i]}" 2>/dev/null; then
                    unset "pids[$i]"
                    pids=("${pids[@]}")
                fi
            done
            [ "${#pids[@]}" -ge "$max_concurrency" ] && sleep 5
        done
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $baseline / $task"
        conda run -n Fraud python3 "$RUNNER_DIR/$runner" \
            --task "$task" \
            --model "$MODEL" \
            --distances 1 5 \
            --output_dir "$output_dir" \
            "${extra_args[@]}" &
        pids+=($!)
    done

    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE  $baseline (all tasks)"
    echo
}

run_baseline react      run_react.py
run_baseline rag        run_rag.py
run_baseline remem      run_remem.py
run_baseline memrl      run_memrl.py
run_baseline reflexion  run_reflexion.py
run_baseline autoskill  run_autoskill.py
run_baseline evotest    run_evotest.py  --evo_model "$MODEL"

echo "========================================"
echo "ALL DONE: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
