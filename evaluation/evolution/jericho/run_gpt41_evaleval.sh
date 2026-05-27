#!/bin/bash
# GPT-4.1 Evolution Evaluation — Jericho, 7 baselines, 1/3 sampled tasks
# Games run in parallel within each baseline; baselines are sequential.
# Usage: nohup bash run_gpt41_evaleval.sh > logs/run_gpt41_evaleval.log 2>&1 &

set -e

MODEL="gpt-4.1"
RUNNER_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$RUNNER_DIR/../.." && pwd)"
OUTPUT_ROOT="$PROJECT_ROOT/main_result/Jericho"
GAMES="balances detective library temple zork1 zork3"

# OpenAI API (real endpoint, no vLLM)
export OPENAI_API_KEY="$(python3 -c "import json; print(json.load(open('$PROJECT_ROOT/api_key.json'))['openai_api_key'])")"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # prevent load_dotenv() from overriding with .env's localhost value

# 1/3 sampled task IDs (stratified by game × type × distance)
TASK_IDS_FILE="$RUNNER_DIR/gpt41_sampled_task_ids.json"
TASK_IDS=$(python3 -c "import json; ids=json.load(open('$TASK_IDS_FILE')); print(' '.join(ids))")

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
    local concurrency=0
    local max_concurrency=6
    for game in $GAMES; do
        local output_dir="$OUTPUT_ROOT/$game/${baseline}_gpt41"
        mkdir -p "$output_dir"
        if [ -f "$output_dir/results.json" ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIP  $baseline / $game (results.json exists)"
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
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $baseline / $game"
        conda run -n Fraud python3 "$RUNNER_DIR/$runner" \
            --game "$game" \
            --model "$MODEL" \
            --distances 1 5 \
            --output_dir "$output_dir" \
            --task_ids $TASK_IDS \
            "${extra_args[@]}" &
        pids+=($!)
    done

    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE  $baseline (all games)"
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
