#!/usr/bin/env python3
"""
BundledWebShopping Evolution Evaluation runner — Non-Evolution baseline

Runs the ReAct agent from the decision point without using any evolution
snapshots. Serves as the lower-bound baseline.
"""

import json
import time
import argparse
import traceback
from pathlib import Path

from utils import (
    DATASET_ROOT, RESULTS_ROOT, GAME_LIST,
    load_snapshots, load_dataset_split,
    make_env, restore_game_state, format_prior_subtasks,
    STEP_LIMIT, MAX_CONTEXT_TOKENS,
)
from src.openai_helpers import chat_completion_with_retries, truncate_text
from run_react import (
    ReActBaseline, run_task as _react_run_task,
    REACT_SYSTEM_PROMPT, REACT_FORMAT, parse_shopping_response,
)


def run_task(task: dict, snapshots: dict, baseline: ReActBaseline,
             dataset, output_dir: Path) -> dict:
    result = _react_run_task(task, snapshots, baseline, dataset, output_dir)
    result["baseline"] = "non_evolution"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", required=True, choices=GAME_LIST)
    parser.add_argument("--model", default="openai/gpt-4.1")
    parser.add_argument("--temperature", default=0.4, type=float)
    parser.add_argument("--task_ids", nargs="*")
    parser.add_argument("--distances", nargs="*", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    ee_path = DATASET_ROOT / args.game / "evolution_evaluation.json"
    with open(ee_path) as f:
        ee_data = json.load(f)
    snapshots = load_snapshots()
    dataset = load_dataset_split()

    tasks = ee_data["tasks"]
    if args.task_ids:
        tasks = [t for t in tasks if t["id"] in args.task_ids]
    if args.distances:
        tasks = [t for t in tasks if t["distance"] in args.distances]

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_slug = args.model.replace("/", "_")
        output_dir = RESULTS_ROOT / args.game / "non_evolution" / model_slug / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running Non-Evolution on {args.game} ({len(tasks)} tasks)")
    print(f"Model: {args.model} | Output: {output_dir}")

    baseline = ReActBaseline(model=args.model, temperature=args.temperature)

    results = []
    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] {task['id']} "
              f"(type={task['type']}, dist={task['distance']})")
        try:
            result = run_task(task, snapshots, baseline, dataset, output_dir)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append({"task_id": task["id"], "error": str(e)})

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({
            "game": args.game, "baseline": "non_evolution",
            "model": args.model, "n_tasks": len(tasks), "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nDone. Results saved to {results_path}")


if __name__ == "__main__":
    main()
