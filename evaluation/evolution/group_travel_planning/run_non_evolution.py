#!/usr/bin/env python3
"""
GroupTravelPlanning Evolution Evaluation runner — Non-Evolution baseline

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
)
from run_react import ReActBaseline, run_task as _react_run_task


def run_task(task, snapshots, baseline, judge_model, dataset, output_dir):
    result = _react_run_task(task, snapshots, baseline, judge_model, dataset, output_dir)
    result["baseline"] = "non_evolution"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", required=True, choices=GAME_LIST)
    parser.add_argument("--model", default="openai/gpt-4.1")
    parser.add_argument("--judge_model", default=None)
    parser.add_argument("--temperature", default=0.4, type=float)
    parser.add_argument("--task_ids", nargs="*")
    parser.add_argument("--distances", nargs="*", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    judge_model = args.judge_model or args.model

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
        import time as _time
        timestamp = _time.strftime("%Y%m%d-%H%M%S")
        from utils import RESULTS_ROOT
        model_slug = args.model.replace("/", "_")
        output_dir = RESULTS_ROOT / args.game / "non_evolution" / model_slug / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running Non-Evolution on {args.game} ({len(tasks)} tasks)")
    baseline = ReActBaseline(model=args.model, temperature=args.temperature)

    results = []
    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] {task['id']} "
              f"(type={task['type']}, dist={task['distance']})")
        try:
            result = run_task(task, snapshots, baseline, judge_model, dataset, output_dir)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            results.append({"task_id": task["id"], "error": str(e)})

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({
            "game": args.game, "baseline": "non_evolution",
            "model": args.model, "judge_model": judge_model,
            "n_tasks": len(tasks), "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nDone. Results saved to {results_path}")


if __name__ == "__main__":
    main()
