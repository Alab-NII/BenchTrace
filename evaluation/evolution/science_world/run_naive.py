#!/usr/bin/env python3
"""
ScienceWorld Evolution Evaluation runner — Naive Concatenation baseline

Concatenates all evolution snapshot trajectories before the test trajectory
(truncated to decision_point), then runs a rolling-scratchpad ReAct episode
from decision_point onwards.

ScienceWorld differences from AlfWorld:
  - env.step(action_str) takes free-form text directly
  - task_name and variation parsed from episode ID
  - Scoring: info["progress"] (0-1) and info["won"]
  - STEP_LIMIT = 100
"""

import json
import time
import argparse
import traceback
from pathlib import Path

from utils import (
    DATASET_ROOT,
    RESULTS_ROOT,
    MAX_CONTEXT_TOKENS,
    STEP_LIMIT,
    GAME_LIST,
    load_snapshots,
    restore_game_state,
    parse_task_and_variation,
    format_trajectory,
)
from src.scienceworld_env import ScienceWorldEnv
from src.openai_helpers import chat_completion_with_retries, truncate_text

from run_react import (
    REACT_SYSTEM_PROMPT,
    REACT_FORMAT,
    parse_react_response,
)


# ---------------------------------------------------------------------------
# Naive Concatenation baseline
# ---------------------------------------------------------------------------

class NaiveConcatBaseline:
    def __init__(self, model: str, temperature: float = 0.4):
        self.model = model
        self.temperature = temperature

    def build_primer(
        self,
        evolution_trajectories: list[tuple[str, list[dict]]],
        pre_dp_traj: list[dict],
    ) -> str:
        parts = []
        for i, (ep_id, traj) in enumerate(evolution_trajectories):
            parts.append(format_trajectory(traj, header=f"Past Episode {i+1} (id: {ep_id})"))

        pre_dp_lines = []
        for s in pre_dp_traj:
            pre_dp_lines.append(f"Obs {s['step']}: {s['obs']}")
            if s.get("action"):
                pre_dp_lines.append(f"Act {s['step']}: {s['action']}")
        parts.append(
            "=== Current Episode History (before decision point) ===\n"
            + "\n".join(pre_dp_lines)
        )
        return truncate_text("\n\n".join(parts), MAX_CONTEXT_TOKENS)

    def make_agent_fn(self, primer: str):
        scratchpad: list[str] = []

        def agent_fn(obs: str, step: int):
            scratchpad.append(f"Obs {step}: {obs}")

            user_prompt = primer + "\n\n"
            user_prompt += "=== Current Episode ===\n"
            user_prompt += "\n".join(scratchpad)
            user_prompt += REACT_FORMAT

            res = chat_completion_with_retries(
                model=self.model,
                sys_prompt=REACT_SYSTEM_PROMPT,
                prompt=truncate_text(user_prompt, MAX_CONTEXT_TOKENS),
                max_tokens=256,
                temperature=self.temperature,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = res.choices[0].message.content if res and res.choices else ""
            thought, action = parse_react_response(raw)

            scratchpad.append(f"Thought {step}: {thought}")
            scratchpad.append(f"Act {step}: {action}")
            return action, raw

        return agent_fn


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(task: dict, snapshots: dict, baseline: NaiveConcatBaseline,
             output_dir: Path) -> dict:
    task_id = task["id"]
    game = task["game"]
    decision_point = task["test_snapshot"]["decision_point"]
    test_ep_id = task["test_snapshot"]["episode_id"]
    evolution_ep_ids = task["evolution_snapshots"]

    test_ep = snapshots[test_ep_id]
    test_trajectory = test_ep["snapshot"]["trajectory"]
    pre_dp_traj = test_trajectory[:decision_point]

    evolution_trajs = [
        (ep_id, snapshots[ep_id]["snapshot"]["trajectory"])
        for ep_id in evolution_ep_ids
    ]
    primer = baseline.build_primer(evolution_trajs, pre_dp_traj)

    task_name, variation = parse_task_and_variation(test_ep_id)
    env = ScienceWorldEnv(task_name, variation, step_limit=STEP_LIMIT)

    ob, info = restore_game_state(env, test_trajectory, decision_point)

    log_path = output_dir / f"{task_id}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Task: {task_id}\n")
        f.write(f"Baseline: naive_concat\n")
        f.write(f"Game: {game}, Decision point: {decision_point}\n")
        f.write(f"Test episode: {test_ep_id}\n")
        f.write(f"Evolution snapshots: {evolution_ep_ids}\n\n")
        f.write(f"=== Restored state at step {decision_point} ===\n")
        f.write(f"OBS: {ob}\n\n")

    agent_fn = baseline.make_agent_fn(primer)
    trajectory = []
    cur_ob, cur_info = ob, info

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            for step in range(decision_point, STEP_LIMIT):
                action, raw = agent_fn(cur_ob, step)

                f.write(f"[Step {step}] OBS: {cur_ob[:120]}\n")
                f.write(f"           RAW: {raw[:200]}\n")
                f.write(f"           ACTION: {action}\n")

                trajectory.append({
                    "step": step,
                    "obs": cur_ob,
                    "action": action,
                    "raw_response": raw,
                })

                cur_ob, done, cur_info = env.step(action)
                trajectory[-1]["progress_after"] = cur_info.get("progress", 0.0)
                trajectory[-1]["won"] = cur_info.get("won", False)

                print(
                    f"  [step {step}] {action[:30]:30s} "
                    f"→ progress={cur_info.get('progress', 0):.3f}, "
                    f"won={cur_info.get('won', False)}"
                )

                if done:
                    break
    finally:
        env.close()

    first_step = trajectory[0] if trajectory else {}
    last_step = trajectory[-1] if trajectory else {}
    return {
        "task_id": task_id,
        "baseline": "naive_concat",
        "game": game,
        "type": task["type"],
        "distance": task["distance"],
        "target_failure_instance": task["target_failure_instance"],
        "test_episode_id": test_ep_id,
        "decision_point": decision_point,
        "scoring_method": task["scoring_method"],
        "action_at_decision_point": first_step.get("action"),
        "obs_after_decision_point": first_step.get("obs"),
        "final_progress": last_step.get("progress_after"),
        "won": last_step.get("won", False),
        "trajectory_from_decision_point": trajectory,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
        output_dir = RESULTS_ROOT / args.game / "naive_concat" / model_slug / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running Naive Concat on {args.game} ({len(tasks)} tasks)")
    print(f"Model: {args.model}")
    print(f"Output: {output_dir}")

    baseline = NaiveConcatBaseline(model=args.model, temperature=args.temperature)

    results = []
    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] {task['id']} "
              f"(type={task['type']}, dist={task['distance']})")
        try:
            result = run_task(task, snapshots, baseline, output_dir)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append({"task_id": task["id"], "error": str(e)})

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({
            "game": args.game,
            "baseline": "naive_concat",
            "model": args.model,
            "n_tasks": len(tasks),
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Results saved to {results_path}")


if __name__ == "__main__":
    main()
