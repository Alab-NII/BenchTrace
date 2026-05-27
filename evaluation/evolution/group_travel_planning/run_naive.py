#!/usr/bin/env python3
"""
GroupTravelPlanning Evolution Evaluation runner — Naive Concatenation baseline

Concatenates all evolution snapshot trajectories before the pre-decision-point
context of the test episode, then runs the ReAct plan-generation loop from the
decision point onwards.

GroupTravelPlanning-specific notes:
  - One env.step() = one traveler's complete plan (free-text).
  - Evolution snapshot trajectories use odd-step entries to hold the actual
    traveler request + generated plan; even-step entries only carry the env
    observation header.
  - Truncate the concatenated primer aggressively — each evolution episode can
    span thousands of tokens of traveler request + plan.
"""

import re
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
    load_dataset_split,
    make_env,
    restore_game_state,
    format_prior_plans,
)
from src.openai_helpers import chat_completion_with_retries, truncate_text
from run_react import REACT_SYSTEM_PROMPT, REACT_FORMAT, parse_travel_response


def _format_episode_for_primer(ep_id: str, trajectory: list[dict],
                               final_score: float) -> str:
    """Render one evolution episode as a compact (request, plan, score) listing."""
    lines = [f"=== Past Episode (id: {ep_id}) ==="]
    for step in trajectory:
        if step.get("action") is None:
            continue
        idx = step.get("subtask_idx", step.get("step", 0) // 2)
        question = step.get("obs", "")
        plan = step.get("action", "")
        plan_clean = re.sub(r"<think>.*?</think>", "", plan, flags=re.DOTALL).strip()
        prog = float(step.get("subtask_progress", 0) or 0)
        lines.append(f"[Traveler {idx + 1}]")
        lines.append(f"Request: {question[:400]}")
        lines.append(f"Plan: {plan_clean[:600]}")
        lines.append(f"Constraint satisfaction: {prog:.0%}")
        lines.append("")
    lines.append(f"Final score: {final_score:.2f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Naive Concatenation baseline
# ---------------------------------------------------------------------------

class NaiveConcatBaseline:
    """Concatenate every evolution episode + pre-dp context as a fixed primer."""

    def __init__(self, model: str, temperature: float = 0.4):
        self.model = model
        self.temperature = temperature

    def build_primer(self, evolution_ep_ids: list[str], snapshots: dict,
                     pre_dp_context: str) -> str:
        parts = []
        for ep_id in evolution_ep_ids:
            ep = snapshots.get(ep_id)
            if ep is None:
                continue
            traj = ep["snapshot"]["trajectory"]
            score = float(ep["snapshot"].get("final_score", 0) or 0)
            parts.append(_format_episode_for_primer(ep_id, traj, score))
        if pre_dp_context:
            parts.append(
                "=== Current Episode — Prior Travelers' Plans (before decision point) ===\n"
                + pre_dp_context
            )
        return truncate_text("\n\n".join(parts), MAX_CONTEXT_TOKENS)

    def make_agent_fn(self, primer: str):
        current_run_plans: list[tuple[str, str]] = []

        def agent_fn(obs: str, subtask_idx: int):
            user_prompt = primer + "\n\n"
            if current_run_plans:
                user_prompt += "=== Plans Generated in This Run ===\n"
                for i, (q, p) in enumerate(current_run_plans):
                    n = subtask_idx - len(current_run_plans) + i + 1
                    user_prompt += f"Traveler {n + 1}: {q[:200]}\nPlan: {p[:400]}\n\n"
            user_prompt += "=== Current Traveler ===\n" + obs + REACT_FORMAT

            res = chat_completion_with_retries(
                model=self.model,
                sys_prompt=REACT_SYSTEM_PROMPT,
                prompt=truncate_text(user_prompt, MAX_CONTEXT_TOKENS),
                max_tokens=1024,
                temperature=self.temperature,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = res.choices[0].message.content if res and res.choices else ""
            action = parse_travel_response(raw)
            current_run_plans.append((obs[:200], action))
            return action, raw

        return agent_fn


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(task: dict, snapshots: dict, baseline: NaiveConcatBaseline,
             judge_model: str, dataset, output_dir: Path) -> dict:
    task_id = task["id"]
    game = task["game"]
    decision_point = task["test_snapshot"]["decision_point"]
    test_ep_id = task["test_snapshot"]["episode_id"]
    evolution_ep_ids = task["evolution_snapshots"]

    test_trajectory = snapshots[test_ep_id]["snapshot"]["trajectory"]
    pre_dp_context = format_prior_plans(test_trajectory, decision_point)
    primer = baseline.build_primer(evolution_ep_ids, snapshots, pre_dp_context)

    env = make_env(test_ep_id, judge_model, dataset)
    ob, info = restore_game_state(env, test_trajectory, decision_point)

    log_path = output_dir / f"{task_id}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Task: {task_id}\nBaseline: naive_concat\n")
        f.write(f"Game: {game}, Decision point: {decision_point}\n")
        f.write(f"Test episode: {test_ep_id}\n")
        f.write(f"Evolution snapshots: {evolution_ep_ids}\n\n")
        f.write(f"=== Primer (first 600 chars) ===\n{primer[:600]}\n\n")
        f.write(f"=== Starting traveler {decision_point // 2} ===\n{ob[:300]}\n\n")

    agent_fn = baseline.make_agent_fn(primer)
    trajectory = []
    cur_ob, cur_info = ob, info
    start_subtask = decision_point // 2

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            for subtask_step in range(start_subtask, start_subtask + STEP_LIMIT):
                action, raw = agent_fn(cur_ob, subtask_step)

                f.write(f"[Traveler {subtask_step}] OBS: {cur_ob[:100]}\n")
                f.write(f"               RAW: {raw[:150]}\n")
                f.write(f"               ACTION: {action[:200]}\n")

                trajectory.append({
                    "subtask_idx": subtask_step,
                    "obs": cur_ob,
                    "action": action,
                    "raw_response": raw,
                })

                cur_ob, done, cur_info = env.step(action)
                trajectory[-1]["progress_after"] = cur_info.get("progress", 0.0)
                trajectory[-1]["won"] = cur_info.get("won", False)

                print(
                    f"  [traveler {subtask_step}] "
                    f"progress={cur_info.get('progress', 0):.3f}, "
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
        "obs_at_decision_point": first_step.get("obs"),
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
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_slug = args.model.replace("/", "_")
        output_dir = RESULTS_ROOT / args.game / "naive_concat" / model_slug / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running Naive Concat on {args.game} ({len(tasks)} tasks)")
    print(f"Model: {args.model} | Judge: {judge_model} | Output: {output_dir}")

    baseline = NaiveConcatBaseline(model=args.model, temperature=args.temperature)

    results = []
    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] {task['id']} "
              f"(type={task['type']}, dist={task['distance']})")
        try:
            result = run_task(task, snapshots, baseline, judge_model, dataset, output_dir)
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
            "judge_model": judge_model,
            "n_tasks": len(tasks),
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Results saved to {results_path}")


if __name__ == "__main__":
    main()
