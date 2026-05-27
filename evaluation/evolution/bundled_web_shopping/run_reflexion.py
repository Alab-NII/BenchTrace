#!/usr/bin/env python3
"""
BundledWebShopping Evolution Evaluation runner — Reflexion baseline

Reflexion (Shinn et al., 2023): after each failed episode, generate a verbal
self-reflection. Evolution snapshots serve as past failed trials.
Reflections are prepended as "Lessons from Past Episodes" in the prompt.
"""

import re
import json
import time
import argparse
import traceback
from pathlib import Path

from utils import (
    DATASET_ROOT, RESULTS_ROOT, GAME_LIST, STEP_LIMIT, MAX_CONTEXT_TOKENS,
    load_snapshots, load_dataset_split, make_env,
    restore_game_state, format_prior_subtasks,
)
from src.openai_helpers import chat_completion_with_retries, truncate_text
from run_react import REACT_SYSTEM_PROMPT, REACT_FORMAT, parse_shopping_response

# ---------------------------------------------------------------------------
# Reflection generation
# ---------------------------------------------------------------------------

REFLECTION_SYSTEM_PROMPT = (
    "You are an expert analyst of bundled web shopping playthroughs. "
    "Identify key mistakes and actionable lessons from a past episode."
)

REFLECTION_PROMPT_TEMPLATE = """\
Below is a bundled web shopping episode. Each subtask required selecting
one compatible product from a list of options.

{trajectory}

Final score: {final_score:.2f} (1.0 = all subtasks correct)

Write a concise reflection (2-3 sentences):
1. What went wrong — which selection criterion was misunderstood or overlooked.
2. What concrete strategy should be applied next time to make correct selections.

Be specific. Focus on actionable lessons, not trajectory summaries.
"""


def _format_traj_for_reflection(trajectory: list[dict]) -> str:
    lines = []
    for step in trajectory:
        if step.get("action") is None:
            continue
        idx = step.get("subtask_idx", step.get("step", 0) // 2)
        obs = step["obs"][:200]
        action = step.get("action", "")
        correct = step.get("correct", None)
        lines.append(f"[Subtask {idx + 1}] {obs[:100]}")
        lines.append(f"  Selected: {action[:80]}  Result: {'✓' if correct else '✗'}")
    return "\n".join(lines)


def generate_reflection(episode: dict, model: str, temperature: float = 0.0) -> str:
    traj = episode["snapshot"]["trajectory"]
    final_score = float(episode["snapshot"].get("final_score", 0) or 0)
    prompt = REFLECTION_PROMPT_TEMPLATE.format(
        trajectory=_format_traj_for_reflection(traj),
        final_score=final_score,
    )
    res = chat_completion_with_retries(
        model=model, sys_prompt=REFLECTION_SYSTEM_PROMPT, prompt=prompt,
        max_tokens=200, temperature=temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return res.choices[0].message.content.strip() if res and res.choices else "(unavailable)"


# ---------------------------------------------------------------------------
# Reflexion agent
# ---------------------------------------------------------------------------

class ReflexionBaseline:
    def __init__(self, model: str, temperature: float = 0.4):
        self.model = model
        self.temperature = temperature

    def build_reflections(self, evolution_ep_ids: list[str], snapshots: dict) -> list[tuple[str, str]]:
        reflections = []
        for ep_id in evolution_ep_ids:
            ep = snapshots.get(ep_id)
            if ep is None:
                continue
            print(f"    [reflexion] reflecting on {ep_id} ...", end=" ", flush=True)
            ref = generate_reflection(ep, self.model)
            print("done")
            reflections.append((ep_id, ref))
        return reflections

    def make_agent_fn(self, reflections: list[tuple[str, str]], pre_dp_context: str):
        current_run_selections: list[str] = []

        def agent_fn(obs: str, subtask_idx: int):
            user_prompt = ""
            if reflections:
                lines = ["=== Lessons from Past Episodes ==="]
                for i, (ep_id, ref) in enumerate(reflections, 1):
                    lines.append(f"[Lesson {i}] {ref}")
                user_prompt += "\n".join(lines) + "\n\n"
            if pre_dp_context:
                user_prompt += "=== Prior Subtask Selections (this episode) ===\n"
                user_prompt += pre_dp_context + "\n\n"
            if current_run_selections:
                user_prompt += "=== Selections Made in This Run ===\n"
                user_prompt += "\n".join(current_run_selections) + "\n\n"
            user_prompt += "=== Current Subtask ===\n" + obs + REACT_FORMAT

            res = chat_completion_with_retries(
                model=self.model, sys_prompt=REACT_SYSTEM_PROMPT,
                prompt=truncate_text(user_prompt, MAX_CONTEXT_TOKENS),
                max_tokens=512, temperature=self.temperature,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = res.choices[0].message.content if res and res.choices else ""
            action = parse_shopping_response(raw)
            current_run_selections.append(f"Subtask {subtask_idx + 1}: Selected {action}")
            return action, raw

        return agent_fn


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(task: dict, snapshots: dict, baseline: ReflexionBaseline,
             dataset, output_dir: Path) -> dict:
    task_id = task["id"]
    game = task["game"]
    decision_point = task["test_snapshot"]["decision_point"]
    test_ep_id = task["test_snapshot"]["episode_id"]
    evolution_ep_ids = task["evolution_snapshots"]

    test_ep = snapshots[test_ep_id]
    test_trajectory = test_ep["snapshot"]["trajectory"]
    pre_dp_context = format_prior_subtasks(test_trajectory, decision_point)

    reflections = baseline.build_reflections(evolution_ep_ids, snapshots)

    env = make_env(test_ep_id, dataset)
    ob, info = restore_game_state(env, test_trajectory, decision_point)

    log_path = output_dir / f"{task_id}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Task: {task_id}\nBaseline: reflexion\n")
        f.write(f"Game: {game}, Decision point: {decision_point}\n")
        f.write(f"Test episode: {test_ep_id}\nEvolution: {evolution_ep_ids}\n\n")
        for ep_id, ref in reflections:
            f.write(f"[{ep_id}] {ref}\n\n")

    agent_fn = baseline.make_agent_fn(reflections, pre_dp_context)
    trajectory = []
    cur_ob, cur_info = ob, info
    start_subtask = decision_point // 2

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            for subtask_step in range(start_subtask, start_subtask + STEP_LIMIT):
                action, raw = agent_fn(cur_ob, subtask_step)
                f.write(f"[Subtask {subtask_step}] ACTION: {action}\n")
                trajectory.append({
                    "subtask_idx": subtask_step, "obs": cur_ob,
                    "action": action, "raw_response": raw,
                })
                cur_ob, done, cur_info = env.step(action)
                trajectory[-1]["progress_after"] = cur_info.get("progress", 0.0)
                trajectory[-1]["won"] = cur_info.get("won", False)
                print(f"  [subtask {subtask_step}] {action:6s} → progress={cur_info.get('progress',0):.3f}")
                if done:
                    break
    finally:
        env.close()

    first_step = trajectory[0] if trajectory else {}
    last_step = trajectory[-1] if trajectory else {}
    return {
        "task_id": task_id, "baseline": "reflexion", "game": game,
        "type": task["type"], "distance": task["distance"],
        "target_failure_instance": task["target_failure_instance"],
        "test_episode_id": test_ep_id, "decision_point": decision_point,
        "scoring_method": task["scoring_method"],
        "action_at_decision_point": first_step.get("action"),
        "obs_at_decision_point": first_step.get("obs"),
        "final_progress": last_step.get("progress_after"),
        "won": last_step.get("won", False),
        "reflections": [{"ep_id": ep_id, "reflection": ref} for ep_id, ref in reflections],
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
        output_dir = RESULTS_ROOT / args.game / "reflexion" / model_slug / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running Reflexion on {args.game} ({len(tasks)} tasks)")
    baseline = ReflexionBaseline(model=args.model, temperature=args.temperature)

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
            "game": args.game, "baseline": "reflexion", "model": args.model,
            "n_tasks": len(tasks), "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nDone. Results saved to {results_path}")


if __name__ == "__main__":
    main()
