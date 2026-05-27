#!/usr/bin/env python3
"""
BundledWebShopping Evolution Evaluation runner — ReAct baseline

ReAct (Yao et al., 2022): interleaves Thought and Selection at every subtask.
No evolution snapshots used: agent relies solely on in-context reasoning.

BundledWebShopping differences from step-based environments:
  - One env.step() = one complete subtask (product selection)
  - Agent outputs a selection letter [A/B/C/...], not a free-form action
  - Prior subtask selections shown as context (compatibility constraints)
  - env.step() returns (obs, done, info) with info["correct"] per subtask
  - decision_point//2 = failing subtask index; restore via direct state setting
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
    format_prior_subtasks,
)
from src.openai_helpers import chat_completion_with_retries, truncate_text

# ---------------------------------------------------------------------------
# ReAct prompts for bundled shopping
# ---------------------------------------------------------------------------

REACT_SYSTEM_PROMPT = (
    "You are an intelligent shopping agent selecting compatible products in a bundled "
    "web shopping task. Each subtask asks you to select one product from a list of options. "
    "Products in the bundle must be technically compatible and fit within the total budget. "
    "Respond with a Thought (one sentence of reasoning about compatibility and budget) "
    "followed by your Selection letter."
)

REACT_FORMAT = (
    "\nRespond using EXACTLY this format (two lines, nothing else):\n"
    "Thought: <one sentence of reasoning about which option best fits the requirements>\n"
    "Selection: [<LETTER>]\n"
)


def parse_shopping_response(response: str) -> str:
    """Extract the selection letter from the agent's response."""
    if not response:
        return "[A]"
    text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    for pat in [
        r"Selection:\s*\[([A-Za-z])\]",
        r"\[([A-Za-z])\]",
        r"\bselect\s+\[?([A-Za-z])\]?",
        r"\bchoose\s+\[?([A-Za-z])\]?",
        r"Option\s+([A-Za-z])\b",
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return f"[{m.group(1).upper()}]"
    return "[A]"


# ---------------------------------------------------------------------------
# ReAct agent
# ---------------------------------------------------------------------------

class ReActBaseline:
    def __init__(self, model: str, temperature: float = 0.4):
        self.model = model
        self.temperature = temperature

    def make_agent_fn(self, pre_dp_context: str):
        current_run_selections: list[str] = []

        def agent_fn(obs: str, subtask_idx: int):
            user_prompt = ""
            if pre_dp_context:
                user_prompt += "=== Prior Subtask Selections (this episode) ===\n"
                user_prompt += pre_dp_context + "\n\n"
            if current_run_selections:
                user_prompt += "=== Selections Made in This Run ===\n"
                user_prompt += "\n".join(current_run_selections) + "\n\n"
            user_prompt += "=== Current Subtask ===\n"
            user_prompt += obs
            user_prompt += REACT_FORMAT

            res = chat_completion_with_retries(
                model=self.model,
                sys_prompt=REACT_SYSTEM_PROMPT,
                prompt=truncate_text(user_prompt, MAX_CONTEXT_TOKENS),
                max_tokens=512,
                temperature=self.temperature,
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

def run_task(task: dict, snapshots: dict, baseline: ReActBaseline,
             dataset, output_dir: Path) -> dict:
    task_id = task["id"]
    game = task["game"]
    decision_point = task["test_snapshot"]["decision_point"]
    test_ep_id = task["test_snapshot"]["episode_id"]

    test_ep = snapshots[test_ep_id]
    test_trajectory = test_ep["snapshot"]["trajectory"]
    pre_dp_context = format_prior_subtasks(test_trajectory, decision_point)

    env = make_env(test_ep_id, dataset)
    ob, info = restore_game_state(env, test_trajectory, decision_point)

    log_path = output_dir / f"{task_id}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Task: {task_id}\nBaseline: react\n")
        f.write(f"Game: {game}, Decision point: {decision_point}\n")
        f.write(f"Test episode: {test_ep_id}\n\n")
        if pre_dp_context:
            f.write(f"=== Prior context ===\n{pre_dp_context}\n\n")
        f.write(f"=== Starting subtask {decision_point // 2} ===\n{ob[:200]}\n\n")

    agent_fn = baseline.make_agent_fn(pre_dp_context)
    trajectory = []
    cur_ob, cur_info = ob, info
    start_subtask = decision_point // 2

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            for subtask_step in range(start_subtask, start_subtask + STEP_LIMIT):
                action, raw = agent_fn(cur_ob, subtask_step)

                f.write(f"[Subtask {subtask_step}] OBS: {cur_ob[:100]}\n")
                f.write(f"              RAW: {raw[:150]}\n")
                f.write(f"              ACTION: {action}\n")

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
                    f"  [subtask {subtask_step}] {action:6s} "
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
        "baseline": "react",
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
        output_dir = RESULTS_ROOT / args.game / "react" / model_slug / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running ReAct on {args.game} ({len(tasks)} tasks)")
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
            "game": args.game,
            "baseline": "react",
            "model": args.model,
            "n_tasks": len(tasks),
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Results saved to {results_path}")


if __name__ == "__main__":
    main()
