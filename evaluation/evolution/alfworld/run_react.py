#!/usr/bin/env python3
"""
AlfWorld Evolution Evaluation runner — ReAct baseline

ReAct (Yao et al., 2022): interleaves Thought and Act at every step.
No evolution snapshots: agent relies solely on in-context reasoning.

AlfWorld differences from JTTL version:
  - env.step() returns (obs, done, info) — no reward in tuple
  - Scoring: info["progress"] (0-1) and info["won"], not integer score
  - No inventory field
  - restore_game_state replays trajectory[1:dp+1] (step 0 action is None)
  - Each task may use a different game_file (looked up at runtime)
"""

import os
import json
import time
import argparse
import re
from pathlib import Path

from utils import (
    DATASET_ROOT,
    RESULTS_ROOT,
    MAX_CONTEXT_TOKENS,
    STEP_LIMIT,
    TASK_SHORT_LIST,
    load_snapshots,
    restore_game_state,
    get_game_file,
    format_trajectory,
)
from src.alfworld_env import AlfWorldEnv
from src.openai_helpers import chat_completion_with_retries, truncate_text

# ---------------------------------------------------------------------------
# ReAct prompts
# ---------------------------------------------------------------------------

REACT_SYSTEM_PROMPT = (
    "You are an expert at completing household tasks in a text-based environment. "
    "At each step you will be given the current observation. "
    "Respond with a Thought (one sentence of reasoning) "
    "followed by an Act (the exact command to execute). "
    "Do NOT repeat a failed action. Try different approaches when stuck."
)

REACT_FORMAT = (
    "\nRespond using EXACTLY this format (two lines, nothing else):\n"
    "Thought: <one sentence of reasoning>\n"
    "Act: <exact command>\n"
)


def parse_react_response(response: str) -> tuple[str, str]:
    thought = ""
    action = "look"
    if not response:
        return thought, action
    for line in response.strip().splitlines():
        m = re.match(r"Thought:\s*(.+)", line, re.IGNORECASE)
        if m:
            thought = m.group(1).strip()
        m = re.match(r"Act:\s*(.+)", line, re.IGNORECASE)
        if m:
            action = m.group(1).strip()
    return thought, action

# ---------------------------------------------------------------------------
# ReAct agent
# ---------------------------------------------------------------------------

class ReActBaseline:
    def __init__(self, model: str, temperature: float = 0.4):
        self.model = model
        self.temperature = temperature

    def make_agent_fn(self, primer: str):
        scratchpad: list[str] = []

        def agent_fn(obs: str, step: int):
            scratchpad.append(f"Obs {step}: {obs}")

            user_prompt = ""
            if primer:
                user_prompt += "=== Episode History (before decision point) ===\n"
                user_prompt += primer + "\n\n"
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

def run_task(task: dict, snapshots: dict, baseline: ReActBaseline,
             output_dir: Path) -> dict:
    task_id = task["id"]
    task_short = task["task"]
    decision_point = task["test_snapshot"]["decision_point"]
    test_ep_id = task["test_snapshot"]["episode_id"]

    test_ep = snapshots[test_ep_id]
    test_trajectory = test_ep["snapshot"]["trajectory"]
    pre_dp_traj = test_trajectory[:decision_point]

    # Build primer from pre-dp history
    primer_lines = []
    for s in pre_dp_traj:
        primer_lines.append(f"Obs {s['step']}: {s['obs']}")
        if s.get("action"):
            primer_lines.append(f"Act {s['step']}: {s['action']}")
    primer = "\n".join(primer_lines)

    # Create env for this specific game instance
    game_file = get_game_file(test_ep_id)
    env = AlfWorldEnv(game_file=game_file, step_limit=STEP_LIMIT)

    ob, info = restore_game_state(env, test_trajectory, decision_point)

    log_path = output_dir / f"{task_id}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Task: {task_id}\n")
        f.write(f"Baseline: react\n")
        f.write(f"Task type: {task_short}, Decision point: {decision_point}\n")
        f.write(f"Test episode: {test_ep_id}\n\n")
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
                trajectory[-1]["progress_strict_after"] = cur_info.get("progress_strict", 0.0)
                trajectory[-1]["progress_lenient_after"] = cur_info.get("progress_lenient", 0.0)
                trajectory[-1]["won"] = cur_info.get("won", False)

                print(
                    f"  [step {step}] {action[:30]:30s} "
                    f"→ progress={cur_info.get('progress',0):.3f}, "
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
        "task": task_short,
        "type": task["type"],
        "distance": task["distance"],
        "target_failure_instance": task["target_failure_instance"],
        "test_episode_id": test_ep_id,
        "decision_point": decision_point,
        "scoring_method": task["scoring_method"],
        "action_at_decision_point": first_step.get("action"),
        "obs_after_decision_point": first_step.get("obs"),
        "final_progress": last_step.get("progress_after"),
        "final_progress_strict": last_step.get("progress_strict_after"),
        "final_progress_lenient": last_step.get("progress_lenient_after"),
        "won": last_step.get("won", False),
        "trajectory_from_decision_point": trajectory,
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASK_SHORT_LIST)
    parser.add_argument("--model", default="openai/gpt-4.1")
    parser.add_argument("--temperature", default=0.4, type=float)
    parser.add_argument("--task_ids", nargs="*", help="Specific task IDs (default: all)")
    parser.add_argument("--distances", nargs="*", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    ee_path = DATASET_ROOT / args.task / "evolution_evaluation.json"
    with open(ee_path) as f:
        ee_data = json.load(f)
    snapshots = load_snapshots()  # load all task types for cross-task snapshots

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
        output_dir = RESULTS_ROOT / args.task / "react" / model_slug / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running ReAct on {args.task} ({len(tasks)} tasks)")
    print(f"Model: {args.model}")
    print(f"Output: {output_dir}")

    baseline = ReActBaseline(model=args.model, temperature=args.temperature)

    results = []
    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] {task['id']} "
              f"(type={task['type']}, dist={task['distance']})")
        try:
            result = run_task(task, snapshots, baseline, output_dir)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            results.append({"task_id": task["id"], "error": str(e)})

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({
            "task": args.task,
            "baseline": "react",
            "model": args.model,
            "n_tasks": len(tasks),
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Results saved to {results_path}")


if __name__ == "__main__":
    main()
