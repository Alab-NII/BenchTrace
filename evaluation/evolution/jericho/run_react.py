#!/usr/bin/env python3
"""
Evolution Evaluation runner - ReAct baseline

ReAct (Yao et al., 2022): interleaves Thought and Act at every step.
  - Each step: agent emits "Thought: ..." then "Act: ..." in one LLM call
  - Observation fed back as next input
  - No evolution snapshots: agent relies solely on in-context reasoning

Format (mirrors the original ReAct paper / alfworld notebook):
    Thought: <reasoning about current state>
    Act: <game command>

Reference: https://github.com/ysymyth/ReAct
"""

import os
import sys
import re
import json
import time
import argparse
from pathlib import Path

EVOTEST_PATH = Path(__file__).parent.parent / "EvoTest"
sys.path.insert(0, str(EVOTEST_PATH))
from src.env import JerichoEnv
from src.openai_helpers import chat_completion_with_retries, truncate_text
from src.utils import game_file

from utils import (
    DATASET_ROOT,
    ROM_DIR,
    RESULTS_ROOT,
    MAX_CONTEXT_TOKENS,
    STEP_LIMIT,
    GAMES,
    load_snapshots,
    restore_game_state,
)

# ---------------------------------------------------------------------------
# ReAct system prompt
# ---------------------------------------------------------------------------

REACT_SYSTEM_PROMPT = (
    "You are an expert player of text-based adventure games. "
    "At each step you will be given the current game observation and your inventory. "
    "Respond with a Thought (one sentence of reasoning about what to do) "
    "followed by an Act (the exact game command to execute). "
    "Do NOT repeat a failed action. Try different approaches when stuck."
)

REACT_FORMAT = (
    "\nRespond using EXACTLY this format (two lines, nothing else):\n"
    "Thought: <one sentence of reasoning>\n"
    "Act: <short game command>\n"
)

# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_react_response(response: str) -> tuple[str, str]:
    """Parse Thought and Act from a ReAct response.
    Returns (thought, action). Falls back gracefully."""
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
    """
    ReAct agent for text-adventure games.

    The agent maintains a running transcript of (Thought, Act, Obs) triples
    as the in-context "scratchpad". No evolution snapshots are provided —
    the agent learns only from within-episode reasoning.
    """

    def __init__(self, model: str, temperature: float = 0.4):
        self.model = model
        self.temperature = temperature

    def make_agent_fn(self, initial_history: list[dict]):
        """
        initial_history: truncated test trajectory up to decision_point
          (list of {step, obs, inv, action}).  Used to prime the scratchpad
          so the agent starts with awareness of what happened before.
        """
        # Build a read-only primer from pre-decision-point history
        primer_lines = []
        for s in initial_history:
            primer_lines.append(f"Obs {s['step']}: {s['obs']}")
            if s.get("inv"):
                primer_lines.append(f"Inv {s['step']}: {s['inv']}")
            primer_lines.append(f"Act {s['step']}: {s['action']}")
        primer = "\n".join(primer_lines)

        # Scratchpad grows with each new (Thought, Act, Obs) triple
        scratchpad: list[str] = []

        def agent_fn(obs: str, inv: str, step: int):
            # Append current obs to scratchpad
            scratchpad.append(f"Obs {step}: {obs}")
            if inv:
                scratchpad.append(f"Inv {step}: {inv}")

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

            # Append thought + act to scratchpad for next step
            scratchpad.append(f"Thought {step}: {thought}")
            scratchpad.append(f"Act {step}: {action}")

            return action, raw

        return agent_fn


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(task: dict, snapshots: dict, baseline: ReActBaseline,
             output_dir: Path, env: JerichoEnv) -> dict:
    game = task["game"]
    task_id = task["id"]
    decision_point = task["test_snapshot"]["decision_point"]
    test_ep_id = task["test_snapshot"]["episode_id"]

    test_ep = snapshots[test_ep_id]
    test_trajectory = test_ep["snapshot"]["trajectory"]
    truncated_test_traj = test_trajectory[:decision_point]

    ob, info = restore_game_state(env, test_trajectory, decision_point)

    log_path = output_dir / f"{task_id}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Task: {task_id}\n")
        f.write(f"Baseline: react\n")
        f.write(f"Game: {game}, Decision point: {decision_point}\n")
        f.write(f"Test episode: {test_ep_id}\n\n")
        f.write(f"=== Restored game state at step {decision_point} ===\n")
        f.write(f"OBS: {ob}\n\n")

    agent_fn = baseline.make_agent_fn(truncated_test_traj)

    # Run episode — inline version of run_episode_from_decision_point
    # (ReAct manages its own scratchpad, so we inline the loop here to
    # pass obs/inv to agent_fn before the env.step call)
    trajectory = []
    cur_ob, cur_info = ob, info

    with open(log_path, "a", encoding="utf-8") as f:
        for step in range(decision_point, STEP_LIMIT):
            action, raw_response = agent_fn(cur_ob, cur_info.get("inv", ""), step)

            f.write(f"[Step {step}] OBS: {cur_ob[:120]}\n")
            f.write(f"           RAW: {raw_response[:200]}\n")
            f.write(f"           ACTION: {action}\n")

            trajectory.append({
                "step": step,
                "obs": cur_ob,
                "inv": cur_info.get("inv", ""),
                "action": action,
                "raw_response": raw_response,
            })

            cur_ob, reward, done, cur_info = env.step(action)
            trajectory[-1]["reward"] = reward
            trajectory[-1]["score_after"] = cur_info.get("score", 0)

            print(f"  [step {step}] {action[:30]:30s} → reward={reward}, score={cur_info.get('score',0)}")

            if done:
                break

    first_step = trajectory[0] if trajectory else {}
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
        "obs_after_decision_point": first_step.get("obs"),
        "final_score": trajectory[-1]["score_after"] if trajectory else None,
        "trajectory_from_decision_point": trajectory,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", required=True, choices=GAMES)
    parser.add_argument("--model", default="openai/gpt-4.1")
    parser.add_argument("--temperature", default=0.4, type=float)
    parser.add_argument("--task_ids", nargs="*", help="Specific task IDs (default: all)")
    parser.add_argument("--distances", nargs="*", type=int, default=None,
                        help="Only run tasks with these distances (default: all). Example: --distances 1 5")
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
        output_dir = RESULTS_ROOT / args.game / "react" / model_slug / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running ReAct on {args.game} ({len(tasks)} tasks)")
    print(f"Model: {args.model}")
    print(f"Output: {output_dir}")

    baseline = ReActBaseline(model=args.model, temperature=args.temperature)
    rom_path = str(ROM_DIR / game_file(args.game))
    env = JerichoEnv(rom_path=rom_path, seed=0, step_limit=STEP_LIMIT, get_valid=False)

    results = []
    try:
        for i, task in enumerate(tasks):
            print(f"\n[{i+1}/{len(tasks)}] {task['id']} (type={task['type']}, dist={task['distance']})")
            try:
                result = run_task(task, snapshots, baseline, output_dir, env)
                results.append(result)
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback; traceback.print_exc()
                results.append({"task_id": task["id"], "error": str(e)})
    finally:
        env.close()

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
