#!/usr/bin/env python3
"""
Evolution Evaluation runner - Non-Evolution baseline

No evolution snapshots given to the agent.
Agent sees: system prompt + test trajectory up to decision_point,
then runs a rolling-scratchpad episode from decision_point onwards.
"""

import json
import time
import argparse
import traceback
from pathlib import Path

from utils import (
    DATASET_ROOT, ROM_DIR, RESULTS_ROOT, MAX_CONTEXT_TOKENS, STEP_LIMIT, GAMES,
    SYSTEM_PROMPT, NON_EVO_ACTION_FORMAT,
    load_snapshots, parse_action, restore_game_state, run_episode_from_decision_point,
    JerichoEnv, chat_completion_with_retries, truncate_text, game_file,
)


class NonEvolutionBaseline:
    """No evolution snapshots. Agent sees the pre-dp trajectory as a primer,
    plus a rolling scratchpad of post-dp (obs, inv, action) accumulating each step."""

    def __init__(self, model: str, temperature: float = 0.4):
        self.model = model
        self.temperature = temperature

    def build_primer(self, test_trajectory: list[dict]) -> str:
        lines = []
        for s in test_trajectory:
            lines.append(f"Obs {s['step']}: {s['obs']}")
            if s.get("inv"):
                lines.append(f"Inv {s['step']}: {s['inv']}")
            lines.append(f"Act {s['step']}: {s['action']}")
        return "\n".join(lines)

    def make_agent_fn(self, primer: str):
        scratchpad = []

        def agent_fn(obs: str, inv: str, step: int):
            scratchpad.append(f"Obs {step}: {obs}")
            if inv:
                scratchpad.append(f"Inv {step}: {inv}")

            user_prompt = ""
            if primer:
                user_prompt += "=== Episode History (before decision point) ===\n"
                user_prompt += primer + "\n\n"
            user_prompt += "=== Current Episode ===\n"
            user_prompt += "\n".join(scratchpad)
            user_prompt += NON_EVO_ACTION_FORMAT

            res = chat_completion_with_retries(
                model=self.model,
                sys_prompt=SYSTEM_PROMPT,
                prompt=truncate_text(user_prompt, MAX_CONTEXT_TOKENS),
                max_tokens=256,
                temperature=self.temperature,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = res.choices[0].message.content if res and res.choices else ""
            action = parse_action(raw)
            scratchpad.append(f"Act {step}: {action}")
            return action, raw

        return agent_fn


def run_task(task: dict, snapshots: dict, baseline: NonEvolutionBaseline,
             output_dir: Path, env: JerichoEnv) -> dict:
    game = task["game"]
    task_id = task["id"]
    decision_point = task["test_snapshot"]["decision_point"]
    test_ep_id = task["test_snapshot"]["episode_id"]

    test_ep = snapshots[test_ep_id]
    test_trajectory = test_ep["snapshot"]["trajectory"]
    truncated_test_traj = test_trajectory[:decision_point]

    context = baseline.build_primer(truncated_test_traj)
    ob, info = restore_game_state(env, test_trajectory, decision_point)

    log_path = output_dir / f"{task_id}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Task: {task_id}\n")
        f.write(f"Baseline: non_evolution\n")
        f.write(f"Game: {game}, Decision point: {decision_point}\n")
        f.write(f"Test episode: {test_ep_id}\n\n")
        f.write(f"=== Restored game state at decision_point {decision_point} ===\n")
        f.write(f"OBS: {ob}\n\n")

    agent_fn = baseline.make_agent_fn(context)
    trajectory_from_dp = run_episode_from_decision_point(
        env, agent_fn, ob, info, decision_point, log_path
    )

    first_step = trajectory_from_dp[0] if trajectory_from_dp else {}
    return {
        "task_id": task_id,
        "baseline": "non_evolution",
        "game": game,
        "type": task["type"],
        "distance": task["distance"],
        "target_failure_instance": task["target_failure_instance"],
        "test_episode_id": test_ep_id,
        "decision_point": decision_point,
        "scoring_method": task["scoring_method"],
        "action_at_decision_point": first_step.get("action"),
        "obs_after_decision_point": first_step.get("obs"),
        "final_score": trajectory_from_dp[-1]["score_after"] if trajectory_from_dp else None,
        "trajectory_from_decision_point": trajectory_from_dp,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", required=True, choices=GAMES)
    parser.add_argument("--model", default="openai/gpt-4.1")
    parser.add_argument("--temperature", default=0.4, type=float)
    parser.add_argument("--task_ids", nargs="*")
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
        output_dir = RESULTS_ROOT / args.game / "non_evolution" / model_slug / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running Non-Evolution on {args.game} ({len(tasks)} tasks)")
    print(f"Model: {args.model}")
    print(f"Output: {output_dir}")

    baseline = NonEvolutionBaseline(model=args.model, temperature=args.temperature)
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
                traceback.print_exc()
                results.append({"task_id": task["id"], "error": str(e)})
    finally:
        env.close()

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({
            "game": args.game,
            "baseline": "non_evolution",
            "model": args.model,
            "n_tasks": len(tasks),
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Results saved to {results_path}")


if __name__ == "__main__":
    main()
