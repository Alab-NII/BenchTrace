#!/usr/bin/env python3
"""
Evolution Evaluation runner - Agent-R baseline (parametric)

Agent-R (Yuan et al., 2025): iterative self-training via MCTS-guided revision
trajectories. The model is fine-tuned on (error context → corrected actions)
pairs, encoding past failure knowledge into weights rather than context.

Adaptation for EvolEval:
  - Training: agentR_build_data.py constructs revision trajectories from
    evolution snapshots; agentR_finetune.py LoRA fine-tunes Qwen3-32B
  - Evaluation: the fine-tuned model runs ReAct from the decision point
  - Evolution snapshots are NOT shown in the prompt (parametric baseline)
  - Pre-dp test trajectory is shown as context (same as Reflexion/MemRL)

Usage:
  # 1. Build revision data (once per game)
  python agentR_build_data.py --game zork1 --model Qwen/Qwen3-32B

  # 2. Fine-tune (once per game or combined)
  python agentR_finetune.py --data agentR_data/zork1.jsonl \\
      --output_dir output/agentR_ckpt/zork1

  # 3. Merge LoRA into base weights (see agentR_finetune.py docstring)

  # 4. Serve merged model with vLLM:
  #    vllm serve output/agentR_merged/zork1 --port 8000

  # 5. Evaluate:
  python run_agentR.py --game zork1 --model output/agentR_merged/zork1

Reference: "Agent-R: Training Language Model Agents to Reflect via Iterative
           Self-Training", Yuan et al., arXiv 2501.11425
"""

import json
import time
import argparse
from pathlib import Path

from utils import (
    DATASET_ROOT, ROM_DIR, RESULTS_ROOT, MAX_CONTEXT_TOKENS, STEP_LIMIT, GAMES,
    load_snapshots, restore_game_state,
    JerichoEnv, chat_completion_with_retries, truncate_text, game_file,
)
from run_react import REACT_SYSTEM_PROMPT, REACT_FORMAT, parse_react_response


# ---------------------------------------------------------------------------
# Agent-R baseline
# ---------------------------------------------------------------------------

class AgentRBaseline:
    """
    Parametric baseline: fine-tuned model, no evolution context in prompt.

    The agent sees:
      - The test trajectory up to the decision point (as context)
      - Rolling scratchpad of observations + thoughts + actions post-dp

    Evolution snapshot knowledge is encoded in the model weights via fine-tuning.
    """

    def __init__(self, model: str, temperature: float = 0.4):
        self.model = model
        self.temperature = temperature

    def build_primer(self, pre_dp_trajectory: list[dict]) -> str:
        if not pre_dp_trajectory:
            return ""
        lines = ["=== Current Episode History (before decision point) ==="]
        for s in pre_dp_trajectory:
            lines.append(f"Obs {s['step']}: {s['obs']}")
            if s.get("inv"):
                lines.append(f"Inv {s['step']}: {s['inv']}")
            lines.append(f"Act {s['step']}: {s['action']}")
        return "\n".join(lines)

    def make_agent_fn(self, primer: str):
        scratchpad: list[str] = []

        def agent_fn(obs: str, inv: str, step: int):
            scratchpad.append(f"Obs {step}: {obs}")
            if inv:
                scratchpad.append(f"Inv {step}: {inv}")

            user_prompt = ""
            if primer:
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

def run_task(
    task: dict,
    snapshots: dict,
    baseline: AgentRBaseline,
    output_dir: Path,
    env: JerichoEnv,
) -> dict:
    game = task["game"]
    task_id = task["id"]
    decision_point = task["test_snapshot"]["decision_point"]
    test_ep_id = task["test_snapshot"]["episode_id"]
    evolution_ep_ids = task["evolution_snapshots"]

    test_ep = snapshots[test_ep_id]
    test_trajectory = test_ep["snapshot"]["trajectory"]
    pre_dp_traj = test_trajectory[:decision_point]

    primer = baseline.build_primer(pre_dp_traj)
    ob, info = restore_game_state(env, test_trajectory, decision_point)

    log_path = output_dir / f"{task_id}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Task: {task_id}\n")
        f.write(f"Baseline: agentR (parametric)\n")
        f.write(f"Game: {game}, Decision point: {decision_point}\n")
        f.write(f"Test episode: {test_ep_id}\n")
        f.write(f"Evolution snapshots (NOT shown to agent — encoded in weights): {evolution_ep_ids}\n\n")
        f.write(f"=== Restored game state at step {decision_point} ===\n")
        f.write(f"OBS: {ob}\n\n")

    agent_fn = baseline.make_agent_fn(primer)
    trajectory = []
    cur_ob, cur_info = ob, info

    with open(log_path, "a", encoding="utf-8") as f:
        for step in range(decision_point, STEP_LIMIT):
            action, raw = agent_fn(cur_ob, cur_info.get("inv", ""), step)

            f.write(f"[Step {step}] OBS: {cur_ob[:120]}\n")
            f.write(f"           RAW: {raw[:200]}\n")
            f.write(f"           ACTION: {action}\n")

            trajectory.append({
                "step": step,
                "obs": cur_ob,
                "inv": cur_info.get("inv", ""),
                "action": action,
                "raw_response": raw,
            })

            cur_ob, reward, done, cur_info = env.step(action)
            trajectory[-1]["reward"] = reward
            trajectory[-1]["score_after"] = cur_info.get("score", 0)

            print(f"  [step {step}] {action[:30]:30s} → reward={reward}, score={cur_info.get('score', 0)}")

            if done:
                break

    first_step = trajectory[0] if trajectory else {}
    return {
        "task_id": task_id,
        "baseline": "agentR",
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
    parser.add_argument("--model", required=True,
                        help="Fine-tuned model name/path (merged LoRA, served via vLLM)")
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
        model_slug = args.model.replace("/", "_").replace("\\", "_")
        output_dir = RESULTS_ROOT / args.game / "agentR" / model_slug / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running Agent-R on {args.game} ({len(tasks)} tasks)")
    print(f"Model: {args.model}  [parametric — evolution snapshots not in prompt]")
    print(f"Output: {output_dir}")

    baseline = AgentRBaseline(model=args.model, temperature=args.temperature)
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
            "baseline": "agentR",
            "model": args.model,
            "n_tasks": len(tasks),
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Results saved to {results_path}")


if __name__ == "__main__":
    main()
