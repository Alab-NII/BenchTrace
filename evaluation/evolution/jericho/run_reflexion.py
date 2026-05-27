#!/usr/bin/env python3
"""
Evolution Evaluation runner - Reflexion baseline

Reflexion (Shinn et al., 2023): after each failed episode, the agent generates
a verbal self-reflection and stores it in episodic memory. Future trials start
with these reflections prepended to the context.

Adaptation for EvolEval:
  - Evolution snapshots serve as the "past failed trials"
  - For each evolution snapshot, an LLM generates a concise reflection
    (what went wrong, what to try instead)
  - Reflections are prepended to the test episode context
  - Test episode runs with ReAct-style rolling scratchpad

Reference: "Reflexion: Language Agents with Verbal Reinforcement Learning"
           Shinn et al., NeurIPS 2023
           https://github.com/noahshinn/reflexion
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
from run_react import (
    REACT_SYSTEM_PROMPT,
    REACT_FORMAT,
    parse_react_response,
)

# ---------------------------------------------------------------------------
# Reflection generation
# ---------------------------------------------------------------------------

REFLECTION_SYSTEM_PROMPT = (
    "You are an expert analyst of text-based adventure game playthroughs. "
    "Your task is to identify key mistakes and lessons from a past episode."
)

REFLECTION_PROMPT_TEMPLATE = """\
Below is a trajectory from a text-based adventure game episode.

{trajectory}

Final score: {final_score}

Write a concise reflection (2-3 sentences) covering:
1. What went wrong or what critical mistake was made.
2. What concrete action or strategy should be tried instead next time.

Be specific. Do not summarize the trajectory — focus on actionable lessons.
"""


def format_trajectory_for_reflection(trajectory: list[dict], max_steps: int = 30) -> str:
    lines = []
    for s in trajectory[:max_steps]:
        lines.append(f"[Step {s['step']}] Obs: {s['obs'][:150]}")
        if s.get("inv"):
            lines.append(f"           Inv: {s['inv'][:80]}")
        lines.append(f"           Act: {s['action']}")
    if len(trajectory) > max_steps:
        lines.append(f"... ({len(trajectory) - max_steps} more steps)")
    return "\n".join(lines)


def generate_reflection(episode: dict, model: str, temperature: float = 0.0) -> str:
    """Generate a verbal reflection from a past episode."""
    traj = episode["snapshot"]["trajectory"]
    final_score = episode["snapshot"].get("final_score", 0) or 0
    traj_text = format_trajectory_for_reflection(traj)

    prompt = REFLECTION_PROMPT_TEMPLATE.format(
        trajectory=traj_text,
        final_score=final_score,
    )

    res = chat_completion_with_retries(
        model=model,
        sys_prompt=REFLECTION_SYSTEM_PROMPT,
        prompt=prompt,
        max_tokens=200,
        temperature=temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    if res and res.choices:
        return res.choices[0].message.content.strip()
    return "(reflection unavailable)"


# ---------------------------------------------------------------------------
# Reflexion agent
# ---------------------------------------------------------------------------

class ReflexionBaseline:
    """
    ReAct agent augmented with verbal reflections from past episodes.

    Context structure:
      === Lessons from Past Episodes ===
      [Reflection 1 (episode id: ...)]
      ...

      === Episode History (before decision point) ===
      Obs 0: ... / Act 0: ...

      === Current Episode ===
      Obs N: ... / Thought N: ... / Act N: ...
    """

    def __init__(self, model: str, temperature: float = 0.4):
        self.model = model
        self.temperature = temperature

    def build_reflections(
        self,
        evolution_ep_ids: list[str],
        snapshots: dict,
    ) -> list[tuple[str, str]]:
        """Generate reflections for all evolution episodes. Returns [(ep_id, reflection)]."""
        reflections = []
        for ep_id in evolution_ep_ids:
            ep = snapshots.get(ep_id)
            if ep is None:
                continue
            print(f"    [reflexion] generating reflection for {ep_id} ...", end=" ", flush=True)
            ref = generate_reflection(ep, self.model, temperature=0.0)
            print("done")
            reflections.append((ep_id, ref))
        return reflections

    def build_primer(
        self,
        reflections: list[tuple[str, str]],
        initial_history: list[dict],
    ) -> str:
        """Build the fixed primer: reflections + pre-dp history."""
        parts = []

        if reflections:
            lines = ["=== Lessons from Past Episodes ==="]
            for i, (ep_id, ref) in enumerate(reflections, 1):
                lines.append(f"[Lesson {i}] {ref}")
            parts.append("\n".join(lines))

        if initial_history:
            hist_lines = ["=== Episode History (before decision point) ==="]
            for s in initial_history:
                hist_lines.append(f"Obs {s['step']}: {s['obs']}")
                if s.get("inv"):
                    hist_lines.append(f"Inv {s['step']}: {s['inv']}")
                hist_lines.append(f"Act {s['step']}: {s['action']}")
            parts.append("\n".join(hist_lines))

        return "\n\n".join(parts)

    def make_agent_fn(self, primer: str):
        """ReAct agent with reflections in the fixed primer."""
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
    baseline: ReflexionBaseline,
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
    truncated_test_traj = test_trajectory[:decision_point]

    # Generate reflections from evolution snapshots
    reflections = baseline.build_reflections(evolution_ep_ids, snapshots)

    # Build primer: reflections + pre-dp history
    primer = baseline.build_primer(reflections, truncated_test_traj)

    # Restore game state
    ob, info = restore_game_state(env, test_trajectory, decision_point)

    log_path = output_dir / f"{task_id}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Task: {task_id}\n")
        f.write(f"Baseline: reflexion\n")
        f.write(f"Game: {game}, Decision point: {decision_point}\n")
        f.write(f"Test episode: {test_ep_id}\n")
        f.write(f"Evolution snapshots: {evolution_ep_ids}\n\n")
        f.write(f"=== Generated Reflections ===\n")
        for ep_id, ref in reflections:
            f.write(f"[{ep_id}] {ref}\n\n")
        f.write(f"=== Restored game state at step {decision_point} ===\n")
        f.write(f"OBS: {ob}\n\n")

    agent_fn = baseline.make_agent_fn(primer)

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
        "baseline": "reflexion",
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
        "reflections": [{"ep_id": ep_id, "reflection": ref} for ep_id, ref in reflections],
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
        output_dir = RESULTS_ROOT / args.game / "reflexion" / model_slug / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running Reflexion on {args.game} ({len(tasks)} tasks)")
    print(f"Model: {args.model}")
    print(f"Output: {output_dir}")

    baseline = ReflexionBaseline(model=args.model, temperature=args.temperature)
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
            "baseline": "reflexion",
            "model": args.model,
            "n_tasks": len(tasks),
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Results saved to {results_path}")


if __name__ == "__main__":
    main()
