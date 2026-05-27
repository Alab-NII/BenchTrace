#!/usr/bin/env python3
"""
AlfWorld Evolution Evaluation runner — Reflexion baseline

Reflexion (Shinn et al., 2023): after each failed episode, the agent generates
a verbal self-reflection. Evolution snapshots serve as the "past failed trials".
Reflections are prepended as "Lessons from Past Episodes" in the primer.

AlfWorld differences from JTTL version:
  - env.step() returns (obs, done, info) — no reward in tuple
  - No inventory field; use info["progress"] and info["won"] for scoring
  - Per-task env (different game_file per test episode)
  - generate_reflection uses info["progress"] (float 0-1) instead of integer score
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
    TASK_SHORT_LIST,
    load_snapshots,
    restore_game_state,
    get_game_file,
)
from src.alfworld_env import AlfWorldEnv
from src.openai_helpers import chat_completion_with_retries, truncate_text

from run_react import (
    REACT_SYSTEM_PROMPT,
    REACT_FORMAT,
    parse_react_response,
)

# ---------------------------------------------------------------------------
# Reflection generation
# ---------------------------------------------------------------------------

REFLECTION_SYSTEM_PROMPT = (
    "You are an expert analyst of household task playthroughs in a text-based environment. "
    "Your task is to identify key mistakes and lessons from a past episode."
)

REFLECTION_PROMPT_TEMPLATE = """\
Below is a trajectory from a household task episode.

{trajectory}

Final progress: {final_progress:.2f} (1.0 = task completed)

Write a concise reflection (2-3 sentences) covering:
1. What went wrong or what critical mistake was made.
2. What concrete action or strategy should be tried instead next time.

Be specific. Do not summarize the trajectory — focus on actionable lessons.
"""


def format_trajectory_for_reflection(trajectory: list[dict], max_steps: int = 30) -> str:
    lines = []
    for s in trajectory[:max_steps]:
        lines.append(f"[Step {s['step']}] Obs: {s['obs'][:150]}")
        lines.append(f"           Act: {s.get('action', '')}")
    if len(trajectory) > max_steps:
        lines.append(f"... ({len(trajectory) - max_steps} more steps)")
    return "\n".join(lines)


def generate_reflection(episode: dict, model: str, temperature: float = 0.0) -> str:
    traj = episode["snapshot"]["trajectory"]
    final_progress = float(episode["snapshot"].get("final_score", 0) or 0)
    traj_text = format_trajectory_for_reflection(traj)

    prompt = REFLECTION_PROMPT_TEMPLATE.format(
        trajectory=traj_text,
        final_progress=final_progress,
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
    def __init__(self, model: str, temperature: float = 0.4):
        self.model = model
        self.temperature = temperature

    def build_reflections(
        self,
        evolution_ep_ids: list[str],
        snapshots: dict,
    ) -> list[tuple[str, str]]:
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
        pre_dp_traj: list[dict],
    ) -> str:
        parts = []

        if reflections:
            lines = ["=== Lessons from Past Episodes ==="]
            for i, (ep_id, ref) in enumerate(reflections, 1):
                lines.append(f"[Lesson {i}] {ref}")
            parts.append("\n".join(lines))

        if pre_dp_traj:
            hist_lines = ["=== Episode History (before decision point) ==="]
            for s in pre_dp_traj:
                hist_lines.append(f"Obs {s['step']}: {s['obs']}")
                if s.get("action"):
                    hist_lines.append(f"Act {s['step']}: {s['action']}")
            parts.append("\n".join(hist_lines))

        return "\n\n".join(parts)

    def make_agent_fn(self, primer: str):
        scratchpad: list[str] = []

        def agent_fn(obs: str, step: int):
            scratchpad.append(f"Obs {step}: {obs}")

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
) -> dict:
    task_id = task["id"]
    task_short = task["task"]
    decision_point = task["test_snapshot"]["decision_point"]
    test_ep_id = task["test_snapshot"]["episode_id"]
    evolution_ep_ids = task["evolution_snapshots"]

    test_ep = snapshots[test_ep_id]
    test_trajectory = test_ep["snapshot"]["trajectory"]
    pre_dp_traj = test_trajectory[:decision_point]

    reflections = baseline.build_reflections(evolution_ep_ids, snapshots)
    primer = baseline.build_primer(reflections, pre_dp_traj)

    game_file = get_game_file(test_ep_id)
    env = AlfWorldEnv(game_file=game_file, step_limit=STEP_LIMIT)

    ob, info = restore_game_state(env, test_trajectory, decision_point)

    log_path = output_dir / f"{task_id}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Task: {task_id}\n")
        f.write(f"Baseline: reflexion\n")
        f.write(f"Task type: {task_short}, Decision point: {decision_point}\n")
        f.write(f"Test episode: {test_ep_id}\n")
        f.write(f"Evolution snapshots: {evolution_ep_ids}\n\n")
        f.write("=== Generated Reflections ===\n")
        for ep_id, ref in reflections:
            f.write(f"[{ep_id}] {ref}\n\n")
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
        "baseline": "reflexion",
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
        "reflections": [{"ep_id": ep_id, "reflection": ref} for ep_id, ref in reflections],
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
        output_dir = RESULTS_ROOT / args.task / "reflexion" / model_slug / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running Reflexion on {args.task} ({len(tasks)} tasks)")
    print(f"Model: {args.model}")
    print(f"Output: {output_dir}")

    baseline = ReflexionBaseline(model=args.model, temperature=args.temperature)

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
            "task": args.task,
            "baseline": "reflexion",
            "model": args.model,
            "n_tasks": len(tasks),
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Results saved to {results_path}")


if __name__ == "__main__":
    main()
