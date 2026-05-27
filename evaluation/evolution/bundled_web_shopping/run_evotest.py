#!/usr/bin/env python3
"""
BundledWebShopping Evolution Evaluation runner — EvoTest baseline

EvoTest: process each evolution snapshot through LLM evolution call,
sequentially updating the guiding prompt. Also builds cross-episode
positive/negative memory from correct/incorrect subtask selections.
Uses evolved prompt + memory during test episode.

BundledWebShopping-specific:
  - Positive memory: subtask steps where correct=True
  - Negative memory: subtask steps where correct=False
  - Evolution: refine selection criteria from subtask trajectories
  - No state extractor code (discrete selection task, no state tracking needed)
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
    restore_game_state, format_prior_subtasks, format_trajectory_for_evo,
)
from src.openai_helpers import chat_completion_with_retries, truncate_text
from run_react import REACT_SYSTEM_PROMPT, REACT_FORMAT, parse_shopping_response

INITIAL_GUIDING_PROMPT = (
    "Carefully evaluate all product options for compatibility and budget fit before selecting. "
    "Always read the product requirements fully and compare all options systematically. "
    "Prioritize the attribute most explicitly specified in the goal (e.g., highest-priced, "
    "highest-rated, specific brand compatibility)."
)

# ---------------------------------------------------------------------------
# EvoTest: evolution + memory extraction
# ---------------------------------------------------------------------------

def evolve(cur_prompt: str, trajectory_text: str,
           evo_model: str, evo_temperature: float) -> str:
    trajectory_text = truncate_text(trajectory_text, max_tokens=20000)
    sys_prompt = (
        "You are an expert at bundled web shopping tasks. Analyze the existing selection "
        "guide and episode history to generate an improved guide that helps an agent "
        "select more compatible and budget-appropriate products."
    )
    user_prompt = f'''
Generate a new improved selection guide for a bundled web shopping agent.

Current guide:
"{cur_prompt}"

Episode history:
--- HISTORY START ---
{trajectory_text}
--- HISTORY END ---

Generate an improved guide that:
1. Reinforces selection criteria that led to correct answers.
2. Adds specific guidance to avoid the types of mistakes made.
3. Prioritizes the most common attribute goals (highest-priced, highest-rated, compatibility).

Respond with the new guide text only (no extra formatting or explanation):
'''
    try:
        res = chat_completion_with_retries(
            model=evo_model, sys_prompt=sys_prompt, prompt=user_prompt,
            max_tokens=500, temperature=evo_temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        new_prompt = res.choices[0].message.content.strip() if res and res.choices else ""
        return new_prompt if new_prompt and len(new_prompt) > 20 else cur_prompt
    except Exception as e:
        print(f"  [evolve] Error: {e}. Keeping current prompt.")
        return cur_prompt


def _extract_positive_from_traj(trajectory: list[dict]) -> list[dict]:
    positives = []
    for step in trajectory:
        if step.get("action") is None:
            continue
        if step.get("correct"):
            positives.append({
                "state": step["obs"][:400],
                "action": step.get("action", "")[:100],
                "delta_score": 1,
            })
    return positives


def _extract_negative_from_traj(trajectory: list[dict]) -> list[dict]:
    negatives = []
    for step in trajectory:
        if step.get("action") is None:
            continue
        if step.get("correct") is False:
            negatives.append({
                "reason": "incorrect_selection",
                "states": [step["obs"][:200]],
                "actions": [step.get("action", "")[:100]],
            })
    return negatives


def _tokenize(text: str) -> list[str]:
    return [t for t in (text or "").lower().replace("\n", " ").split() if t.isalnum()]


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _retrieve_similar_positives(positives: list[dict], query_state: str,
                                 k: int = 3) -> list[dict]:
    q_tok = _tokenize(query_state)
    scored = sorted(positives,
                    key=lambda ex: _jaccard(q_tok, _tokenize(ex["state"])), reverse=True)
    return [ex for ex in scored[:k] if _jaccard(q_tok, _tokenize(ex["state"])) > 0]


def _format_few_shot(positives: list[dict]) -> str:
    if not positives:
        return "(none)"
    lines = []
    for ex in positives:
        lines.append(f"Q: {ex['state'][:200]}\nSelected: {ex['action'][:80]}")
        lines.append("---")
    return "\n".join(lines)


def _format_negative_block(negatives: list[dict]) -> str:
    if not negatives:
        return ""
    parts = []
    for neg in negatives[-3:]:
        parts.append("Incorrect selection in situation like: " + neg["states"][0][:100])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# EvoTest baseline
# ---------------------------------------------------------------------------

class EvoTestBaseline:
    def __init__(self, model: str, evo_model: str,
                 temperature: float = 0.4, evo_temperature: float = 0.7):
        self.model = model
        self.evo_model = evo_model
        self.temperature = temperature
        self.evo_temperature = evo_temperature

    def evolve_from_snapshots(self, evolution_ep_ids: list[str], snapshots: dict,
                               mem_path: Path = None,
                               neg_mem_path: Path = None):
        prompt = INITIAL_GUIDING_PROMPT
        all_positives: list[dict] = []
        all_negatives: list[dict] = []

        for i, ep_id in enumerate(evolution_ep_ids):
            ep = snapshots[ep_id]
            traj = ep["snapshot"]["trajectory"]
            final_score = float(ep["snapshot"].get("final_score", 0) or 0)
            pos = _extract_positive_from_traj(traj)
            neg = _extract_negative_from_traj(traj)
            history_str = format_trajectory_for_evo(traj, final_score)
            prompt = evolve(prompt, history_str, self.evo_model, self.evo_temperature)
            all_positives.extend(pos)
            all_negatives.extend(neg)
            print(f"  [evolve {i+1}/{len(evolution_ep_ids)}] ep={ep_id} "
                  f"+{len(pos)} pos, +{len(neg)} neg | prompt: {prompt[:80].replace(chr(10),' ')!r}")

        if mem_path:
            with open(mem_path, "w", encoding="utf-8") as f:
                for item in all_positives:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        if neg_mem_path:
            with open(neg_mem_path, "w", encoding="utf-8") as f:
                for item in all_negatives:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        return prompt, all_positives, all_negatives

    def make_agent_fn(self, guiding_prompt: str, pre_dp_context: str,
                      positives: list[dict], negatives: list[dict]):
        current_run_selections: list[str] = []
        neg_block = _format_negative_block(negatives)

        def agent_fn(obs: str, subtask_idx: int):
            few_shot = _format_few_shot(_retrieve_similar_positives(positives, obs))
            few_shot_block = (
                f"SUCCESSFUL SELECTIONS FROM PAST EPISODES:\n{few_shot}\n\n"
                if few_shot != "(none)" else ""
            )
            neg_section = f"INCORRECT SELECTION PATTERNS TO AVOID:\n{neg_block}\n\n" if neg_block else ""

            sys_prompt = (
                "You are an intelligent shopping agent selecting compatible products.\n\n"
                f"Selection guide: {guiding_prompt}"
            )
            user_prompt = few_shot_block + neg_section
            if pre_dp_context:
                user_prompt += "=== Prior Subtask Selections (this episode) ===\n"
                user_prompt += pre_dp_context + "\n\n"
            if current_run_selections:
                user_prompt += "=== Selections Made in This Run ===\n"
                user_prompt += "\n".join(current_run_selections) + "\n\n"
            user_prompt += "=== Current Subtask ===\n" + obs + REACT_FORMAT

            res = chat_completion_with_retries(
                model=self.model, sys_prompt=sys_prompt,
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

def run_task(task: dict, snapshots: dict, baseline: EvoTestBaseline,
             dataset, output_dir: Path) -> dict:
    task_id = task["id"]
    game = task["game"]
    decision_point = task["test_snapshot"]["decision_point"]
    test_ep_id = task["test_snapshot"]["episode_id"]
    evolution_ep_ids = task["evolution_snapshots"]

    test_trajectory = snapshots[test_ep_id]["snapshot"]["trajectory"]
    pre_dp_context = format_prior_subtasks(test_trajectory, decision_point)

    guiding_prompt, positives, negatives = baseline.evolve_from_snapshots(
        evolution_ep_ids, snapshots,
        mem_path=output_dir / f"{task_id}_mem.jsonl",
        neg_mem_path=output_dir / f"{task_id}_neg_mem.jsonl",
    )

    env = make_env(test_ep_id, dataset)
    ob, info = restore_game_state(env, test_trajectory, decision_point)

    log_path = output_dir / f"{task_id}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Task: {task_id}\nBaseline: evotest\n")
        f.write(f"Game: {game}, Decision point: {decision_point}\n")
        f.write(f"Evolved prompt: {guiding_prompt[:200]}\n")
        f.write(f"Positive: {len(positives)}, Negative: {len(negatives)}\n\n")

    agent_fn = baseline.make_agent_fn(guiding_prompt, pre_dp_context, positives, negatives)
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
        "task_id": task_id, "baseline": "evotest", "game": game,
        "type": task["type"], "distance": task["distance"],
        "target_failure_instance": task["target_failure_instance"],
        "test_episode_id": test_ep_id, "decision_point": decision_point,
        "scoring_method": task["scoring_method"],
        "evolved_prompt": guiding_prompt,
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
    parser.add_argument("--evo_model", default="openai/gpt-4.1")
    parser.add_argument("--temperature", default=0.4, type=float)
    parser.add_argument("--evo_temperature", default=0.7, type=float)
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
        output_dir = RESULTS_ROOT / args.game / "evotest" / model_slug / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running EvoTest on {args.game} ({len(tasks)} tasks)")
    print(f"Agent: {args.model} | Evo: {args.evo_model}")

    baseline = EvoTestBaseline(
        model=args.model, evo_model=args.evo_model,
        temperature=args.temperature, evo_temperature=args.evo_temperature,
    )

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
            "game": args.game, "baseline": "evotest",
            "model": args.model, "evo_model": args.evo_model,
            "n_tasks": len(tasks), "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nDone. Results saved to {results_path}")


if __name__ == "__main__":
    main()
