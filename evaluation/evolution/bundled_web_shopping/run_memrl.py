#!/usr/bin/env python3
"""
BundledWebShopping Evolution Evaluation runner — MemRL baseline

MemRL: Q-value-weighted episodic memory retrieval with script compression.
  - Build: each evolution snapshot → LLM-generated selection script
  - Retrieve: hybrid score = (1-q)*cosine_sim + q*q_value (q = final_score)
  - Agent: ReAct with HIGH-SCORE / LOW-SCORE memory buckets in context
"""

import re
import json
import time
import argparse
import traceback
import numpy as np
from pathlib import Path

from utils import (
    DATASET_ROOT, RESULTS_ROOT, GAME_LIST, STEP_LIMIT, MAX_CONTEXT_TOKENS,
    load_snapshots, load_dataset_split, make_env,
    restore_game_state, format_prior_subtasks,
)
from src.openai_helpers import chat_completion_with_retries, truncate_text
from run_react import REACT_SYSTEM_PROMPT, REACT_FORMAT, parse_shopping_response

Q_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Sentence encoder
# ---------------------------------------------------------------------------

_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
    return _encoder


def _encode(texts: list[str]) -> np.ndarray:
    return _get_encoder().encode(texts, normalize_embeddings=True, show_progress_bar=False)


# ---------------------------------------------------------------------------
# Script generation
# ---------------------------------------------------------------------------

SCRIPT_SYSTEM = (
    "You are an expert at bundled web shopping. Extract a concise selection strategy "
    "from a past shopping episode."
)

SCRIPT_TEMPLATE = """\
Below is a bundled web shopping episode:
{trajectory}

Final score: {final_score:.2f} (1.0 = all subtasks correct)

Extract a 3-5 step selection strategy: what criteria to prioritize, what mistakes to avoid.
Keep it concise and actionable.
"""


def _format_traj_for_script(trajectory: list[dict]) -> str:
    lines = []
    for step in trajectory:
        if step.get("action") is None:
            continue
        idx = step.get("subtask_idx", step.get("step", 0) // 2)
        action = step.get("action", "")
        correct = step.get("correct", None)
        m = re.search(r"\[([A-Za-z])\]", action)
        sel = f"[{m.group(1).upper()}]" if m else "(?)"
        status = "✓" if correct else "✗" if correct is not None else ""
        lines.append(f"Subtask {idx+1}: Selected {sel} {status} — {action[:60]}")
    return "\n".join(lines)


def generate_script(ep: dict, model: str, temperature: float = 0.0) -> str:
    traj = ep["snapshot"]["trajectory"]
    score = float(ep["snapshot"].get("final_score", 0) or 0)
    prompt = SCRIPT_TEMPLATE.format(
        trajectory=_format_traj_for_script(traj),
        final_score=score,
    )
    res = chat_completion_with_retries(
        model=model, sys_prompt=SCRIPT_SYSTEM, prompt=prompt,
        max_tokens=200, temperature=temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return res.choices[0].message.content.strip() if res and res.choices else ""


# ---------------------------------------------------------------------------
# MemRL entry & retrieval
# ---------------------------------------------------------------------------

class MemRLEntry:
    def __init__(self, ep_id, input_text, script, q_value):
        self.ep_id = ep_id
        self.input_text = input_text
        self.script = script
        self.q_value = q_value
        self.embedding = None


def build_memrl_entries(evolution_ep_ids: list[str], snapshots: dict,
                        model: str) -> list[MemRLEntry]:
    entries = []
    for ep_id in evolution_ep_ids:
        ep = snapshots.get(ep_id)
        if ep is None:
            continue
        traj = ep["snapshot"]["trajectory"]
        first_obs = next((s["obs"] for s in traj if s.get("action") is None), "")
        q_value = float(ep["snapshot"].get("final_score", 0) or 0)
        print(f"    Generating script for {ep_id} (q={q_value:.2f})...")
        script = generate_script(ep, model)
        entry = MemRLEntry(ep_id=ep_id, input_text=first_obs[:300],
                           script=script, q_value=q_value)
        entries.append(entry)

    if entries:
        embs = _encode([e.input_text for e in entries])
        for e, emb in zip(entries, embs):
            e.embedding = emb
    return entries


def retrieve_memrl(query: str, entries: list[MemRLEntry],
                   top_k: int = 3, q_weight: float = 0.5) -> list[MemRLEntry]:
    if not entries:
        return []
    query_emb = _encode([query])[0]
    embs = np.stack([e.embedding for e in entries])
    cos_sims = embs @ query_emb
    q_values = np.array([e.q_value for e in entries])
    scores = (1 - q_weight) * cos_sims + q_weight * q_values
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [entries[i] for i in top_idx]


# ---------------------------------------------------------------------------
# MemRL agent
# ---------------------------------------------------------------------------

class MemRLBaseline:
    def __init__(self, model: str, temperature: float = 0.4, top_k: int = 3):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k

    def build_entries(self, evolution_ep_ids: list[str], snapshots: dict) -> list[MemRLEntry]:
        return build_memrl_entries(evolution_ep_ids, snapshots, self.model)

    def make_agent_fn(self, entries: list[MemRLEntry], pre_dp_context: str):
        current_run_selections: list[str] = []

        def agent_fn(obs: str, subtask_idx: int):
            retrieved = retrieve_memrl(obs, entries, top_k=self.top_k)
            high = [e for e in retrieved if e.q_value >= Q_THRESHOLD]
            low  = [e for e in retrieved if e.q_value < Q_THRESHOLD]

            mem_block = ""
            if high:
                mem_block += "=== HIGH-SCORE STRATEGIES ===\n"
                for e in high:
                    mem_block += f"[q={e.q_value:.2f}] {e.script[:200]}\n"
                mem_block += "\n"
            if low:
                mem_block += "=== LOW-SCORE PATTERNS TO AVOID ===\n"
                for e in low:
                    mem_block += f"[q={e.q_value:.2f}] {e.script[:200]}\n"
                mem_block += "\n"

            user_prompt = mem_block
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

def run_task(task: dict, snapshots: dict, baseline: MemRLBaseline,
             dataset, output_dir: Path) -> dict:
    task_id = task["id"]
    game = task["game"]
    decision_point = task["test_snapshot"]["decision_point"]
    test_ep_id = task["test_snapshot"]["episode_id"]
    evolution_ep_ids = task["evolution_snapshots"]

    test_trajectory = snapshots[test_ep_id]["snapshot"]["trajectory"]
    pre_dp_context = format_prior_subtasks(test_trajectory, decision_point)

    entries = baseline.build_entries(evolution_ep_ids, snapshots)
    print(f"  MemRL entries: {len(entries)}")

    env = make_env(test_ep_id, dataset)
    ob, info = restore_game_state(env, test_trajectory, decision_point)

    log_path = output_dir / f"{task_id}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Task: {task_id}\nBaseline: memrl\n")
        f.write(f"Game: {game}, Decision point: {decision_point}\n\n")

    agent_fn = baseline.make_agent_fn(entries, pre_dp_context)
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
        "task_id": task_id, "baseline": "memrl", "game": game,
        "type": task["type"], "distance": task["distance"],
        "target_failure_instance": task["target_failure_instance"],
        "test_episode_id": test_ep_id, "decision_point": decision_point,
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
    parser.add_argument("--top_k", default=3, type=int)
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
        output_dir = RESULTS_ROOT / args.game / "memrl" / model_slug / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running MemRL on {args.game} ({len(tasks)} tasks)")
    baseline = MemRLBaseline(model=args.model, temperature=args.temperature, top_k=args.top_k)

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
            "game": args.game, "baseline": "memrl", "model": args.model,
            "n_tasks": len(tasks), "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nDone. Results saved to {results_path}")


if __name__ == "__main__":
    main()
