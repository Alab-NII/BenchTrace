#!/usr/bin/env python3
"""
BundledWebShopping Evolution Evaluation runner — ReMem baseline

ReMem: ReAct + episodic memory retrieval.
  - Memory bank: one entry per evolution snapshot (episode summary)
  - Retrieve top-k entries via cosine similarity to current subtask obs
  - Agent uses retrieved memories as context for selection
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

MAX_THINK_REFINE_ITERS = 2

REMEM_SYSTEM_PROMPT = (
    "You are an intelligent shopping agent. You have access to relevant past experience "
    "shown at the top — use it to identify the correct selection criteria and avoid repeating "
    "past mistakes. Select the product that best fits the requirements."
)

THINK_PRUNE_ACT_FORMAT = (
    "\nOutput EXACTLY ONE LINE using one of these formats:\n"
    "  Think: <one sentence reasoning about which option to select>\n"
    "  Think-Prune: <comma-separated memory IDs to discard, e.g. '1,3'>\n"
    "  Selection: [<LETTER>]\n"
    "Think once, then select. After Selection, stop.\n"
)

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
# Memory entry
# ---------------------------------------------------------------------------

class MemoryEntry:
    def __init__(self, ep_id, input_text, summary, is_successful):
        self.ep_id = ep_id
        self.input_text = input_text
        self.summary = summary
        self.is_successful = is_successful
        self.embedding = None

    def to_text(self, mem_id: int) -> str:
        status = "SUCCESS" if self.is_successful else "FAILURE"
        return f"[Memory {mem_id}] [{status}] {self.summary[:300]}"


def _build_episode_summary(ep: dict) -> str:
    traj = ep["snapshot"]["trajectory"]
    lines = []
    for step in traj:
        if step.get("action") is None:
            continue
        idx = step.get("subtask_idx", step.get("step", 0) // 2)
        action = step.get("action", "")
        correct = step.get("correct", None)
        m = re.search(r"\[([A-Za-z])\]", action)
        sel = f"[{m.group(1).upper()}]" if m else "(?)"
        status = "✓" if correct else "✗" if correct is not None else ""
        lines.append(f"Subtask {idx+1}: Selected {sel} {status}")
    score = ep["snapshot"].get("final_score", 0)
    return " | ".join(lines) + f" | Score: {score:.2f}"


def build_memory_bank(evolution_ep_ids: list[str], snapshots: dict) -> list[MemoryEntry]:
    entries = []
    for ep_id in evolution_ep_ids:
        ep = snapshots.get(ep_id)
        if ep is None:
            continue
        traj = ep["snapshot"]["trajectory"]
        first_obs = next((s["obs"] for s in traj if s.get("action") is None), "")
        summary = _build_episode_summary(ep)
        score = float(ep["snapshot"].get("final_score", 0) or 0)
        entry = MemoryEntry(
            ep_id=ep_id,
            input_text=first_obs[:300],
            summary=summary,
            is_successful=(score > 0.5),
        )
        entries.append(entry)

    if entries:
        embs = _encode([e.input_text for e in entries])
        for e, emb in zip(entries, embs):
            e.embedding = emb
    return entries


def retrieve_memories(query: str, bank: list[MemoryEntry], top_k: int = 3,
                      active_ids: set = None) -> list[tuple[int, MemoryEntry]]:
    active = [(i, e) for i, e in enumerate(bank)
              if active_ids is None or i not in active_ids]
    if not active:
        return []
    query_emb = _encode([query])[0]
    embs = np.stack([e.embedding for _, e in active])
    scores = embs @ query_emb
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(active[j][0], active[j][1]) for j in top_idx]


# ---------------------------------------------------------------------------
# ReMem agent
# ---------------------------------------------------------------------------

class ReMemBaseline:
    def __init__(self, model: str, temperature: float = 0.4, top_k: int = 3):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k

    def make_agent_fn(self, memory_bank: list[MemoryEntry], pre_dp_context: str):
        current_run_selections: list[str] = []
        pruned_ids: set[int] = set()

        def agent_fn(obs: str, subtask_idx: int):
            retrieved = retrieve_memories(obs, memory_bank, top_k=self.top_k,
                                          active_ids=pruned_ids)

            mem_block = ""
            if retrieved:
                lines = ["=== Relevant Past Episodes ==="]
                for mem_id, entry in retrieved:
                    lines.append(entry.to_text(mem_id + 1))
                mem_block = "\n".join(lines) + "\n\n"

            for _ in range(MAX_THINK_REFINE_ITERS):
                user_prompt = mem_block
                if pre_dp_context:
                    user_prompt += "=== Prior Subtask Selections (this episode) ===\n"
                    user_prompt += pre_dp_context + "\n\n"
                if current_run_selections:
                    user_prompt += "=== Selections Made in This Run ===\n"
                    user_prompt += "\n".join(current_run_selections) + "\n\n"
                user_prompt += "=== Current Subtask ===\n" + obs + THINK_PRUNE_ACT_FORMAT

                res = chat_completion_with_retries(
                    model=self.model, sys_prompt=REMEM_SYSTEM_PROMPT,
                    prompt=truncate_text(user_prompt, MAX_CONTEXT_TOKENS),
                    max_tokens=256, temperature=self.temperature,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                raw = res.choices[0].message.content if res and res.choices else ""
                line = raw.strip().splitlines()[0] if raw.strip() else ""

                if line.startswith("Think-Prune:"):
                    ids_str = line[len("Think-Prune:"):].strip()
                    for id_s in ids_str.split(","):
                        try:
                            pruned_ids.add(int(id_s.strip()) - 1)
                        except ValueError:
                            pass
                    retrieved = retrieve_memories(obs, memory_bank, top_k=self.top_k,
                                                  active_ids=pruned_ids)
                    if retrieved:
                        lines = ["=== Relevant Past Episodes ==="]
                        for mem_id, entry in retrieved:
                            lines.append(entry.to_text(mem_id + 1))
                        mem_block = "\n".join(lines) + "\n\n"
                    continue

                action = parse_shopping_response(raw)
                current_run_selections.append(f"Subtask {subtask_idx + 1}: Selected {action}")
                return action, raw

            # fallback
            action = parse_shopping_response(raw if 'raw' in dir() else "")
            current_run_selections.append(f"Subtask {subtask_idx + 1}: Selected {action}")
            return action, ""

        return agent_fn


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(task: dict, snapshots: dict, baseline: ReMemBaseline,
             dataset, output_dir: Path) -> dict:
    task_id = task["id"]
    game = task["game"]
    decision_point = task["test_snapshot"]["decision_point"]
    test_ep_id = task["test_snapshot"]["episode_id"]
    evolution_ep_ids = task["evolution_snapshots"]

    test_trajectory = snapshots[test_ep_id]["snapshot"]["trajectory"]
    pre_dp_context = format_prior_subtasks(test_trajectory, decision_point)

    memory_bank = build_memory_bank(evolution_ep_ids, snapshots)
    print(f"  Memory bank: {len(memory_bank)} entries")

    env = make_env(test_ep_id, dataset)
    ob, info = restore_game_state(env, test_trajectory, decision_point)

    log_path = output_dir / f"{task_id}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Task: {task_id}\nBaseline: remem\n")
        f.write(f"Game: {game}, Decision point: {decision_point}\n")
        f.write(f"Memory entries: {len(memory_bank)}\n\n")

    agent_fn = baseline.make_agent_fn(memory_bank, pre_dp_context)
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
        "task_id": task_id, "baseline": "remem", "game": game,
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
        output_dir = RESULTS_ROOT / args.game / "remem" / model_slug / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running ReMem on {args.game} ({len(tasks)} tasks)")
    baseline = ReMemBaseline(model=args.model, temperature=args.temperature, top_k=args.top_k)

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
            "game": args.game, "baseline": "remem", "model": args.model,
            "n_tasks": len(tasks), "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nDone. Results saved to {results_path}")


if __name__ == "__main__":
    main()
