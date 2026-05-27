#!/usr/bin/env python3
"""
ScienceWorld Evolution Evaluation runner — MemRL baseline

MemRL: Q-value-weighted episodic memory retrieval with script compression.
  - Build: each evolution snapshot → LLM-generated script (3-5 steps)
  - Retrieve: hybrid score = (1-q_weight)*cosine_sim + q_weight*q_value
  - Agent: ReAct with HIGH-SCORE / LOW-SCORE memory buckets in context

ScienceWorld differences from AlfWorld:
  - env.step(action_str) takes free-form text directly
  - task_name and variation parsed from episode ID
  - q_value = final_score directly (already 0-1 progress, no GAME_MAX_SCORE)
  - Encoder uses cpu
  - STEP_LIMIT = 100
"""

import re
import json
import time
import argparse
import traceback
import numpy as np
from pathlib import Path

from utils import (
    DATASET_ROOT,
    RESULTS_ROOT,
    MAX_CONTEXT_TOKENS,
    STEP_LIMIT,
    GAME_LIST,
    load_snapshots,
    restore_game_state,
    parse_task_and_variation,
)
from src.scienceworld_env import ScienceWorldEnv
from src.openai_helpers import chat_completion_with_retries, truncate_text

from run_react import REACT_SYSTEM_PROMPT, REACT_FORMAT, parse_react_response

Q_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Sentence encoder (shared, lazy-loaded)
# ---------------------------------------------------------------------------

_encoder = None

def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")
    return _encoder

def _encode(texts: list[str]) -> np.ndarray:
    return _get_encoder().encode(texts, normalize_embeddings=True, show_progress_bar=False)


# ---------------------------------------------------------------------------
# Script generation
# ---------------------------------------------------------------------------

SCRIPT_SYSTEM_PROMPT = (
    "You are an expert analyst of science experiment trajectories in a text-based environment. "
    "Your task is to extract a concise, reusable strategy script from a past episode."
)

SCRIPT_PROMPT_TEMPLATE = """\
Below is a trajectory from a science experiment episode.

{trajectory}

Final progress: {final_progress:.2f} (1.0 = experiment completed)

Write a high-level script (3-5 numbered steps) capturing:
1. The key strategy or approach attempted.
2. What succeeded and what failed.
3. Concrete lessons for future attempts.

Be generic enough to apply to similar situations, but specific enough to be actionable.
Output ONLY the numbered steps, nothing else.
"""


def format_trajectory_for_script(trajectory: list[dict], max_steps: int = 25) -> str:
    lines = []
    for s in trajectory[:max_steps]:
        lines.append(f"[Step {s['step']}] Obs: {s['obs'][:120]}")
        lines.append(f"           Act: {s.get('action', '')}")
    if len(trajectory) > max_steps:
        lines.append(f"... ({len(trajectory) - max_steps} more steps)")
    return "\n".join(lines)


def generate_script(episode: dict, model: str, temperature: float = 0.0) -> str:
    traj = episode["snapshot"]["trajectory"]
    final_score = float(episode["snapshot"].get("final_score", 0) or 0)
    traj_text = format_trajectory_for_script(traj)

    prompt = SCRIPT_PROMPT_TEMPLATE.format(
        trajectory=traj_text,
        final_progress=final_score,
    )
    res = chat_completion_with_retries(
        model=model,
        sys_prompt=SCRIPT_SYSTEM_PROMPT,
        prompt=prompt,
        max_tokens=200,
        temperature=temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    if res and res.choices:
        return res.choices[0].message.content.strip()
    return "(script unavailable)"


# ---------------------------------------------------------------------------
# Memory entry
# ---------------------------------------------------------------------------

class MemoryEntry:
    def __init__(self, ep_id: str, q_value: float, final_score: float,
                 script: str, traj_excerpt: str, key_obs: str):
        self.ep_id = ep_id
        self.q_value = q_value
        self.final_score = final_score
        self.script = script
        self.traj_excerpt = traj_excerpt
        self.key_obs = key_obs
        self.embedding: np.ndarray | None = None

    def format_for_prompt(self, index: int) -> str:
        return (
            f"[Memory {index}] Progress: {self.final_score:.2f}  (Q={self.q_value:.2f})\n"
            f"SCRIPT:\n{self.script}\n\n"
            f"KEY MOMENTS:\n{self.traj_excerpt}"
        )


def build_memory_entry(ep_id: str, episode: dict, model: str) -> MemoryEntry:
    snap = episode["snapshot"]
    final_score = float(snap.get("final_score", 0) or 0)
    q_value = final_score  # already 0-1 (progress)
    traj = snap["trajectory"]
    traj_excerpt = format_trajectory_for_script(traj, max_steps=15)
    key_obs = traj[0]["obs"] if traj else ""

    print(f"    [memrl] building memory for {ep_id} ...", end=" ", flush=True)
    script = generate_script(episode, model)
    print("done")

    return MemoryEntry(
        ep_id=ep_id,
        q_value=q_value,
        final_score=final_score,
        script=script,
        traj_excerpt=traj_excerpt,
        key_obs=key_obs,
    )


# ---------------------------------------------------------------------------
# Q-weighted retrieval
# ---------------------------------------------------------------------------

def retrieve_memories(
    query_obs: str,
    memories: list[MemoryEntry],
    k: int = 3,
    q_weight: float = 0.4,
) -> tuple[list[MemoryEntry], list[MemoryEntry]]:
    if not memories:
        return [], []

    missing = [m for m in memories if m.embedding is None]
    if missing:
        embs = _encode([m.key_obs for m in missing])
        for mem, emb in zip(missing, embs):
            mem.embedding = emb

    query_emb = _encode([query_obs])[0]

    scored = []
    for mem in memories:
        cos_sim = float(np.dot(query_emb, mem.embedding))
        hybrid = (1.0 - q_weight) * cos_sim + q_weight * mem.q_value
        scored.append((hybrid, mem))

    scored.sort(key=lambda x: -x[0])
    top_k = [m for _, m in scored[:k]]

    high = [m for m in top_k if m.q_value >= Q_THRESHOLD]
    low  = [m for m in top_k if m.q_value < Q_THRESHOLD]
    return high, low


# ---------------------------------------------------------------------------
# MemRL agent
# ---------------------------------------------------------------------------

MEMRL_SYSTEM_PROMPT = (
    "You are an expert at completing science experiments in a text-based environment. "
    "You have access to memories from past episodes. "
    "Study the HIGH-SCORE memories to learn good strategies, "
    "and study the LOW-SCORE memories to avoid past mistakes. "
    "At each step, respond with a Thought (one sentence of reasoning) "
    "followed by an Act (the free-form text command to execute)."
)


def build_memory_context(high_mems: list[MemoryEntry], low_mems: list[MemoryEntry]) -> str:
    parts = ["=== Memory Bank ==="]

    if high_mems:
        parts.append("--- HIGH-SCORE MEMORIES (strategies to follow) ---")
        for i, m in enumerate(high_mems, 1):
            parts.append(m.format_for_prompt(i))
    else:
        parts.append("--- HIGH-SCORE MEMORIES: none retrieved ---")

    parts.append("")

    if low_mems:
        parts.append("--- LOW-SCORE MEMORIES (mistakes to avoid) ---")
        offset = len(high_mems)
        for i, m in enumerate(low_mems, offset + 1):
            parts.append(m.format_for_prompt(i))
    else:
        parts.append("--- LOW-SCORE MEMORIES: none retrieved ---")

    return "\n".join(parts)


class MemRLBaseline:
    def __init__(self, model: str, temperature: float = 0.4,
                 k: int = 3, q_weight: float = 0.4):
        self.model = model
        self.temperature = temperature
        self.k = k
        self.q_weight = q_weight

    def build_memories(self, evolution_ep_ids: list[str], snapshots: dict) -> list[MemoryEntry]:
        memories = []
        for ep_id in evolution_ep_ids:
            ep = snapshots.get(ep_id)
            if ep is None:
                continue
            mem = build_memory_entry(ep_id, ep, self.model)
            memories.append(mem)
        return memories

    def make_agent_fn(self, memory_context: str, pre_dp_traj: list[dict]):
        primer_lines = []
        for s in pre_dp_traj:
            primer_lines.append(f"Obs {s['step']}: {s['obs']}")
            if s.get("action"):
                primer_lines.append(f"Act {s['step']}: {s['action']}")
        episode_history = "\n".join(primer_lines)

        scratchpad: list[str] = []

        def agent_fn(obs: str, step: int):
            scratchpad.append(f"Obs {step}: {obs}")

            user_prompt = ""
            if memory_context:
                user_prompt += memory_context + "\n\n"
            if episode_history:
                user_prompt += "=== Episode History (before decision point) ===\n"
                user_prompt += episode_history + "\n\n"
            user_prompt += "=== Current Episode ===\n"
            user_prompt += "\n".join(scratchpad)
            user_prompt += REACT_FORMAT

            res = chat_completion_with_retries(
                model=self.model,
                sys_prompt=MEMRL_SYSTEM_PROMPT,
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

def run_task(task: dict, snapshots: dict, baseline: MemRLBaseline,
             output_dir: Path) -> dict:
    task_id = task["id"]
    game = task["game"]
    decision_point = task["test_snapshot"]["decision_point"]
    test_ep_id = task["test_snapshot"]["episode_id"]
    evolution_ep_ids = task["evolution_snapshots"]

    test_ep = snapshots[test_ep_id]
    test_trajectory = test_ep["snapshot"]["trajectory"]
    pre_dp_traj = test_trajectory[:decision_point]

    memories = baseline.build_memories(evolution_ep_ids, snapshots)

    query_obs = pre_dp_traj[-1]["obs"] if pre_dp_traj else test_trajectory[0]["obs"]
    high_mems, low_mems = retrieve_memories(
        query_obs, memories, k=baseline.k, q_weight=baseline.q_weight
    )
    memory_context = build_memory_context(high_mems, low_mems)

    task_name, variation = parse_task_and_variation(test_ep_id)
    env = ScienceWorldEnv(task_name, variation, step_limit=STEP_LIMIT)

    ob, info = restore_game_state(env, test_trajectory, decision_point)

    log_path = output_dir / f"{task_id}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Task: {task_id}\n")
        f.write(f"Baseline: memrl\n")
        f.write(f"Game: {game}, Decision point: {decision_point}\n")
        f.write(f"Test episode: {test_ep_id}\n")
        f.write(f"Evolution snapshots: {evolution_ep_ids}\n\n")
        f.write("=== Retrieved Memories ===\n")
        f.write(f"HIGH-SCORE ({len(high_mems)}): {[m.ep_id for m in high_mems]}\n")
        f.write(f"LOW-SCORE  ({len(low_mems)}): {[m.ep_id for m in low_mems]}\n\n")
        f.write(memory_context + "\n\n")
        f.write(f"=== Restored state at step {decision_point} ===\n")
        f.write(f"OBS: {ob}\n\n")

    agent_fn = baseline.make_agent_fn(memory_context, pre_dp_traj)
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
        "baseline": "memrl",
        "game": game,
        "type": task["type"],
        "distance": task["distance"],
        "target_failure_instance": task["target_failure_instance"],
        "test_episode_id": test_ep_id,
        "decision_point": decision_point,
        "scoring_method": task["scoring_method"],
        "action_at_decision_point": first_step.get("action"),
        "obs_after_decision_point": first_step.get("obs"),
        "final_progress": last_step.get("progress_after"),
        "won": last_step.get("won", False),
        "trajectory_from_decision_point": trajectory,
        "memories_retrieved": {
            "high_score": [{"ep_id": m.ep_id, "q_value": m.q_value, "script": m.script} for m in high_mems],
            "low_score":  [{"ep_id": m.ep_id, "q_value": m.q_value, "script": m.script} for m in low_mems],
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", required=True, choices=GAME_LIST)
    parser.add_argument("--model", default="openai/gpt-4.1")
    parser.add_argument("--temperature", default=0.4, type=float)
    parser.add_argument("--k", default=3, type=int)
    parser.add_argument("--q_weight", default=0.4, type=float)
    parser.add_argument("--task_ids", nargs="*")
    parser.add_argument("--distances", nargs="*", type=int, default=None)
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
        output_dir = RESULTS_ROOT / args.game / "memrl" / model_slug / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running MemRL on {args.game} ({len(tasks)} tasks)")
    print(f"Model: {args.model}  k={args.k}  q_weight={args.q_weight}")
    print(f"Output: {output_dir}")

    baseline = MemRLBaseline(
        model=args.model,
        temperature=args.temperature,
        k=args.k,
        q_weight=args.q_weight,
    )

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
            "game": args.game,
            "baseline": "memrl",
            "model": args.model,
            "k": args.k,
            "q_weight": args.q_weight,
            "n_tasks": len(tasks),
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Results saved to {results_path}")


if __name__ == "__main__":
    main()
