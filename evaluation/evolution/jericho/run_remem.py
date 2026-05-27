#!/usr/bin/env python3
"""
Evolution Evaluation runner - ReMem baseline

Built on top of ReAct (rolling scratchpad + Thought/Act format), with the
ReMem memory retrieval module and Think-Prune-Act loop added on top:

  Memory bank:
    - One entry per evolution episode (trajectory + outcome).
    - Embedding computed from entry.to_text() using BAAI/bge-base-en-v1.5.

  Per test episode:
    1. Retrieve top-k entries ONCE at episode start, using the initial obs
       as the query (cosine similarity over embeddings).
    2. Think-Prune-Act loop at each step (up to MAX_THINK_REFINE_ITERS):
         Think:       add reasoning to trace, re-prompt
         Think-Prune: discard specified memory IDs, re-prompt
         Action:      execute game command, move to next step
    3. Rolling scratchpad (Obs/Thought/Act) accumulates across steps.

Reference: "Evo-Memory: Adaptive Memory Refinement for LLM Agents"
           github.com/zhaosnw/evo_mem
"""

import os
import sys
import re
import json
import time
import argparse
import numpy as np
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

from run_react import parse_react_response

MAX_THINK_REFINE_ITERS = 3

THINK_PRUNE_ACT_FORMAT_INITIAL = """
Output EXACTLY ONE LINE using one of these three formats:
  Think: <one sentence of reasoning about what to do next>
  Think-Prune: <comma-separated memory IDs to remove, e.g. "1,3">
  Action: <short game command to execute>

Output Think: once to reason, then Action: to act. Do not output Think: more than once.
"""

THINK_PRUNE_ACT_FORMAT_MUST_ACT = """
Output EXACTLY ONE LINE. You have already reasoned — now you MUST choose:
  Think-Prune: <comma-separated memory IDs to remove, if needed>
  Action: <short game command to execute>

Do NOT output Think: again. You MUST output Action: or Think-Prune: only.
"""


# ---------------------------------------------------------------------------
# Memory entry
# ---------------------------------------------------------------------------

class MemoryEntry:
    def __init__(self, task_id, input_text, trajectory=None, is_successful=False):
        self.task_id = task_id
        self.input_text = input_text          # initial obs (goal proxy)
        self.trajectory = trajectory or []    # condensed (action, obs) pairs
        self.is_successful = is_successful
        self.embedding = None

    def to_text(self) -> str:
        parts = [f"Task: {self.input_text[:300]}"]
        if self.trajectory:
            traj_lines = "\n".join(
                f"  {s['action']}: {s['observation'][:120]}"
                for s in self.trajectory[:20]
            )
            parts.append(f"Trajectory:\n{traj_lines}")
        parts.append("Result: " + ("Success" if self.is_successful else "Failure"))
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Embedding retrieval
# ---------------------------------------------------------------------------

_encoder = None

def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda:2")
    return _encoder

def _encode(texts: list[str]) -> np.ndarray:
    return _get_encoder().encode(texts, normalize_embeddings=True, show_progress_bar=False)

def _ensure_embeddings(entries: list[MemoryEntry]):
    missing = [e for e in entries if e.embedding is None]
    if not missing:
        return
    embs = _encode([e.to_text() for e in missing])
    for entry, emb in zip(missing, embs):
        entry.embedding = emb

def retrieve(memory_bank: list[MemoryEntry], query: str, top_k: int) -> list[MemoryEntry]:
    if not memory_bank:
        return []
    _ensure_embeddings(memory_bank)
    q_emb = _encode([query])[0]
    scored = sorted(memory_bank,
                    key=lambda e: float(np.dot(q_emb, e.embedding)),
                    reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Build memory bank from evolution snapshots
# ---------------------------------------------------------------------------

def _snapshot_to_entry(ep_id: str, episode: dict) -> MemoryEntry:
    traj = episode["snapshot"]["trajectory"]
    final_score = episode["snapshot"].get("final_score", 0) or 0
    initial_obs = traj[0]["obs"][:300] if traj else ""
    condensed = [
        {"action": s.get("action", ""), "observation": s.get("obs", "")[:120]}
        for s in traj[:30]
    ]
    return MemoryEntry(
        task_id=ep_id,
        input_text=initial_obs,
        trajectory=condensed,
        is_successful=final_score > 0,
    )


# ---------------------------------------------------------------------------
# Format retrieved memories and parse Think-Prune-Act responses
# ---------------------------------------------------------------------------

def _format_memories_with_ids(memories: list[MemoryEntry], display_ids: list[int]) -> str:
    if not memories:
        return ""
    lines = ["=== Retrieved Past Experience ==="]
    for did, entry in zip(display_ids, memories):
        outcome = "Success" if entry.is_successful else "Failure"
        lines.append(f"[Memory {did}] Outcome: {outcome}")
        lines.append(entry.to_text())
        lines.append("")
    return "\n".join(lines)

def _parse_tpa_response(response: str) -> tuple[str, str]:
    """Returns (type, content): type in {'think', 'prune', 'action'}.
    Priority: Action > Think-Prune > Think.
    Scans all lines so model can output Think then Action in one response."""
    action_val = think_prune_val = think_val = None
    for line in response.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        if action_val is None and re.match(r"Action:\s*", line, re.IGNORECASE):
            action_val = re.sub(r"(?i)^Action:\s*", "", line).strip()
        elif think_prune_val is None and re.match(r"Think-Prune:\s*", line, re.IGNORECASE):
            think_prune_val = re.sub(r"(?i)^Think-Prune:\s*", "", line).strip()
        elif think_val is None and re.match(r"Think:\s*", line, re.IGNORECASE):
            think_val = re.sub(r"(?i)^Think:\s*", "", line).strip()
    if action_val is not None:
        return "action", action_val
    if think_prune_val is not None:
        return "prune", think_prune_val
    if think_val is not None:
        return "think", think_val
    return "action", "look"

def _parse_prune_ids(content: str, n_memories: int) -> set[int]:
    ids = set()
    for token in re.split(r"[,\s]+", content):
        try:
            i = int(token)
            if 1 <= i <= n_memories:
                ids.add(i)
        except ValueError:
            pass
    return ids


# ---------------------------------------------------------------------------
# ReMem baseline (ReAct + memory retrieval)
# ---------------------------------------------------------------------------

REMEM_SYSTEM_PROMPT = (
    "You are an expert player of text-based adventure games. "
    "You have access to relevant past experience shown at the top of each prompt — "
    "use it to avoid repeating past failures and to guide your decisions. "
    "Respond with a Thought (one sentence of reasoning) "
    "followed by an Act (the exact game command to execute). "
    "Do NOT repeat a failed action. Try different approaches when stuck."
)


class ReMemBaseline:
    def __init__(self, model: str, top_k: int = 3, temperature: float = 0.4):
        self.model = model
        self.top_k = top_k
        self.temperature = temperature

    def build_memory_bank(self, evolution_episodes: list[dict]) -> list[MemoryEntry]:
        bank = [_snapshot_to_entry(ep["id"], ep) for ep in evolution_episodes]
        print(f"  [remem] memory bank: {len(bank)} entries")
        return bank

    def make_agent_fn(self, retrieved: list[MemoryEntry], initial_history: list[dict]):
        # Active memory set (prunable during episode)
        active = list(range(len(retrieved)))

        # Build primer from pre-dp history
        primer_lines = []
        for s in initial_history:
            primer_lines.append(f"Obs {s['step']}: {s['obs']}")
            if s.get("inv"):
                primer_lines.append(f"Inv {s['step']}: {s['inv']}")
            primer_lines.append(f"Act {s['step']}: {s['action']}")
        primer = "\n".join(primer_lines)

        # Rolling scratchpad (Obs/Inv/Thought/Act per step)
        scratchpad: list[str] = []

        def agent_fn(obs: str, inv: str, step: int):
            scratchpad.append(f"Obs {step}: {obs}")
            if inv:
                scratchpad.append(f"Inv {step}: {inv}")

            # Build static parts of prompt
            episode_history = ""
            if primer:
                episode_history += "=== Episode History (before decision point) ===\n"
                episode_history += primer + "\n\n"
            episode_history += "=== Current Episode ===\n"
            episode_history += "\n".join(scratchpad)

            # Think-Prune-Act loop
            reasoning_trace = []
            final_action = "look"
            full_response = ""
            had_think = False

            for iter_idx in range(MAX_THINK_REFINE_ITERS):
                cur_memories = [retrieved[i] for i in active]
                display_ids = list(range(1, len(cur_memories) + 1))
                mem_text = _format_memories_with_ids(cur_memories, display_ids)
                trace_text = "\n".join(reasoning_trace) if reasoning_trace else "(none yet)"

                # After first Think, force the model to act
                fmt = THINK_PRUNE_ACT_FORMAT_MUST_ACT if had_think else THINK_PRUNE_ACT_FORMAT_INITIAL

                user_prompt = ""
                if mem_text:
                    user_prompt += mem_text + "\n"
                user_prompt += episode_history + "\n\n"
                if reasoning_trace:
                    user_prompt += f"CURRENT REASONING:\n{trace_text}\n"
                user_prompt += fmt

                res = chat_completion_with_retries(
                    model=self.model,
                    sys_prompt=REMEM_SYSTEM_PROMPT,
                    prompt=truncate_text(user_prompt, MAX_CONTEXT_TOKENS),
                    max_tokens=256,
                    temperature=self.temperature,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                raw = res.choices[0].message.content if res and res.choices else ""
                full_response = raw

                rtype, content = _parse_tpa_response(raw)

                if rtype == "think":
                    reasoning_trace.append(f"Think: {content[:200]}")
                    had_think = True
                elif rtype == "prune":
                    ids_to_prune = _parse_prune_ids(content, len(cur_memories))
                    keep = [idx for j, idx in enumerate(active) if (j + 1) not in ids_to_prune]
                    active.clear(); active.extend(keep)
                    reasoning_trace.append(f"Think-Prune: removed memories {ids_to_prune}")
                else:  # action
                    final_action = content
                    break

            scratchpad.append(f"Thought {step}: {reasoning_trace[-1] if reasoning_trace else ''}")
            scratchpad.append(f"Act {step}: {final_action}")

            return final_action, full_response

        return agent_fn


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(task: dict, snapshots: dict, baseline: ReMemBaseline,
             output_dir: Path, env: JerichoEnv) -> dict:
    game = task["game"]
    task_id = task["id"]
    decision_point = task["test_snapshot"]["decision_point"]
    test_ep_id = task["test_snapshot"]["episode_id"]
    evolution_ep_ids = task["evolution_snapshots"]

    test_ep = snapshots[test_ep_id]
    test_trajectory = test_ep["snapshot"]["trajectory"]
    truncated_test_traj = test_trajectory[:decision_point]

    evolution_episodes = [snapshots[ep_id] for ep_id in evolution_ep_ids]
    memory_bank = baseline.build_memory_bank(evolution_episodes)

    # Retrieve once using initial obs as query
    test_goal = test_trajectory[0]["obs"] if test_trajectory else ""
    retrieved = retrieve(memory_bank, test_goal, baseline.top_k)
    print(f"  [remem] retrieved {len(retrieved)} memories")

    ob, info = restore_game_state(env, test_trajectory, decision_point)

    log_path = output_dir / f"{task_id}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Task: {task_id}\n")
        f.write(f"Baseline: remem\n")
        f.write(f"Game: {game}, Decision point: {decision_point}\n")
        f.write(f"Test episode: {test_ep_id}\n")
        f.write(f"Memory bank: {len(memory_bank)} entries, retrieved: {len(retrieved)}\n")
        f.write(f"Retrieved: {[r.task_id for r in retrieved]}\n\n")
        f.write(f"=== Restored game state at step {decision_point} ===\n")
        f.write(f"OBS: {ob}\n\n")

    agent_fn = baseline.make_agent_fn(retrieved, truncated_test_traj)

    # Inline episode loop (same as ReAct — scratchpad managed inside agent_fn)
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
        "baseline": "remem",
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
    parser.add_argument("--top_k", default=3, type=int)
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
        output_dir = RESULTS_ROOT / args.game / "remem" / model_slug / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running ReMem on {args.game} ({len(tasks)} tasks)")
    print(f"Model: {args.model}, top_k={args.top_k}")
    print(f"Output: {output_dir}")

    baseline = ReMemBaseline(model=args.model, top_k=args.top_k, temperature=args.temperature)
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
            "baseline": "remem",
            "model": args.model,
            "n_tasks": len(tasks),
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Results saved to {results_path}")


if __name__ == "__main__":
    main()
