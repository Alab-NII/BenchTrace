#!/usr/bin/env python3
"""
Evolution Evaluation runner - RAG baseline

RAG: at each step, embed the current observation with BAAI/bge-base-en-v1.5,
retrieve the top-k most similar steps from evolution snapshots (cosine similarity),
synthesize the retrieved context into actionable insights via one LLM call,
then feed the insights into a ReAct-style decision prompt.

Knowledge base: step-level index built from all evolution snapshots for the task.
Each entry: (obs, inv, action, score_after) from a past episode step.
Embedding model: BAAI/bge-base-en-v1.5 (local, CPU).
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
from run_react import REACT_SYSTEM_PROMPT, REACT_FORMAT, parse_react_response

# ---------------------------------------------------------------------------
# Sentence encoder (shared, lazy-loaded)
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

# ---------------------------------------------------------------------------
# Step index: build from evolution snapshots
# ---------------------------------------------------------------------------

class StepEntry:
    """One step from a past evolution snapshot episode."""
    def __init__(self, ep_id: str, step: int, obs: str, inv: str,
                 action: str, score_after: float):
        self.ep_id = ep_id
        self.step = step
        self.obs = obs
        self.inv = inv
        self.action = action
        self.score_after = score_after

    def format(self) -> str:
        parts = [f"Obs: {self.obs[:200]}"]
        if self.inv:
            parts.append(f"Inv: {self.inv[:80]}")
        parts.append(f"Action: {self.action}")
        parts.append(f"Score after: {self.score_after:.0f}")
        return " | ".join(parts)


def build_step_index(evolution_ep_ids: list[str],
                     snapshots: dict) -> tuple[list[StepEntry], np.ndarray | None]:
    """
    Build a flat list of StepEntry objects and a (N, D) embedding matrix
    from all steps across the given evolution episode IDs.
    """
    entries: list[StepEntry] = []
    for ep_id in evolution_ep_ids:
        ep = snapshots.get(ep_id)
        if ep is None:
            continue
        traj = ep["snapshot"]["trajectory"]
        for s in traj:
            entries.append(StepEntry(
                ep_id=ep_id,
                step=s["step"],
                obs=s.get("obs", ""),
                inv=s.get("inv", ""),
                action=s.get("action", ""),
                score_after=float(s.get("score_after", s.get("reward", 0)) or 0),
            ))

    if not entries:
        return entries, None

    texts = [e.obs for e in entries]
    embeddings = _encode(texts)
    return entries, embeddings


def retrieve_steps(current_obs: str,
                   entries: list[StepEntry],
                   embeddings: np.ndarray,
                   top_k: int = 5) -> list[StepEntry]:
    """Return top-k most similar steps by cosine similarity on obs text."""
    if not entries or embeddings is None:
        return []
    query = _encode([current_obs])              # (1, D)
    sims = (embeddings @ query.T).flatten()     # (N,)  — already L2-normalised
    top_idx = np.argsort(sims)[::-1][:top_k]
    return [entries[i] for i in top_idx]

# ---------------------------------------------------------------------------
# RAG synthesis prompt
# ---------------------------------------------------------------------------

RAG_SYNTHESIS_SYSTEM = (
    "You are an expert analyst of text-based adventure game trajectories. "
    "Given a set of retrieved past steps that are similar to the current game state, "
    "synthesize 2-3 concise, actionable insights: what to try and what to avoid."
)

RAG_SYNTHESIS_TEMPLATE = """\
Current game state:
{current_obs}

Retrieved similar steps from past episodes:
{retrieved}

In 2-3 sentences, what should the agent do (or avoid) based on these past experiences?"""


def synthesize_insights(current_obs: str,
                        retrieved: list[StepEntry],
                        model: str,
                        temperature: float = 0.0) -> str:
    """One LLM call: compress retrieved steps into actionable insights."""
    if not retrieved:
        return ""
    retrieved_text = "\n".join(
        f"[{i+1}] {e.format()}" for i, e in enumerate(retrieved)
    )
    prompt = RAG_SYNTHESIS_TEMPLATE.format(
        current_obs=current_obs[:400],
        retrieved=retrieved_text,
    )
    res = chat_completion_with_retries(
        model=model,
        sys_prompt=RAG_SYNTHESIS_SYSTEM,
        prompt=prompt,
        max_tokens=150,
        temperature=temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    if res and res.choices:
        return res.choices[0].message.content.strip()
    return ""

# ---------------------------------------------------------------------------
# RAG agent
# ---------------------------------------------------------------------------

class RAGBaseline:
    def __init__(self, model: str, temperature: float = 0.4, top_k: int = 5):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k

    def make_agent_fn(self,
                      primer: str,
                      entries: list[StepEntry],
                      embeddings: np.ndarray | None):
        scratchpad: list[str] = []

        def agent_fn(obs: str, inv: str, step: int):
            scratchpad.append(f"Obs {step}: {obs}")
            if inv:
                scratchpad.append(f"Inv {step}: {inv}")

            # Retrieve + synthesize
            retrieved = retrieve_steps(obs, entries, embeddings, self.top_k)
            insights = synthesize_insights(obs, retrieved, self.model, temperature=0.0)

            user_prompt = ""
            if primer:
                user_prompt += primer + "\n\n"
            if insights:
                user_prompt += "=== Insights from Similar Past Steps ===\n"
                user_prompt += insights + "\n\n"
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

            return action, raw, insights

        return agent_fn

# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(task: dict, snapshots: dict, baseline: RAGBaseline,
             output_dir: Path, env: JerichoEnv) -> dict:
    game = task["game"]
    task_id = task["id"]
    decision_point = task["test_snapshot"]["decision_point"]
    test_ep_id = task["test_snapshot"]["episode_id"]
    evolution_ep_ids = task["evolution_snapshots"]

    test_ep = snapshots[test_ep_id]
    test_trajectory = test_ep["snapshot"]["trajectory"]
    pre_dp_traj = test_trajectory[:decision_point]

    # Build primer (pre-dp history)
    primer_lines = []
    for s in pre_dp_traj:
        primer_lines.append(f"Obs {s['step']}: {s['obs']}")
        if s.get("inv"):
            primer_lines.append(f"Inv {s['step']}: {s['inv']}")
        primer_lines.append(f"Act {s['step']}: {s['action']}")
    primer = ""
    if primer_lines:
        primer = "=== Episode History (before decision point) ===\n" + "\n".join(primer_lines)

    # Build RAG step index from evolution snapshots
    entries, embeddings = build_step_index(evolution_ep_ids, snapshots)

    ob, info = restore_game_state(env, test_trajectory, decision_point)

    log_path = output_dir / f"{task_id}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Task: {task_id}\n")
        f.write(f"Baseline: rag\n")
        f.write(f"Game: {game}, Decision point: {decision_point}\n")
        f.write(f"Test episode: {test_ep_id}\n")
        f.write(f"Evolution snapshots ({len(evolution_ep_ids)}): {evolution_ep_ids}\n")
        f.write(f"Step index size: {len(entries)} steps\n\n")
        f.write(f"=== Restored game state at step {decision_point} ===\n")
        f.write(f"OBS: {ob}\n\n")

    agent_fn = baseline.make_agent_fn(primer, entries, embeddings)
    trajectory = []
    cur_ob, cur_info = ob, info

    with open(log_path, "a", encoding="utf-8") as f:
        for step in range(decision_point, STEP_LIMIT):
            action, raw, insights = agent_fn(cur_ob, cur_info.get("inv", ""), step)

            f.write(f"[Step {step}] OBS: {cur_ob[:120]}\n")
            if insights:
                f.write(f"           INSIGHTS: {insights[:200]}\n")
            f.write(f"           RAW: {raw[:200]}\n")
            f.write(f"           ACTION: {action}\n")

            trajectory.append({
                "step": step,
                "obs": cur_ob,
                "inv": cur_info.get("inv", ""),
                "action": action,
                "raw_response": raw,
                "rag_insights": insights,
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
        "baseline": "rag",
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
    parser.add_argument("--top_k", default=5, type=int,
                        help="Number of steps to retrieve per query")
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
        output_dir = RESULTS_ROOT / args.game / "rag" / model_slug / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running RAG on {args.game} ({len(tasks)} tasks)")
    print(f"Model: {args.model}, top_k={args.top_k}")
    print(f"Output: {output_dir}")

    baseline = RAGBaseline(model=args.model, temperature=args.temperature,
                           top_k=args.top_k)
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
            "baseline": "rag",
            "model": args.model,
            "top_k": args.top_k,
            "n_tasks": len(tasks),
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Results saved to {results_path}")


if __name__ == "__main__":
    main()
