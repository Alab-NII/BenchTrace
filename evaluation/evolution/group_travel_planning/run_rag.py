#!/usr/bin/env python3
"""
GroupTravelPlanning Evolution Evaluation runner — RAG baseline

Subtask-level RAG: each traveler subtask in evolution trajectories is indexed
as a (constraints, plan) pair. At the decision subtask:
  1. Retrieve top-k relevant (constraint, plan) pairs via cosine similarity
  2. LLM synthesis call to extract planning insights
  3. Plan generation using insights as context
"""

import re
import json
import time
import argparse
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils import (
    DATASET_ROOT, RESULTS_ROOT, GAME_LIST, STEP_LIMIT, MAX_CONTEXT_TOKENS,
    load_snapshots, load_dataset_split, make_env,
    restore_game_state, format_prior_plans,
)
from src.openai_helpers import chat_completion_with_retries, truncate_text
from run_react import REACT_SYSTEM_PROMPT, REACT_FORMAT, parse_travel_response

# ---------------------------------------------------------------------------
# Sentence encoder
# ---------------------------------------------------------------------------

_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda:0")
    return _encoder


def _encode(texts: list[str]) -> np.ndarray:
    return _get_encoder().encode(texts, normalize_embeddings=True, show_progress_bar=False)


# ---------------------------------------------------------------------------
# Subtask index
# ---------------------------------------------------------------------------

@dataclass
class TravelerEntry:
    ep_id: str
    subtask_idx: int
    obs: str       # raw question text (constraints)
    action: str    # generated plan
    score: float   # subtask_progress
    embedding: np.ndarray = None

    def to_text(self) -> str:
        plan_clean = re.sub(r"<think>.*?</think>", "", self.action, flags=re.DOTALL).strip()
        return (f"[{self.ep_id} traveler {self.subtask_idx}] score={self.score:.0%}\n"
                f"Request: {self.obs[:200]}\nPlan: {plan_clean[:300]}")


def build_traveler_index(evolution_ep_ids: list[str], snapshots: dict) -> list[TravelerEntry]:
    entries = []
    for ep_id in evolution_ep_ids:
        ep = snapshots.get(ep_id)
        if ep is None:
            continue
        traj = ep["snapshot"]["trajectory"]
        for step in traj:
            if step.get("action") is None:
                continue
            obs = step.get("obs", "")
            action = step.get("action", "")
            score = float(step.get("subtask_progress", 0) or 0)
            subtask_idx = step.get("subtask_idx", step.get("step", 0) // 2)
            entry = TravelerEntry(
                ep_id=ep_id, subtask_idx=subtask_idx,
                obs=obs, action=action, score=score,
            )
            entries.append(entry)

    if entries:
        texts = [f"{e.obs[:200]} {e.action[:100]}" for e in entries]
        embs = _encode(texts)
        for e, emb in zip(entries, embs):
            e.embedding = emb
    return entries


def retrieve_travelers(query: str, index: list[TravelerEntry], top_k: int = 5) -> list[TravelerEntry]:
    if not index:
        return []
    query_emb = _encode([query])[0]
    embs = np.stack([e.embedding for e in index])
    scores = embs @ query_emb
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [index[i] for i in top_idx]


# ---------------------------------------------------------------------------
# RAG synthesis
# ---------------------------------------------------------------------------

RAG_SYNTHESIS_SYSTEM = (
    "You are an expert group travel planner. "
    "Synthesize insights from past travel planning examples to help with the current request."
)

RAG_SYNTHESIS_TEMPLATE = """\
Current traveler request:
{query}

Retrieved past planning examples:
{examples}

In 2-3 sentences, synthesize actionable planning insights: what constraint types
are present, what strategies worked well, and what to watch out for (especially
JOIN/RELATION constraints referencing prior travelers' specific choices).
"""


def synthesize_insights(query: str, retrieved: list[TravelerEntry],
                        model: str, temperature: float = 0.0) -> str:
    examples_text = "\n---\n".join(e.to_text() for e in retrieved)
    prompt = RAG_SYNTHESIS_TEMPLATE.format(query=query[:300], examples=examples_text)
    res = chat_completion_with_retries(
        model=model, sys_prompt=RAG_SYNTHESIS_SYSTEM, prompt=prompt,
        max_tokens=200, temperature=temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return res.choices[0].message.content.strip() if res and res.choices else ""


# ---------------------------------------------------------------------------
# RAG agent
# ---------------------------------------------------------------------------

class RAGBaseline:
    def __init__(self, model: str, temperature: float = 0.4, top_k: int = 5):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k

    def make_agent_fn(self, index: list[TravelerEntry], pre_dp_context: str):
        current_run_plans: list[tuple[str, str]] = []

        def agent_fn(obs: str, subtask_idx: int):
            retrieved = retrieve_travelers(obs, index, top_k=self.top_k)
            insights = synthesize_insights(obs, retrieved, self.model) if retrieved else ""

            user_prompt = ""
            if insights:
                user_prompt += "=== Insights from Past Episodes ===\n" + insights + "\n\n"
            if pre_dp_context:
                user_prompt += "=== Prior Travelers' Plans (this episode) ===\n"
                user_prompt += pre_dp_context + "\n\n"
            if current_run_plans:
                user_prompt += "=== Plans Generated in This Run ===\n"
                for i, (q, p) in enumerate(current_run_plans):
                    n = subtask_idx - len(current_run_plans) + i + 1
                    user_prompt += f"Traveler {n + 1}: {q[:200]}\nPlan: {p[:400]}\n\n"
            user_prompt += "=== Current Traveler ===\n" + obs + REACT_FORMAT

            res = chat_completion_with_retries(
                model=self.model, sys_prompt=REACT_SYSTEM_PROMPT,
                prompt=truncate_text(user_prompt, MAX_CONTEXT_TOKENS),
                max_tokens=1024, temperature=self.temperature,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = res.choices[0].message.content if res and res.choices else ""
            action = parse_travel_response(raw)
            current_run_plans.append((obs[:200], action))
            return action, raw

        return agent_fn


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(task: dict, snapshots: dict, baseline: RAGBaseline,
             judge_model: str, dataset, output_dir: Path) -> dict:
    task_id = task["id"]
    game = task["game"]
    decision_point = task["test_snapshot"]["decision_point"]
    test_ep_id = task["test_snapshot"]["episode_id"]
    evolution_ep_ids = task["evolution_snapshots"]

    test_trajectory = snapshots[test_ep_id]["snapshot"]["trajectory"]
    pre_dp_context = format_prior_plans(test_trajectory, decision_point)

    print(f"  Building traveler index from {len(evolution_ep_ids)} evolution episodes...")
    index = build_traveler_index(evolution_ep_ids, snapshots)

    env = make_env(test_ep_id, judge_model, dataset)
    ob, info = restore_game_state(env, test_trajectory, decision_point)

    log_path = output_dir / f"{task_id}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Task: {task_id}\nBaseline: rag\n")
        f.write(f"Game: {game}, Decision point: {decision_point}\n")
        f.write(f"Index size: {len(index)} traveler entries\n\n")

    agent_fn = baseline.make_agent_fn(index, pre_dp_context)
    trajectory = []
    cur_ob, cur_info = ob, info
    start_subtask = decision_point // 2

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            for subtask_step in range(start_subtask, start_subtask + STEP_LIMIT):
                action, raw = agent_fn(cur_ob, subtask_step)
                f.write(f"[Traveler {subtask_step}] ACTION: {action[:200]}\n")
                trajectory.append({
                    "subtask_idx": subtask_step, "obs": cur_ob,
                    "action": action, "raw_response": raw,
                })
                cur_ob, done, cur_info = env.step(action)
                trajectory[-1]["progress_after"] = cur_info.get("progress", 0.0)
                trajectory[-1]["won"] = cur_info.get("won", False)
                print(f"  [traveler {subtask_step}] progress={cur_info.get('progress',0):.3f}")
                if done:
                    break
    finally:
        env.close()

    first_step = trajectory[0] if trajectory else {}
    last_step = trajectory[-1] if trajectory else {}
    return {
        "task_id": task_id, "baseline": "rag", "game": game,
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
    parser.add_argument("--judge_model", default=None)
    parser.add_argument("--temperature", default=0.4, type=float)
    parser.add_argument("--top_k", default=5, type=int)
    parser.add_argument("--task_ids", nargs="*")
    parser.add_argument("--distances", nargs="*", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    judge_model = args.judge_model or args.model

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
        output_dir = RESULTS_ROOT / args.game / "rag" / model_slug / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running RAG on {args.game} ({len(tasks)} tasks)")
    baseline = RAGBaseline(model=args.model, temperature=args.temperature, top_k=args.top_k)

    results = []
    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] {task['id']} "
              f"(type={task['type']}, dist={task['distance']})")
        try:
            result = run_task(task, snapshots, baseline, judge_model, dataset, output_dir)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append({"task_id": task["id"], "error": str(e)})

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({
            "game": args.game, "baseline": "rag", "model": args.model,
            "judge_model": judge_model, "n_tasks": len(tasks), "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nDone. Results saved to {results_path}")


if __name__ == "__main__":
    main()
