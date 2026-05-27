#!/usr/bin/env python3
"""
AlfWorld Evolution Evaluation runner — RAG baseline

Step-level RAG: each step in evolution snapshot trajectories is indexed
as a (obs, action) pair. At each decision step:
  1. Retrieve top-k relevant (obs, action) pairs via cosine similarity
  2. LLM call to synthesize insights from retrieved steps (temp=0)
  3. ReAct decision call using insights as context

AlfWorld differences from JTTL version:
  - env.step() returns (obs, done, info) — no reward
  - No inventory field; steps have no score field
  - Per-task env (different game_file per test episode)
  - Scoring: info["progress"] (0-1) and info["won"]
  - Encoder uses cpu
"""

import json
import time
import argparse
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
# Step-level index
# ---------------------------------------------------------------------------

@dataclass
class StepEntry:
    ep_id: str
    step: int
    obs: str
    action: str
    embedding: np.ndarray = None

    def to_text(self) -> str:
        return f"[{self.ep_id} step {self.step}]\nObs: {self.obs}\nAction: {self.action}"


def build_step_index(evolution_ep_ids: list[str], snapshots: dict) -> list[StepEntry]:
    entries = []
    for ep_id in evolution_ep_ids:
        ep = snapshots.get(ep_id)
        if ep is None:
            continue
        traj = ep["snapshot"]["trajectory"]
        for step in traj:
            if not step.get("action"):
                continue
            entries.append(StepEntry(
                ep_id=ep_id,
                step=step["step"],
                obs=step["obs"],
                action=step["action"],
            ))
    if entries:
        embs = _encode([e.obs for e in entries])
        for entry, emb in zip(entries, embs):
            entry.embedding = emb
    return entries


def retrieve_steps(query_obs: str, index: list[StepEntry], top_k: int = 5) -> list[StepEntry]:
    if not index:
        return []
    query_emb = _encode([query_obs])[0]
    scored = sorted(index, key=lambda e: float(np.dot(query_emb, e.embedding)), reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Insight synthesis
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM_PROMPT = (
    "You are an expert analyst of household task episodes. "
    "Given relevant past observations and actions, extract concise actionable insights."
)

SYNTHESIS_PROMPT_TEMPLATE = """\
The following are relevant steps retrieved from past household task episodes.
Current observation: {current_obs}

Retrieved steps:
{retrieved_text}

Write 2-3 concise, actionable insights for what to do (or avoid) next,
based on what worked or failed in these past steps.
Focus on actions relevant to the current observation.
"""


def synthesize_insights(current_obs: str, retrieved: list[StepEntry], model: str) -> str:
    if not retrieved:
        return ""
    retrieved_text = "\n\n".join(e.to_text() for e in retrieved)
    prompt = SYNTHESIS_PROMPT_TEMPLATE.format(
        current_obs=current_obs[:300],
        retrieved_text=retrieved_text,
    )
    res = chat_completion_with_retries(
        model=model,
        sys_prompt=SYNTHESIS_SYSTEM_PROMPT,
        prompt=prompt,
        max_tokens=200,
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    if res and res.choices:
        return res.choices[0].message.content.strip()
    return ""


# ---------------------------------------------------------------------------
# RAG baseline
# ---------------------------------------------------------------------------

class RAGBaseline:
    def __init__(self, model: str, temperature: float = 0.4, top_k: int = 5):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k

    def make_agent_fn(self, step_index: list[StepEntry], pre_dp_traj: list[dict]):
        primer_lines = []
        for s in pre_dp_traj:
            primer_lines.append(f"Obs {s['step']}: {s['obs']}")
            if s.get("action"):
                primer_lines.append(f"Act {s['step']}: {s['action']}")
        primer = "\n".join(primer_lines)

        scratchpad: list[str] = []

        def agent_fn(obs: str, step: int):
            scratchpad.append(f"Obs {step}: {obs}")

            # Retrieve + synthesize
            retrieved = retrieve_steps(obs, step_index, top_k=self.top_k)
            insights = synthesize_insights(obs, retrieved, self.model)
            insights_block = (
                f"=== Insights from Past Episodes ===\n{insights}\n\n"
                if insights else ""
            )

            user_prompt = ""
            if insights_block:
                user_prompt += insights_block
            if primer:
                user_prompt += "=== Episode History (before decision point) ===\n"
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

def run_task(task: dict, snapshots: dict, baseline: RAGBaseline,
             output_dir: Path) -> dict:
    task_id = task["id"]
    task_short = task["task"]
    decision_point = task["test_snapshot"]["decision_point"]
    test_ep_id = task["test_snapshot"]["episode_id"]
    evolution_ep_ids = task["evolution_snapshots"]

    test_ep = snapshots[test_ep_id]
    test_trajectory = test_ep["snapshot"]["trajectory"]
    pre_dp_traj = test_trajectory[:decision_point]

    print(f"  Building step index from {len(evolution_ep_ids)} evolution snapshots...")
    step_index = build_step_index(evolution_ep_ids, snapshots)
    print(f"  Step index: {len(step_index)} entries")

    game_file = get_game_file(test_ep_id)
    env = AlfWorldEnv(game_file=game_file, step_limit=STEP_LIMIT)

    ob, info = restore_game_state(env, test_trajectory, decision_point)

    log_path = output_dir / f"{task_id}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Task: {task_id}\n")
        f.write(f"Baseline: rag\n")
        f.write(f"Task type: {task_short}, Decision point: {decision_point}\n")
        f.write(f"Test episode: {test_ep_id}\n")
        f.write(f"Evolution snapshots: {evolution_ep_ids}\n")
        f.write(f"Step index size: {len(step_index)}\n\n")
        f.write(f"=== Restored state at step {decision_point} ===\n")
        f.write(f"OBS: {ob}\n\n")

    agent_fn = baseline.make_agent_fn(step_index, pre_dp_traj)
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
        "baseline": "rag",
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
        "step_index_size": len(step_index),
        "trajectory_from_decision_point": trajectory,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASK_SHORT_LIST)
    parser.add_argument("--model", default="openai/gpt-4.1")
    parser.add_argument("--temperature", default=0.4, type=float)
    parser.add_argument("--top_k", default=5, type=int, help="Steps to retrieve per decision")
    parser.add_argument("--task_ids", nargs="*")
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
        output_dir = RESULTS_ROOT / args.task / "rag" / model_slug / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running RAG on {args.task} ({len(tasks)} tasks)")
    print(f"Model: {args.model}, top_k={args.top_k}")
    print(f"Output: {output_dir}")

    baseline = RAGBaseline(model=args.model, temperature=args.temperature, top_k=args.top_k)

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
            "baseline": "rag",
            "model": args.model,
            "top_k": args.top_k,
            "n_tasks": len(tasks),
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Results saved to {results_path}")


if __name__ == "__main__":
    main()
