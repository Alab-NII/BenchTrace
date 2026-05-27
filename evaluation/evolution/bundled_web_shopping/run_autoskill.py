#!/usr/bin/env python3
"""
BundledWebShopping Evolution Evaluation runner — AutoSkill baseline

AutoSkill: experience-driven lifelong learning via skill self-evolution.
Each past evolution snapshot → LLM-extracted skill (name, description,
instructions, triggers). At inference, top-k skills retrieved by cosine
similarity to the current subtask question.
"""

import re
import json
import time
import argparse
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from utils import (
    DATASET_ROOT, RESULTS_ROOT, GAME_LIST, STEP_LIMIT, MAX_CONTEXT_TOKENS,
    load_snapshots, load_dataset_split, make_env,
    restore_game_state, format_prior_subtasks,
)
from src.openai_helpers import chat_completion_with_retries, truncate_text
from run_react import REACT_SYSTEM_PROMPT, REACT_FORMAT, parse_shopping_response

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
# Skill data structure
# ---------------------------------------------------------------------------

@dataclass
class Skill:
    ep_id: str
    name: str
    description: str
    instructions: str
    triggers: list[str]
    embedding: np.ndarray = field(repr=False)


# ---------------------------------------------------------------------------
# Skill extraction
# ---------------------------------------------------------------------------

SKILL_EXTRACTION_SYSTEM = (
    "You are an expert analyst of bundled web shopping playthroughs. "
    "Extract a concise, reusable selection skill from a past episode."
)

SKILL_EXTRACTION_TEMPLATE = """\
Below is a trajectory from a bundled web shopping episode.
Final score: {final_score:.2f} (1.0 = all subtasks correct)

{trajectory_text}

Extract ONE reusable skill — a generalizable selection strategy learned from this episode.
Focus on WHAT the agent should do (or avoid) when faced with similar product selection tasks.

Respond in EXACTLY this format (no extra text):
name: [2-5 word kebab-case name]
description: [1 sentence: what shopping scenario this skill applies to]
instructions: [2-3 sentences: what selection strategy to use]
triggers: [3 short phrases that indicate this skill is relevant, comma-separated]
"""


def _format_traj_for_extraction(trajectory: list[dict]) -> str:
    lines = []
    for step in trajectory:
        if step.get("action") is None:
            continue
        idx = step.get("subtask_idx", step.get("step", 0) // 2)
        obs = step["obs"][:150]
        action = step.get("action", "")[:100]
        correct = step.get("correct", None)
        lines.append(f"[Subtask {idx+1}] Q: {obs[:80]}")
        lines.append(f"           A: {action[:80]}  {'✓' if correct else '✗'}")
    return "\n".join(lines)


def _parse_skill_response(text: str, ep_id: str, embedding: np.ndarray) -> Skill | None:
    name = description = instructions = ""
    triggers = []
    for line in text.strip().splitlines():
        if m := re.match(r"name:\s*(.+)", line, re.IGNORECASE):
            name = m.group(1).strip()
        elif m := re.match(r"description:\s*(.+)", line, re.IGNORECASE):
            description = m.group(1).strip()
        elif m := re.match(r"instructions:\s*(.+)", line, re.IGNORECASE):
            instructions = m.group(1).strip()
        elif m := re.match(r"triggers:\s*(.+)", line, re.IGNORECASE):
            triggers = [t.strip() for t in m.group(1).split(",") if t.strip()]
    if not name or not instructions:
        return None
    return Skill(ep_id=ep_id, name=name, description=description,
                 instructions=instructions, triggers=triggers, embedding=embedding)


def extract_skill(ep_id: str, trajectory: list[dict], final_score: float,
                  model: str, temperature: float) -> Skill | None:
    traj_text = _format_traj_for_extraction(trajectory)
    prompt = SKILL_EXTRACTION_TEMPLATE.format(
        final_score=final_score, trajectory_text=traj_text,
    )
    res = chat_completion_with_retries(
        model=model, sys_prompt=SKILL_EXTRACTION_SYSTEM, prompt=prompt,
        max_tokens=256, temperature=temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    raw = res.choices[0].message.content if res and res.choices else ""
    embedding = _encode([f"{prompt}\n{raw}"])[0]
    return _parse_skill_response(raw, ep_id, embedding)


# ---------------------------------------------------------------------------
# Skill retrieval
# ---------------------------------------------------------------------------

def retrieve_skills(query: str, skills: list[Skill], top_k: int = 3) -> list[Skill]:
    if not skills:
        return []
    query_emb = _encode([query])[0]
    skill_embs = np.stack([s.embedding for s in skills])
    scores = skill_embs @ query_emb
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [skills[i] for i in top_indices]


def render_skills(skills: list[Skill], max_chars: int = 1500) -> str:
    if not skills:
        return ""
    lines = ["=== Relevant Skills from Past Episodes ==="]
    for s in skills:
        block = (f"\n[{s.name}]\n"
                 f"When to use: {s.description}\n"
                 f"Strategy: {s.instructions}\n")
        if len("\n".join(lines)) + len(block) > max_chars:
            break
        lines.append(block)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# AutoSkill baseline
# ---------------------------------------------------------------------------

class AutoSkillBaseline:
    def __init__(self, model: str, temperature: float = 0.4, top_k: int = 3):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k

    def build_skill_library(self, evolution_ep_ids: list[str], snapshots: dict) -> list[Skill]:
        skills = []
        for ep_id in evolution_ep_ids:
            ep = snapshots[ep_id]
            trajectory = ep["snapshot"]["trajectory"]
            final_score = float(ep["snapshot"].get("final_score", 0) or 0)
            skill = extract_skill(ep_id, trajectory, final_score, self.model, self.temperature)
            if skill:
                skills.append(skill)
                print(f"    Extracted skill [{skill.name}] from {ep_id}")
            else:
                print(f"    Failed to extract skill from {ep_id}")
        return skills

    def make_agent_fn(self, skill_library: list[Skill], pre_dp_context: str,
                      initial_obs: str):
        retrieved = retrieve_skills(initial_obs, skill_library, top_k=self.top_k)
        skill_context = render_skills(retrieved)
        current_run_selections: list[str] = []

        def agent_fn(obs: str, subtask_idx: int):
            user_prompt = ""
            if skill_context:
                user_prompt += skill_context + "\n\n"
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

def run_task(task: dict, snapshots: dict, baseline: AutoSkillBaseline,
             dataset, output_dir: Path) -> dict:
    task_id = task["id"]
    game = task["game"]
    decision_point = task["test_snapshot"]["decision_point"]
    test_ep_id = task["test_snapshot"]["episode_id"]
    evolution_ep_ids = task["evolution_snapshots"]

    test_trajectory = snapshots[test_ep_id]["snapshot"]["trajectory"]
    pre_dp_context = format_prior_subtasks(test_trajectory, decision_point)

    print(f"  Building skill library from {len(evolution_ep_ids)} evolution snapshots...")
    skill_library = baseline.build_skill_library(evolution_ep_ids, snapshots)
    print(f"  Skill library: {len(skill_library)} skills")

    env = make_env(test_ep_id, dataset)
    ob, info = restore_game_state(env, test_trajectory, decision_point)

    log_path = output_dir / f"{task_id}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Task: {task_id}\nBaseline: autoskill\n")
        f.write(f"Game: {game}, Decision point: {decision_point}\n")
        f.write("=== Skills ===\n")
        for s in skill_library:
            f.write(f"  [{s.name}] {s.description}\n")
        f.write("\n")

    agent_fn = baseline.make_agent_fn(skill_library, pre_dp_context, ob)
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
        "task_id": task_id, "baseline": "autoskill", "game": game,
        "type": task["type"], "distance": task["distance"],
        "target_failure_instance": task["target_failure_instance"],
        "test_episode_id": test_ep_id, "decision_point": decision_point,
        "scoring_method": task["scoring_method"],
        "action_at_decision_point": first_step.get("action"),
        "obs_at_decision_point": first_step.get("obs"),
        "final_progress": last_step.get("progress_after"),
        "won": last_step.get("won", False),
        "skills_extracted": [
            {"ep_id": s.ep_id, "name": s.name, "description": s.description,
             "instructions": s.instructions, "triggers": s.triggers}
            for s in skill_library
        ],
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
        output_dir = RESULTS_ROOT / args.game / "autoskill" / model_slug / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running AutoSkill on {args.game} ({len(tasks)} tasks)")
    baseline = AutoSkillBaseline(model=args.model, temperature=args.temperature, top_k=args.top_k)

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
            "game": args.game, "baseline": "autoskill", "model": args.model,
            "n_tasks": len(tasks), "results": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nDone. Results saved to {results_path}")


if __name__ == "__main__":
    main()
