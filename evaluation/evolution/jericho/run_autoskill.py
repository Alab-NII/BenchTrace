#!/usr/bin/env python3
"""
Evolution Evaluation runner - AutoSkill baseline

AutoSkill (Yang et al., 2025): experience-driven lifelong learning via skill
self-evolution. Each past episode is compressed into a reusable "skill" by an
LLM, stored with a semantic embedding, and retrieved at inference time via
cosine similarity.

Adaptation for EvolEval:
  Build (SKILL EXTRACTION):
    Each evolution snapshot is passed to an LLM that extracts one reusable
    strategy skill: a name, description (when to use), instructions (what to
    do), and trigger phrases.

  Retrieve:
    At episode start, embed the initial observation with BAAI/bge-base-en-v1.5.
    Score each skill by cosine similarity and retrieve the top-k most relevant.

  Agent:
    ReAct-style rolling scratchpad. Relevant skills prepended as context.

Reference: "AutoSkill: Experience-Driven Lifelong Learning via Skill
           Self-Evolution", Yang et al., arXiv 2603.01145.
           https://github.com/ECNU-ICALK/AutoSkill
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
    DATASET_ROOT, ROM_DIR, RESULTS_ROOT, MAX_CONTEXT_TOKENS, STEP_LIMIT, GAMES,
    load_snapshots, restore_game_state,
    JerichoEnv, chat_completion_with_retries, truncate_text, game_file,
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
        _encoder = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
    return _encoder

def _encode(texts: list[str]) -> np.ndarray:
    enc = _get_encoder()
    return enc.encode(texts, normalize_embeddings=True, show_progress_bar=False)

# ---------------------------------------------------------------------------
# Skill data structure
# ---------------------------------------------------------------------------

@dataclass
class Skill:
    ep_id: str
    name: str
    description: str        # when to use this skill
    instructions: str       # what to do
    triggers: list[str]
    embedding: np.ndarray = field(repr=False)

# ---------------------------------------------------------------------------
# Skill extraction prompts
# ---------------------------------------------------------------------------

SKILL_EXTRACTION_SYSTEM = (
    "You are an expert analyst of text-based adventure game playthroughs. "
    "Your task is to extract a concise, reusable strategy skill from a past episode."
)

SKILL_EXTRACTION_TEMPLATE = """\
Below is a trajectory from a text adventure game.
Final score: {final_score} / {max_score}

{trajectory_text}

Extract ONE reusable skill — a generalizable strategy learned from this episode.
Focus on WHAT the agent should do (or avoid) in situations like those encountered here.

Respond in EXACTLY this format (no extra text):
name: [2-5 word kebab-case name]
description: [1 sentence: what game situation this skill applies to]
instructions: [2-3 sentences: what strategy to use in this situation]
triggers: [3 short phrases that indicate this skill is relevant, comma-separated]
"""

# ---------------------------------------------------------------------------
# Skill extraction
# ---------------------------------------------------------------------------

def _format_trajectory_for_extraction(trajectory: list[dict], max_steps: int = 30) -> str:
    """Format a trajectory excerpt for the extraction LLM."""
    lines = []
    # Use the last max_steps steps (most informative)
    for s in trajectory[-max_steps:]:
        lines.append(f"[Step {s['step']}] Obs: {s['obs'][:120]}")
        if s.get("inv"):
            lines.append(f"           Inv: {s['inv'][:80]}")
        lines.append(f"           Act: {s['action']}")
    return "\n".join(lines)


def _parse_skill_response(text: str, ep_id: str, embedding: np.ndarray) -> Skill | None:
    """Parse LLM skill extraction response into a Skill object."""
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


def extract_skill(
    ep_id: str,
    trajectory: list[dict],
    final_score: float,
    max_score: int,
    model: str,
    temperature: float,
) -> Skill | None:
    """Extract a skill from a single evolution snapshot via LLM."""
    traj_text = _format_trajectory_for_extraction(trajectory)
    prompt = SKILL_EXTRACTION_TEMPLATE.format(
        final_score=final_score,
        max_score=max_score,
        trajectory_text=traj_text,
    )
    res = chat_completion_with_retries(
        model=model,
        sys_prompt=SKILL_EXTRACTION_SYSTEM,
        prompt=prompt,
        max_tokens=256,
        temperature=temperature,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    raw = res.choices[0].message.content if res and res.choices else ""

    # Embed the skill using description + instructions as the semantic anchor
    skill_text = f"{prompt}\n{raw}"
    embedding = _encode([skill_text])[0]

    return _parse_skill_response(raw, ep_id, embedding)

# ---------------------------------------------------------------------------
# Skill retrieval
# ---------------------------------------------------------------------------

def retrieve_skills(query: str, skills: list[Skill], top_k: int = 3) -> list[Skill]:
    """Retrieve top-k skills by cosine similarity to query."""
    if not skills:
        return []
    query_emb = _encode([query])[0]
    skill_embs = np.stack([s.embedding for s in skills])
    scores = skill_embs @ query_emb  # cosine sim (embeddings are normalized)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [skills[i] for i in top_indices]

# ---------------------------------------------------------------------------
# Skill rendering
# ---------------------------------------------------------------------------

def render_skills(skills: list[Skill], max_chars: int = 2000) -> str:
    """Render retrieved skills as a text block for injection into the prompt."""
    if not skills:
        return ""
    lines = ["=== Relevant Skills from Past Episodes ==="]
    for s in skills:
        block = (
            f"\n[{s.name}]\n"
            f"When to use: {s.description}\n"
            f"Strategy: {s.instructions}\n"
        )
        if len("\n".join(lines)) + len(block) > max_chars:
            break
        lines.append(block)
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# AutoSkill baseline
# ---------------------------------------------------------------------------

GAME_MAX_SCORE = {
    "balances": 15, "detective": 290, "library": 19,
    "temple": 5, "zork1": 45, "zork3": 3,
}


class AutoSkillBaseline:
    """
    ReAct agent augmented with AutoSkill: skills extracted from evolution
    snapshots are retrieved by semantic similarity and injected into the prompt.
    """

    def __init__(self, model: str, temperature: float = 0.4, top_k: int = 3):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k

    def build_skill_library(
        self,
        evolution_ep_ids: list[str],
        snapshots: dict,
        game: str,
    ) -> list[Skill]:
        """Extract one skill per evolution snapshot."""
        max_score = GAME_MAX_SCORE.get(game, 1)
        skills = []
        for ep_id in evolution_ep_ids:
            ep = snapshots[ep_id]
            trajectory = ep["snapshot"]["trajectory"]
            final_score = ep["snapshot"].get("final_score", 0) or 0
            skill = extract_skill(
                ep_id=ep_id,
                trajectory=trajectory,
                final_score=final_score,
                max_score=max_score,
                model=self.model,
                temperature=self.temperature,
            )
            if skill:
                skills.append(skill)
                print(f"    Extracted skill [{skill.name}] from {ep_id}")
            else:
                print(f"    Failed to extract skill from {ep_id}")
        return skills

    def make_agent_fn(self, initial_history: list[dict], skill_library: list[Skill], initial_obs: str):
        """
        initial_history: pre-dp test trajectory
        skill_library: extracted skills from evolution snapshots
        initial_obs: first observation at decision point (used for retrieval)
        """
        # Retrieve once at episode start using the initial observation
        retrieved = retrieve_skills(initial_obs, skill_library, top_k=self.top_k)
        skill_context = render_skills(retrieved)

        # Build read-only primer from pre-dp history
        primer_lines = []
        for s in initial_history:
            primer_lines.append(f"Obs {s['step']}: {s['obs']}")
            if s.get("inv"):
                primer_lines.append(f"Inv {s['step']}: {s['inv']}")
            primer_lines.append(f"Act {s['step']}: {s['action']}")
        primer = "\n".join(primer_lines)

        scratchpad: list[str] = []

        def agent_fn(obs: str, inv: str, step: int):
            scratchpad.append(f"Obs {step}: {obs}")
            if inv:
                scratchpad.append(f"Inv {step}: {inv}")

            user_prompt = ""
            if skill_context:
                user_prompt += skill_context + "\n\n"
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

def run_task(task: dict, snapshots: dict, baseline: AutoSkillBaseline,
             output_dir: Path, env: JerichoEnv) -> dict:
    game = task["game"]
    task_id = task["id"]
    decision_point = task["test_snapshot"]["decision_point"]
    test_ep_id = task["test_snapshot"]["episode_id"]
    evolution_ep_ids = task["evolution_snapshots"]

    test_ep = snapshots[test_ep_id]
    test_trajectory = test_ep["snapshot"]["trajectory"]
    truncated_test_traj = test_trajectory[:decision_point]

    print(f"  Building skill library from {len(evolution_ep_ids)} evolution snapshots...")
    skill_library = baseline.build_skill_library(evolution_ep_ids, snapshots, game)
    print(f"  Skill library: {len(skill_library)} skills")

    ob, info = restore_game_state(env, test_trajectory, decision_point)

    log_path = output_dir / f"{task_id}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Task: {task_id}\n")
        f.write(f"Baseline: autoskill\n")
        f.write(f"Game: {game}, Decision point: {decision_point}\n")
        f.write(f"Test episode: {test_ep_id}\n")
        f.write(f"Evolution snapshots: {evolution_ep_ids}\n\n")
        f.write(f"=== Extracted Skills ===\n")
        for s in skill_library:
            f.write(f"  [{s.name}] {s.description}\n")
        f.write(f"\n=== Restored game state at step {decision_point} ===\n")
        f.write(f"OBS: {ob}\n\n")

    agent_fn = baseline.make_agent_fn(truncated_test_traj, skill_library, ob)

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
        "baseline": "autoskill",
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
    parser.add_argument("--game", required=True, choices=GAMES)
    parser.add_argument("--model", default="openai/gpt-4.1")
    parser.add_argument("--temperature", default=0.4, type=float)
    parser.add_argument("--top_k", default=3, type=int, help="Skills to retrieve per task")
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
        output_dir = RESULTS_ROOT / args.game / "autoskill" / model_slug / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running AutoSkill on {args.game} ({len(tasks)} tasks)")
    print(f"Model: {args.model}, top_k={args.top_k}")
    print(f"Output: {output_dir}")

    baseline = AutoSkillBaseline(model=args.model, temperature=args.temperature, top_k=args.top_k)
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
                traceback.print_exc()
                results.append({"task_id": task["id"], "error": str(e)})
    finally:
        env.close()

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({
            "game": args.game,
            "baseline": "autoskill",
            "model": args.model,
            "top_k": args.top_k,
            "n_tasks": len(tasks),
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Results saved to {results_path}")


if __name__ == "__main__":
    main()
