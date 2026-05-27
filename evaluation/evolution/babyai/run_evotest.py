#!/usr/bin/env python3
"""
BabyAI Evolution Evaluation runner — EvoTest baseline

EvoTest: process each evolution snapshot through LLM evolution call,
sequentially updating the guiding prompt and state extractor code.
Also builds cross-episode positive/negative memory.
Uses evolved prompt + memory during test episode.

BabyAI differences from AlfWorld:
  - Actions constrained to 7 discrete strings; env.step() takes integer index
  - env created with BabyAIEnv(level_id, obs_to_reward, seed=instance_seed)
  - instance_seed and level_id looked up via get_seed_and_level()
  - labels loaded via load_labels() for obs_to_reward patterns
  - No score-up observations; positive memory uses won flag (same as AlfWorld)
  - Negative memory uses repeated-action loop detection (same heuristic)
  - Scoring: info["progress"] (0-1) and info["won"]
  - No inventory field
  - _format_trajectory_for_evo skips inv/score fields
"""

import re
import json
import time
import argparse
import traceback
from pathlib import Path

from utils import (
    DATASET_ROOT,
    RESULTS_ROOT,
    MAX_CONTEXT_TOKENS,
    STEP_LIMIT,
    GAME_LIST,
    load_snapshots,
    load_labels,
    restore_game_state,
    get_seed_and_level,
    action_str_to_idx,
)
from src.babyai_env import BabyAIEnv
from src.openai_helpers import chat_completion_with_retries, truncate_text

INITIAL_GUIDING_PROMPT = "Explore the environment systematically and interact with objects to complete the navigation and manipulation task."
INITIAL_CODE = "def extract_state(game_history):\n    return 'Task in progress.'\n"


# ---------------------------------------------------------------------------
# EvoTest core: evolve + parse + validate
# ---------------------------------------------------------------------------

def evolve(cur_prompt: str, cur_code: str, game_history_str: str,
           evo_model: str, evo_temperature: float) -> tuple[str, str]:
    game_history_str = truncate_text(game_history_str, max_tokens=28000)

    sys_prompt = (
        "You are an expert at navigation and manipulation tasks in grid-based environments. "
        "Your goal is to analyze the existing prompt, state extractor code, and episode history, "
        "and generate a better prompt and state extractor code that will help an LLM agent "
        "complete more tasks."
    )

    user_prompt = f'''
Generate a new improved guiding prompt and state extractor code for a grid navigation agent.

The LLM agent used the following guiding prompt:
"{cur_prompt}"

Here is the history of that episode:
--- EPISODE HISTORY START ---
{game_history_str}
--- EPISODE HISTORY END ---

PART 1: Generate a new improved guiding prompt. Consider:
1. Identify useful actions that led to task progress. Give step-by-step instructions.
2. Discourage actions that led to no progress or getting stuck.
3. List possible next areas to explore and brainstorm next attempts.

PART 2: Generate a state extractor Python function in a <code>...</code> block that
summarizes what milestones the agent has completed so far, based on the episode history string.

Format your response with NO additional text:

<prompt>
[Your generated prompt here]
</prompt>
<code>
def extract_state(game_history):
    # [Return a string summarizing the current state]
</code>
'''

    try:
        response = chat_completion_with_retries(
            model=evo_model,
            sys_prompt=sys_prompt,
            prompt=user_prompt,
            max_tokens=3000,
            temperature=evo_temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        full_response = response.choices[0].message.content.strip()
        new_prompt, new_code = _parse_evolution_response(full_response)
        ret_prompt = new_prompt if new_prompt and len(new_prompt) > 10 else cur_prompt
        ret_code = new_code if new_code and _validate_code(new_code) else cur_code
        return ret_prompt, ret_code
    except Exception as e:
        print(f"  [evolve] Error: {e}. Keeping current prompt.")
        return cur_prompt, cur_code


def _parse_evolution_response(response: str) -> tuple[str, str]:
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    code = ""
    if "<code>" in response and "</code>" in response:
        cs = response.find("<code>") + len("<code>")
        ce = response.find("</code>")
        if cs < ce:
            code = response[cs:ce].strip()
            response = response[:response.find("<code>")].strip()
    prompt = ""
    if "<prompt>" in response and "</prompt>" in response:
        ps = response.find("<prompt>") + len("<prompt>")
        pe = response.find("</prompt>")
        prompt = response[ps:pe].strip()
    else:
        prompt = response.strip()
    return prompt, code


def _validate_code(code: str) -> bool:
    try:
        ns = {}
        if "def extract_state(game_history):" not in code:
            return False
        exec(code, ns)
        return callable(ns.get("extract_state")) and ns["extract_state"]("") is not None
    except Exception:
        return False


def _extract_state(code: str, game_history: list[dict]) -> str:
    try:
        ns = {}
        exec(code, ns)
        history_str = _format_game_history_for_evo(game_history)
        return str(ns["extract_state"](history_str))
    except Exception:
        return ""


def _format_game_history_for_evo(history: list[dict]) -> str:
    lines = ["GAME HISTORY:"]
    for i, entry in enumerate(history):
        lines.append(f"Step {i+1}:")
        lines.append(f"STATE: {entry.get('state', '')}")
        if entry.get("action"):
            lines.append(f"ACTION TAKEN: {entry['action']}")
    return "\n".join(lines)


def _format_trajectory_for_evo(trajectory: list[dict]) -> str:
    lines = ["GAME HISTORY:"]
    for step in trajectory:
        lines.append(f"Step {step['step'] + 1}:")
        lines.append(f"STATE: {step['obs']}")
        if step.get("action"):
            lines.append(f"ACTION TAKEN: {step['action']}")
        lines.append("------------")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cross-episode memory: positive/negative extraction
# (BabyAI: no score-up observations; use won flag for positive detection
#  and repeated-action loops for negative detection — same as AlfWorld)
# ---------------------------------------------------------------------------

def _extract_positive_from_traj(trajectory: list[dict]) -> list[dict]:
    """If the episode was won, the last step before done is positive."""
    positives = []
    for i, step in enumerate(trajectory):
        if step.get("won") and i > 0:
            prev = trajectory[i - 1]
            positives.append({
                "state": prev["obs"][:400],
                "action": prev.get("action", ""),
                "delta_score": 1,
            })
            break  # one positive per won episode
    return positives


def _extract_negative_from_traj(trajectory: list[dict], min_loop_len: int = 4) -> list[dict]:
    negatives = []
    n = len(trajectory)
    i = 0
    while i < n:
        seg_start = i
        seen_actions: dict[str, int] = {}
        j = i
        while j < n:
            if trajectory[j].get("won"):
                break
            action = trajectory[j].get("action", "").strip().lower()
            if action:
                seen_actions[action] = seen_actions.get(action, 0) + 1
            j += 1
        seg_len = j - seg_start
        repeated = {a: c for a, c in seen_actions.items() if c >= 2}
        if seg_len >= min_loop_len and repeated:
            segment = trajectory[seg_start:j]
            negatives.append({
                "reason": "loop_no_progress",
                "length": seg_len,
                "states": [s["obs"][:200] for s in segment[-10:]],
                "actions": [s.get("action", "") for s in segment[-10:]],
            })
        i = j + 1
    return negatives


def _tokenize(text: str) -> list[str]:
    return [t for t in (text or "").lower().replace("\n", " ").split() if t.isalnum()]


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _retrieve_similar_positives(positives: list[dict], query_state: str, k: int = 3) -> list[dict]:
    q_tok = _tokenize(query_state)
    scored = sorted(positives, key=lambda ex: _jaccard(q_tok, _tokenize(ex["state"])), reverse=True)
    return [ex for ex in scored[:k] if _jaccard(q_tok, _tokenize(ex["state"])) > 0]


def _format_few_shot(positives: list[dict]) -> str:
    if not positives:
        return "(none)"
    lines = []
    for ex in positives:
        lines.append(f"STATE: {ex['state'][:300]}")
        lines.append(f"ACTION: {ex['action']}")
        lines.append("---")
    return "\n".join(lines)


def _format_negative_block(negatives: list[dict]) -> str:
    if not negatives:
        return ""
    parts = []
    for neg in negatives[-3:]:
        parts.append(f"Failure type: {neg['reason']}")
        parts.append("Actions to avoid repeating: " + ", ".join(neg["actions"][-5:]))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# EvoTest Baseline
# ---------------------------------------------------------------------------

class EvoTestBaseline:
    def __init__(self, model: str, evo_model: str,
                 temperature: float = 0.4, evo_temperature: float = 0.7):
        self.model = model
        self.evo_model = evo_model
        self.temperature = temperature
        self.evo_temperature = evo_temperature

    def evolve_from_snapshots(
        self,
        evolution_trajectories: list[tuple[str, list[dict]]],
        evolution_episodes: list[dict],
        mem_path: Path = None,
        neg_mem_path: Path = None,
    ) -> tuple[str, str, list[dict], list[dict]]:
        prompt = INITIAL_GUIDING_PROMPT
        code = INITIAL_CODE
        all_positives: list[dict] = []
        all_negatives: list[dict] = []

        for i, ((ep_id, traj), ep) in enumerate(zip(evolution_trajectories, evolution_episodes)):
            pos = _extract_positive_from_traj(traj)
            neg = _extract_negative_from_traj(traj)
            history_str = _format_trajectory_for_evo(traj)
            prompt, code = evolve(prompt, code, history_str, self.evo_model, self.evo_temperature)
            all_positives.extend(pos)
            all_negatives.extend(neg)
            print(f"  [evolve {i+1}/{len(evolution_trajectories)}] ep={ep_id} "
                  f"+{len(pos)} pos, +{len(neg)} neg | prompt: {prompt[:80].replace(chr(10),' ')!r}")

        if mem_path:
            with open(mem_path, "w", encoding="utf-8") as f:
                for item in all_positives:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        if neg_mem_path:
            with open(neg_mem_path, "w", encoding="utf-8") as f:
                for item in all_negatives:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        return prompt, code, all_positives, all_negatives

    def build_primer(self, pre_dp_traj: list[dict]) -> str:
        lines = []
        for s in pre_dp_traj:
            lines.append(f"Obs {s['step']}: {s['obs']}")
            if s.get("action"):
                lines.append(f"Act {s['step']}: {s['action']}")
        return "\n".join(lines)

    def make_agent_fn(self, guiding_prompt: str, code: str, primer: str,
                      positives: list[dict], negatives: list[dict]):
        in_episode_history = []
        scratchpad = []
        neg_block = _format_negative_block(negatives)

        def agent_fn(obs: str, step: int):
            scratchpad.append(f"Obs {step}: {obs}")

            state_summary = _extract_state(code, in_episode_history)
            state_line = f"GAME STATE SUMMARY: {state_summary}\n\n" if state_summary else ""

            few_shot = _format_few_shot(_retrieve_similar_positives(positives, obs))
            few_shot_block = (
                f"SUCCESSFUL ACTIONS FROM PAST EPISODES:\n{few_shot}\n\n"
                if few_shot != "(none)" else ""
            )

            neg_section = f"FAILURE PATTERNS TO AVOID:\n{neg_block}\n\n" if neg_block else ""

            episode_history = ""
            if primer:
                episode_history += "=== Episode History (before decision point) ===\n"
                episode_history += primer + "\n\n"
            episode_history += "=== Current Episode ===\n"
            episode_history += "\n".join(scratchpad)

            sys_prompt = (
                "You are an expert at completing navigation and manipulation tasks "
                "in a grid-based environment.\n\n"
                f"Follow this guide: {guiding_prompt}"
            )

            from utils import ACTIONS_PROMPT
            user_prompt = (
                state_line
                + few_shot_block
                + neg_section
                + episode_history
                + f"\nValid actions: {ACTIONS_PROMPT}\n"
                + "\nRespond using EXACTLY this format (two lines, nothing else):\n"
                + "Thought: <one sentence of reasoning>\n"
                + "Act: <one action from the list above>\n"
            )

            res = chat_completion_with_retries(
                model=self.model,
                sys_prompt=sys_prompt,
                prompt=truncate_text(user_prompt, MAX_CONTEXT_TOKENS),
                max_tokens=256,
                temperature=self.temperature,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = res.choices[0].message.content if res and res.choices else ""

            thought, action = "", "move forward"
            for line in raw.strip().splitlines():
                m = re.match(r"Thought:\s*(.+)", line, re.IGNORECASE)
                if m:
                    thought = m.group(1).strip()
                m = re.match(r"Act:\s*(.+)", line, re.IGNORECASE)
                if m:
                    action = m.group(1).strip()

            scratchpad.append(f"Thought {step}: {thought}")
            scratchpad.append(f"Act {step}: {action}")
            in_episode_history.append({"state": obs, "action": action})
            return action, raw

        return agent_fn


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(task: dict, snapshots: dict, baseline: EvoTestBaseline,
             output_dir: Path, labels: dict) -> dict:
    task_id = task["id"]
    game = task["game"]
    decision_point = task["test_snapshot"]["decision_point"]
    test_ep_id = task["test_snapshot"]["episode_id"]
    evolution_ep_ids = task["evolution_snapshots"]

    test_trajectory = snapshots[test_ep_id]["snapshot"]["trajectory"]
    pre_dp_traj = test_trajectory[:decision_point]

    evolution_trajs = [
        (ep_id, snapshots[ep_id]["snapshot"]["trajectory"])
        for ep_id in evolution_ep_ids
    ]
    evolution_episodes = [snapshots[ep_id] for ep_id in evolution_ep_ids]

    guiding_prompt, code, positives, negatives = baseline.evolve_from_snapshots(
        evolution_trajs, evolution_episodes,
        mem_path=output_dir / f"{task_id}_mem.jsonl",
        neg_mem_path=output_dir / f"{task_id}_neg_mem.jsonl",
    )
    primer = baseline.build_primer(pre_dp_traj)

    instance_seed, level_id = get_seed_and_level(test_ep_id)
    obs_to_reward = labels.get(level_id)
    env = BabyAIEnv(level_id, obs_to_reward=obs_to_reward, seed=instance_seed,
                    step_limit=STEP_LIMIT)

    ob, info = restore_game_state(env, test_trajectory, decision_point)

    log_path = output_dir / f"{task_id}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Task: {task_id}\n")
        f.write(f"Baseline: evotest\n")
        f.write(f"Game: {game}, Decision point: {decision_point}\n")
        f.write(f"Test episode: {test_ep_id}\n")
        f.write(f"Evolution snapshots: {evolution_ep_ids}\n")
        f.write(f"Evolved prompt: {guiding_prompt[:200]}\n")
        f.write(f"Positive examples: {len(positives)}, Negative examples: {len(negatives)}\n\n")
        f.write(f"=== Restored state at step {decision_point} ===\n")
        f.write(f"OBS: {ob}\n\n")

    agent_fn = baseline.make_agent_fn(guiding_prompt, code, primer, positives, negatives)
    trajectory = []
    cur_ob, cur_info = ob, info

    try:
        with open(log_path, "a", encoding="utf-8") as f:
            for step in range(decision_point, STEP_LIMIT):
                action, raw = agent_fn(cur_ob, step)
                action_idx = action_str_to_idx(action)

                f.write(f"[Step {step}] OBS: {cur_ob[:120]}\n")
                f.write(f"           RAW: {raw[:200]}\n")
                f.write(f"           ACTION: {action}\n")

                trajectory.append({
                    "step": step,
                    "obs": cur_ob,
                    "action": action,
                    "raw_response": raw,
                })

                cur_ob, done, cur_info = env.step(action_idx)
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
        "baseline": "evotest",
        "game": game,
        "type": task["type"],
        "distance": task["distance"],
        "target_failure_instance": task["target_failure_instance"],
        "test_episode_id": test_ep_id,
        "decision_point": decision_point,
        "scoring_method": task["scoring_method"],
        "evolved_prompt": guiding_prompt,
        "action_at_decision_point": first_step.get("action"),
        "obs_after_decision_point": first_step.get("obs"),
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
    parser.add_argument("--model", default="openai/gpt-4.1", help="Agent LLM model")
    parser.add_argument("--evo_model", default="openai/gpt-4.1", help="Evolution LLM model")
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
    labels = load_labels()

    tasks = ee_data["tasks"]
    if args.task_ids:
        tasks = [t for t in tasks if t["id"] in args.task_ids]
    if args.distances:
        tasks = [t for t in tasks if t["distance"] in args.distances]

    baseline = EvoTestBaseline(
        model=args.model,
        evo_model=args.evo_model,
        temperature=args.temperature,
        evo_temperature=args.evo_temperature,
    )

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_slug = args.model.replace("/", "_")
        output_dir = RESULTS_ROOT / args.game / "evotest" / model_slug / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running EvoTest on {args.game} ({len(tasks)} tasks)")
    print(f"Agent model: {args.model}, Evo model: {args.evo_model}")
    print(f"Output: {output_dir}")

    results = []
    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] {task['id']} "
              f"(type={task['type']}, dist={task['distance']})")
        try:
            result = run_task(task, snapshots, baseline, output_dir, labels)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append({"task_id": task["id"], "error": str(e)})

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({
            "game": args.game,
            "baseline": "evotest",
            "model": args.model,
            "evo_model": args.evo_model,
            "n_tasks": len(tasks),
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Results saved to {results_path}")


if __name__ == "__main__":
    main()
