#!/usr/bin/env python3
"""
Evolution Evaluation runner - EvoTest

EvoTest baseline:
  - Process each evolution snapshot through EvoTest's _evolve() LLM call,
    updating the guiding prompt and state extractor code sequentially.
  - Use the resulting evolved prompt to play from decision_point onwards.
  - Agent sees: evolved system prompt + test trajectory up to decision_point.

Reuses helpers from run_non_evolution.py (env setup, game state restore,
episode runner) and core logic from EvoTest's OurAgent (_evolve, action gen).
"""

import os
import json
import time
import argparse
import re
from pathlib import Path

from utils import (
    DATASET_ROOT, ROM_DIR, RESULTS_ROOT, MAX_CONTEXT_TOKENS, STEP_LIMIT, GAMES,
    SYSTEM_PROMPT, ACTION_FORMAT_INSTRUCTION,
    load_snapshots, format_trajectory, parse_action,
    restore_game_state, run_episode_from_decision_point,
    JerichoEnv, chat_completion_with_retries, truncate_text, game_file,
)

INITIAL_GUIDING_PROMPT = "Explore systematically and examine objects to make progress."
INITIAL_CODE = "def extract_state(game_history):\n    return 'Game in progress.'\n"


# ---------------------------------------------------------------------------
# EvoTest core: evolve and action generation
# (extracted from OurAgent, stripped of UCB tree / node management)
# ---------------------------------------------------------------------------

def evolve(cur_prompt: str, cur_code: str, game_history_str: str, evo_model: str, evo_temperature: float) -> tuple[str, str]:
    """
    Call the evolution LLM to produce an improved guiding prompt and state
    extractor code from the given game history. Returns (new_prompt, new_code).
    Falls back to current values on failure.
    """
    game_history_str = truncate_text(game_history_str, max_tokens=28000)

    sys_prompt = (
        "You are an expert at text adventure games. Your goal is to analyze the existing "
        "prompt, state extractor code, and game history, and generate a better prompt and "
        "state extractor code that will help an LLM agent achieve higher scores."
    )

    user_prompt = f'''
Generate a new improved guiding prompt and state extractor code for a text adventure game agent.

The LLM agent used the following guiding prompt:
"{cur_prompt}"

Here is the history of that game session:
--- GAME HISTORY START ---
{game_history_str}
--- GAME HISTORY END ---

PART 1: Generate a new improved guiding prompt. Consider:
1. Identify useful actions that led to score increases. Give step-by-step instructions.
2. Discourage actions that led to negative outcomes or getting stuck.
3. List possible next areas to explore and brainstorm next attempts.

PART 2: Generate a state extractor Python function in a <code>...</code> block that
summarizes what milestones the agent has completed so far, based on the game history string.

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
    # Strip <think> blocks first so their content doesn't interfere with tag search
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
    """Run state extractor code on formatted game history."""
    try:
        ns = {}
        exec(code, ns)
        history_str = _format_game_history_for_evo(game_history)
        return str(ns["extract_state"](history_str))
    except Exception:
        return ""


def _format_game_history_for_evo(history: list[dict]) -> str:
    """Format in-episode history list for the state extractor."""
    lines = ["GAME HISTORY:"]
    for i, entry in enumerate(history):
        lines.append(f"Step {i+1}:")
        lines.append(f"STATE: {entry.get('state', '')}")
        if entry.get("action"):
            lines.append(f"ACTION TAKEN: {entry['action']}")
    return "\n".join(lines)


def _format_trajectory_for_evo(trajectory: list[dict]) -> str:
    """Format a snapshot trajectory as game history string for _evolve()."""
    lines = ["GAME HISTORY:"]
    for step in trajectory:
        lines.append(f"Step {step['step'] + 1}:")
        lines.append(f"STATE: {step['obs']}")
        if step.get("inv"):
            lines.append(f"INVENTORY: {step['inv']}")
        if step.get("action"):
            lines.append(f"ACTION TAKEN: {step['action']}")
        if step.get("score") is not None:
            lines.append(f"SCORE: {step['score']}")
        lines.append("------------")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cross-episode memory: extract positive/negative examples from trajectories
# ---------------------------------------------------------------------------

SCORE_UP_RE = re.compile(r"\[your score has just gone up", re.IGNORECASE)

_WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "twenty-five": 25, "fifty": 50, "hundred": 100,
}

def _parse_delta(obs: str) -> int:
    """Parse score delta from 'gone up by X point(s)' — handles digits and English words."""
    m = re.search(r"gone up by ([\w-]+)", obs, re.IGNORECASE)
    if not m:
        return 1
    raw = m.group(1).strip().lower()
    if raw.isdigit():
        return int(raw)
    return _WORD_TO_NUM.get(raw, 1)


def _extract_positive_from_traj(trajectory: list[dict]) -> list[dict]:
    """
    Offline equivalent of CrossEpisodeMemory.add_positive():
    step[N].obs contains score-up text → step[N-1] is the (state, action) that caused it.
    """
    positives = []
    for i, step in enumerate(trajectory):
        if SCORE_UP_RE.search(step.get("obs", "")) and i > 0:
            prev = trajectory[i - 1]
            positives.append({
                "state": prev["obs"][:400],
                "action": prev.get("action", ""),
                "delta_score": _parse_delta(step["obs"]),
            })
    return positives


def _extract_negative_from_traj(trajectory: list[dict], min_loop_len: int = 4) -> list[dict]:
    """
    Offline equivalent of CrossEpisodeMemory.add_negative():
    detect consecutive zero-gain segments where actions repeat (loop_zero_gain).
    """
    negatives = []
    n = len(trajectory)
    i = 0
    while i < n:
        # find runs where score doesn't increase and actions repeat
        seg_start = i
        seen_actions: dict[str, int] = {}
        j = i
        while j < n:
            obs = trajectory[j].get("obs", "")
            if SCORE_UP_RE.search(obs):
                break  # score went up, end of zero-gain run
            action = trajectory[j].get("action", "").strip().lower()
            if action:
                seen_actions[action] = seen_actions.get(action, 0) + 1
            j += 1
        seg_len = j - seg_start
        repeated = {a: c for a, c in seen_actions.items() if c >= 2}
        if seg_len >= min_loop_len and repeated:
            segment = trajectory[seg_start:j]
            negatives.append({
                "reason": "loop_zero_gain",
                "length": seg_len,
                "states": [s["obs"][:200] for s in segment[-10:]],
                "actions": [s.get("action", "") for s in segment[-10:]],
            })
        i = j + 1  # skip past the score-up step and continue
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
        lines.append(f"ACTION: {ex['action']}  (gained +{ex['delta_score']} pts)")
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
    """
    Simulates EvoTest's learning process on pre-collected evolution snapshots:
    - Sequentially calls evolve() to update guiding prompt + state extractor.
    - Builds CrossEpisodeMemory (positive/negative examples) from snapshots.
    - Uses evolved prompt + memory during test episode action generation.
    """

    def __init__(self, model: str, evo_model: str, temperature: float = 0.4, evo_temperature: float = 0.7):
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
        """
        Process evolution snapshots sequentially through evolve().
        Also builds positive/negative memory from each snapshot.
        Optionally persists to mem_path / neg_mem_path (one entry per line, JSONL).
        Returns (guiding_prompt, state_extractor_code, positives, negatives).
        """
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

    def build_primer(self, test_trajectory: list[dict]) -> str:
        """Format pre-dp trajectory as a fixed primer (same format as rolling scratchpad)."""
        lines = []
        for s in test_trajectory:
            lines.append(f"Obs {s['step']}: {s['obs']}")
            if s.get("inv"):
                lines.append(f"Inv {s['step']}: {s['inv']}")
            lines.append(f"Act {s['step']}: {s['action']}")
        return "\n".join(lines)

    def make_agent_fn(self, guiding_prompt: str, code: str, primer: str,
                      positives: list[dict], negatives: list[dict]):
        in_episode_history = []  # for state extractor
        scratchpad = []          # full rolling Obs/Inv/Thought/Act history post-dp
        neg_block = _format_negative_block(negatives)

        def agent_fn(obs: str, inv: str, step: int):
            scratchpad.append(f"Obs {step}: {obs}")
            if inv:
                scratchpad.append(f"Inv {step}: {inv}")

            state_summary = _extract_state(code, in_episode_history)
            state_line = f"GAME STATE SUMMARY: {state_summary}\n\n" if state_summary else ""

            few_shot = _format_few_shot(_retrieve_similar_positives(positives, obs))
            few_shot_block = f"SUCCESSFUL ACTIONS FROM PAST EPISODES:\n{few_shot}\n\n" if few_shot != "(none)" else ""

            neg_section = f"FAILURE PATTERNS TO AVOID:\n{neg_block}\n\n" if neg_block else ""

            episode_history = ""
            if primer:
                episode_history += "=== Episode History (before decision point) ===\n"
                episode_history += primer + "\n\n"
            episode_history += "=== Current Episode ===\n"
            episode_history += "\n".join(scratchpad)

            sys_prompt = (
                "You are an expert player aiming to complete a text-based adventure game. "
                "Points are given for making progress in the game.\n\n"
                f"Follow this guide: {guiding_prompt}"
            )

            user_prompt = (
                state_line
                + few_shot_block
                + neg_section
                + episode_history
                + "\nRespond using EXACTLY this format (two lines, nothing else):\n"
                + "Thought: <one sentence of reasoning>\n"
                + "Act: <short game command>\n"
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

            # Parse Thought + Act (same as ReAct)
            thought, action = "", "look"
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
# Main runner
# ---------------------------------------------------------------------------

def run_task(task: dict, snapshots: dict, baseline: EvoTestBaseline, output_dir: Path, env: JerichoEnv) -> dict:
    game = task["game"]
    task_id = task["id"]
    decision_point = task["test_snapshot"]["decision_point"]
    test_ep_id = task["test_snapshot"]["episode_id"]
    evolution_ep_ids = task["evolution_snapshots"]

    test_trajectory = snapshots[test_ep_id]["snapshot"]["trajectory"]
    truncated_test_traj = test_trajectory[:decision_point]

    evolution_trajs = [
        (ep_id, snapshots[ep_id]["snapshot"]["trajectory"])
        for ep_id in evolution_ep_ids
    ]
    evolution_episodes = [snapshots[ep_id] for ep_id in evolution_ep_ids]

    # Evolve prompt and build cross-episode memory from snapshots
    guiding_prompt, code, positives, negatives = baseline.evolve_from_snapshots(
        evolution_trajs, evolution_episodes,
        mem_path=output_dir / f"{task_id}_mem.jsonl",
        neg_mem_path=output_dir / f"{task_id}_neg_mem.jsonl",
    )
    primer = baseline.build_primer(truncated_test_traj)

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
        f.write(f"=== Restored game state at decision_point {decision_point} ===\n")
        f.write(f"OBS: {ob}\n\n")

    agent_fn = baseline.make_agent_fn(guiding_prompt, code, primer, positives, negatives)
    trajectory_from_dp = run_episode_from_decision_point(
        env, agent_fn, ob, info, decision_point, log_path
    )

    first_step = trajectory_from_dp[0] if trajectory_from_dp else {}
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
        "final_score": trajectory_from_dp[-1]["score_after"] if trajectory_from_dp else None,
        "trajectory_from_decision_point": trajectory_from_dp,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", required=True, choices=["balances", "detective", "library", "temple", "zork1", "zork3"])
    parser.add_argument("--model", default="openai/gpt-4.1", help="Agent LLM model")
    parser.add_argument("--evo_model", default="openai/gpt-4.1", help="Evolution LLM model")
    parser.add_argument("--temperature", default=0.4, type=float)
    parser.add_argument("--evo_temperature", default=0.7, type=float)
    parser.add_argument("--task_ids", nargs="*", help="Specific task IDs to run (default: all)")
    parser.add_argument("--distances", nargs="*", type=int, default=None,
                        help="Only run tasks with these distances (default: all). Example: --distances 1 5")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    ee_path = DATASET_ROOT / args.game / "evolution_evaluation.json"
    with open(ee_path) as f:
        ee_data = json.load(f)
    snapshots = load_snapshots()  # load all games to support T3 cross-game snapshots

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

    print(f"Running evotest on {args.game} ({len(tasks)} tasks)")
    print(f"Agent model: {args.model}, Evo model: {args.evo_model}")
    print(f"Output: {output_dir}")

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
                results.append({"task_id": task["id"], "error": str(e)})
    finally:
        env.close()

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
