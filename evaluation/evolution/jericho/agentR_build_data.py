#!/usr/bin/env python3
"""
Agent-R: Build revision trajectories from Jericho game snapshots.

For each snapshot (error trajectory):
  1. Step-by-step LLM error identification — find the first 'bad' action
  2. Restore Jericho env to that state
  3. Collect corrected continuation via ReAct agent
  4. Filter: keep only examples where corrected_score > original_score
  5. Format as multi-turn chat: error context in first user msg, corrections as assistant turns

Output: JSONL consumed by agentR_finetune.py

Usage:
  python agentR_build_data.py --game zork1 --model Qwen/Qwen3-32B \\
      --output agentR_data/zork1.jsonl --n_workers 4

Reference: Yuan et al., "Agent-R: Training Language Model Agents to Reflect
           via Iterative Self-Training", arXiv 2501.11425
"""

import json
import traceback
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import (
    DATASET_ROOT, ROM_DIR, STEP_LIMIT, GAMES,
    load_snapshots, restore_game_state,
    JerichoEnv, chat_completion_with_retries, truncate_text, game_file,
)
from run_react import REACT_SYSTEM_PROMPT, REACT_FORMAT, parse_react_response

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "agentR_data"

MAX_CORRECTION_STEPS = 40
MAX_HISTORY_LINES = 60   # verifier prompt: last N history lines
MAX_CHECK_STEPS = 50     # max steps to scan for error identification

VERIFIER_SYSTEM = (
    "You are an expert at text adventure games. "
    "Given a game history and a specific action, judge whether the action was a mistake."
)

VERIFIER_PROMPT = """\
=== Game history up to this step ===
{history}

=== Action taken ===
{action}

=== Result ===
Reward: {reward}
Next observation: {next_obs}

Was this action a mistake? Reply with exactly one word: good, uncertain, or bad.
Answer:"""


# ---------------------------------------------------------------------------
# Error identification (Agent-R step-by-step verifier)
# ---------------------------------------------------------------------------

def identify_error_step(
    trajectory: list[dict],
    model: str,
    max_check: int = MAX_CHECK_STEPS,
) -> int | None:
    """
    Check each action sequentially until the first 'bad' judgment.
    Two consecutive 'uncertain' judgments also count as 'bad'.
    Returns trajectory index of the first error, or None if not found.
    """
    uncertain_count = 0

    for i in range(min(len(trajectory), max_check)):
        # Build recent history context (truncated)
        history_steps = trajectory[max(0, i - 20):i]
        history_lines = []
        for s in history_steps:
            history_lines.append(f"[Step {s['step']}] Obs: {s['obs'][:150]}")
            if s.get("inv"):
                history_lines.append(f"           Inv: {s['inv'][:60]}")
            history_lines.append(f"           Act: {s['action']}")
        history_text = "\n".join(history_lines) or "(start of game)"

        action = trajectory[i]["action"]
        reward = trajectory[i].get("reward", 0)
        next_obs = trajectory[i + 1]["obs"][:200] if i + 1 < len(trajectory) else "(game ended)"

        prompt = VERIFIER_PROMPT.format(
            history=history_text,
            action=action,
            reward=reward,
            next_obs=next_obs,
        )

        res = chat_completion_with_retries(
            model=model,
            sys_prompt=VERIFIER_SYSTEM,
            prompt=truncate_text(prompt, 6000),
            max_tokens=10,
            temperature=0.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        if not res or not res.choices:
            continue

        judgment = res.choices[0].message.content.strip().lower()

        if "bad" in judgment:
            return i
        elif "uncertain" in judgment:
            uncertain_count += 1
            if uncertain_count >= 2:
                return i
        else:
            uncertain_count = 0

    return None


# ---------------------------------------------------------------------------
# Corrected continuation via ReAct
# ---------------------------------------------------------------------------

def collect_corrected_continuation(
    trajectory: list[dict],
    error_step: int,
    model: str,
    rom_path: str,
    temperature: float = 0.7,
) -> list[dict]:
    """
    Restore game state to error_step, then run ReAct to collect corrected path.
    Each call creates its own JerichoEnv (safe for ThreadPoolExecutor).
    """
    env = JerichoEnv(rom_path=rom_path, seed=0, step_limit=STEP_LIMIT, get_valid=False)
    try:
        ob, info = restore_game_state(env, trajectory, error_step)

        # Pre-error context for the prompt (last 60 lines)
        ctx_lines = []
        for s in trajectory[:error_step]:
            ctx_lines.append(f"Obs {s['step']}: {s['obs'][:150]}")
            if s.get("inv"):
                ctx_lines.append(f"Inv {s['step']}: {s['inv'][:60]}")
            ctx_lines.append(f"Act {s['step']}: {s['action']}")
        pre_error_context = "\n".join(ctx_lines[-MAX_HISTORY_LINES:])

        scratchpad: list[str] = []
        correction: list[dict] = []

        for step_offset in range(MAX_CORRECTION_STEPS):
            step_num = error_step + step_offset

            scratchpad.append(f"Obs {step_num}: {ob}")
            if info.get("inv"):
                scratchpad.append(f"Inv {step_num}: {info['inv']}")

            user_prompt = ""
            if pre_error_context:
                user_prompt += "=== Game History (before correction point) ===\n"
                user_prompt += pre_error_context + "\n\n"
            user_prompt += "=== Current Episode (correct from here) ===\n"
            user_prompt += "\n".join(scratchpad)
            user_prompt += REACT_FORMAT

            res = chat_completion_with_retries(
                model=model,
                sys_prompt=REACT_SYSTEM_PROMPT,
                prompt=truncate_text(user_prompt, 16000),
                max_tokens=256,
                temperature=temperature,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw = res.choices[0].message.content if res and res.choices else ""
            thought, action = parse_react_response(raw)

            scratchpad.append(f"Thought {step_num}: {thought}")
            scratchpad.append(f"Act {step_num}: {action}")

            new_ob, reward, done, new_info = env.step(action)

            correction.append({
                "step": step_num,
                "obs": ob,
                "inv": info.get("inv", ""),
                "thought": thought,
                "action": action,
                "reward": reward,
                "score_after": new_info.get("score", 0),
            })

            ob, info = new_ob, new_info
            if done:
                break

        return correction
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Format revision example
# ---------------------------------------------------------------------------

def format_revision_example(
    episode_id: str,
    game: str,
    trajectory: list[dict],
    error_step: int,
    correction: list[dict],
) -> dict:
    """
    Build a multi-turn chat training example.

    Structure:
      system: ReAct system prompt
      user:   [error trajectory as context] + [obs at error step] + [format instruction]
      assistant: [corrected thought + action]      ← loss computed here
      user:   [next obs + format instruction]
      assistant: ...
      ...
    """
    # Error trajectory context (everything before the bad action)
    ctx_lines = []
    for s in trajectory[:error_step]:
        ctx_lines.append(f"[Step {s['step']}] Obs: {s['obs'][:200]}")
        if s.get("inv"):
            ctx_lines.append(f"           Inv: {s['inv'][:80]}")
        ctx_lines.append(f"           Act: {s['action']}")
    context_text = "\n".join(ctx_lines) if ctx_lines else "(game start)"

    # Observation at the error step (the state where correction begins)
    err_obs = trajectory[error_step]["obs"] if error_step < len(trajectory) else correction[0]["obs"]
    err_inv = (trajectory[error_step].get("inv", "") if error_step < len(trajectory)
               else correction[0].get("inv", ""))

    first_user = (
        f"=== Past Game History ===\n{context_text}\n\n"
        f"=== Current State (correction starts here) ===\n"
        f"Obs {error_step}: {err_obs}"
        + (f"\nInv {error_step}: {err_inv}" if err_inv else "")
        + REACT_FORMAT
    )

    messages = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": first_user},
    ]

    for i, s in enumerate(correction):
        messages.append({
            "role": "assistant",
            "content": f"Thought: {s['thought']}\nAct: {s['action']}",
        })
        if i + 1 < len(correction):
            nxt = correction[i + 1]
            messages.append({
                "role": "user",
                "content": (
                    f"Obs {nxt['step']}: {nxt['obs']}"
                    + (f"\nInv {nxt['step']}: {nxt['inv']}" if nxt.get("inv") else "")
                    + REACT_FORMAT
                ),
            })

    return {
        "episode_id": episode_id,
        "game": game,
        "error_step": error_step,
        "original_score": float(trajectory[-1].get("score_after", 0) if trajectory else 0),
        "corrected_score": float(correction[-1]["score_after"]) if correction else 0.0,
        "n_correction_steps": len(correction),
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# Process one snapshot (called from worker thread)
# ---------------------------------------------------------------------------

def process_snapshot(
    ep_id: str,
    episode: dict,
    game: str,
    model: str,
    rom_path: str,
    max_check: int,
) -> dict | None:
    snap = episode["snapshot"]
    trajectory = snap["trajectory"]
    original_score = float(snap.get("final_score", 0) or 0)

    if len(trajectory) < 5:
        return None

    print(f"  [{game}] {ep_id}: identifying error ...", flush=True)
    error_step = identify_error_step(trajectory, model, max_check)
    if error_step is None:
        print(f"  [{game}] {ep_id}: no error found, skipped", flush=True)
        return None

    print(f"  [{game}] {ep_id}: error at step {error_step}, collecting correction ...", flush=True)
    try:
        correction = collect_corrected_continuation(
            trajectory, error_step, model, rom_path
        )
    except Exception as e:
        print(f"  [{game}] {ep_id}: correction failed: {e}", flush=True)
        traceback.print_exc()
        return None

    if not correction:
        return None

    corrected_score = correction[-1]["score_after"]
    if corrected_score <= original_score:
        print(f"  [{game}] {ep_id}: no improvement ({original_score} → {corrected_score}), skipped",
              flush=True)
        return None

    print(f"  [{game}] {ep_id}: OK (score {original_score} → {corrected_score})", flush=True)
    return format_revision_example(ep_id, game, trajectory, error_step, correction)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", required=True, choices=GAMES)
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--output", default=None)
    parser.add_argument("--n_workers", default=4, type=int,
                        help="Parallel worker threads (each spawns its own JerichoEnv)")
    parser.add_argument("--max_check", default=MAX_CHECK_STEPS, type=int,
                        help="Max steps to scan per trajectory for error identification")
    parser.add_argument("--trajs_path", default=None,
                        help="Path to fresh trajectories JSON (from agentR_orig_collect.py). "
                             "If omitted, loads from snapshots.json in the dataset.")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else DATA_DIR / f"{args.game}.jsonl"

    if args.trajs_path:
        with open(args.trajs_path, encoding="utf-8") as f:
            episodes_list = json.load(f)
        snapshots = {ep["id"]: ep for ep in episodes_list}
        print(f"Loaded {len(snapshots)} fresh trajectories from {args.trajs_path}")
    else:
        snapshots = load_snapshots(args.game)
    rom_path = str(ROM_DIR / game_file(args.game))
    ep_ids = list(snapshots.keys())

    print(f"Building revision trajectories for {args.game}: {len(ep_ids)} episodes")
    print(f"Model: {args.model}  workers: {args.n_workers}  max_check: {args.max_check}")
    print(f"Output: {output_path}")

    results = []
    with ThreadPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {
            executor.submit(
                process_snapshot,
                ep_id, snapshots[ep_id], args.game, args.model, rom_path, args.max_check,
            ): ep_id
            for ep_id in ep_ids
        }
        for future in as_completed(futures):
            ep_id = futures[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"  ERROR {ep_id}: {e}", flush=True)
                traceback.print_exc()

    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(results)} / {len(ep_ids)} revision trajectories → {output_path}")


if __name__ == "__main__":
    main()
