"""
annotate_episodes.py

Rule-based annotation of LLM agent errors in Jericho text adventure games.
Uses the game walkthrough as ground truth to annotate each episode.

For each episode, produces:
  - whether: did the agent fail to reach maximum score?
  - where:   the first score milestone the agent missed (game context)
  - what:    the correct action needed to achieve that milestone

Usage:
    conda run -n Fraud python annotate_episodes.py \
        --game detective \
        --run_dir ../EvoTest/output/detective/our/qwen3-32b/<timestamp> \
        --output annotations_detective.json
"""

import argparse
import json
import os
import re
from pathlib import Path

import jericho
from error_classifier import classify_errors


ROM_DIR = Path(__file__).resolve().parent.parent / "EvoTest" / "jericho-games"


# ── Walkthrough analysis ───────────────────────────────────────────────────────

def extract_score_events(game: str) -> list[dict]:
    """
    Replay the game walkthrough and record every action that increases the score.
    Returns a list of milestone dicts, ordered by occurrence.
    """
    rom_path = str(ROM_DIR / f"{game}.z5")
    env = jericho.FrotzEnv(rom_path)
    walkthrough = env.get_walkthrough()

    ob, info = env.reset()
    prev_score = info.get("score", 0)
    milestones = []

    for step_idx, action in enumerate(walkthrough):
        ob, reward, done, info = env.step(action)
        score = info.get("score", 0)
        if score > prev_score:
            milestones.append({
                "walkthrough_step": step_idx,
                "action": action,
                "score_before": prev_score,
                "score_after": score,
                "delta": score - prev_score,
                "observation_after": ob.strip(),
            })
            prev_score = score
        if done:
            break

    env.close()
    return milestones


# ── Episode log parsing ────────────────────────────────────────────────────────

def parse_episode_log(log_path: Path) -> list[dict]:
    """
    Parse a Jericho episode log file into a list of step dicts:
      {step, obs, action, reward, cum_score}
    """
    text = log_path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"={5,}", text)

    steps = []
    for block in blocks:
        step_m   = re.search(r"\[STEP\]\s*(\d+)", block)
        obs_m    = re.search(r"\[OBS\]\s*(.*?)(?=\n-{3,}|\Z)", block, re.S)
        action_m = re.search(r"\[CHOSEN_ACTION\]\s*(.+)", block)
        reward_m = re.search(r"\[REWARD\]\s*(-?\d+(?:\.\d+)?)", block)
        score_m  = re.search(r"\[CUM_REWARD\]\s*(-?\d+(?:\.\d+)?)", block)

        if not step_m:
            continue
        steps.append({
            "step":      int(step_m.group(1)),
            "obs":       obs_m.group(1).strip() if obs_m else "",
            "action":    action_m.group(1).strip() if action_m else "",
            "reward":    float(reward_m.group(1)) if reward_m else 0.0,
            "cum_score": float(score_m.group(1)) if score_m else 0.0,
        })

    return sorted(steps, key=lambda x: x["step"])


# ── Annotation logic ───────────────────────────────────────────────────────────

def annotate_episode(trajectory: list[dict], milestones: list[dict]) -> dict:
    """
    Compare agent trajectory against walkthrough milestones.

    Returns:
      whether  – bool, True if the agent failed to reach max score
      where    – the first missed milestone (with surrounding trajectory context)
      what     – the correct action and expected score gain for that milestone
    """
    if not trajectory:
        return {"whether": True, "where": None, "what": None, "note": "empty trajectory"}

    final_score = trajectory[-1]["cum_score"]
    max_score   = milestones[-1]["score_after"] if milestones else 0

    # Whether
    whether = final_score < max_score

    if not whether:
        return {
            "whether": False,
            "final_score": final_score,
            "max_score": max_score,
            "where": None,
            "what": None,
        }

    # Find the first milestone the agent never reached
    first_missed = None
    for m in milestones:
        if m["score_after"] > final_score:
            first_missed = m
            break

    # Where: find the last step where the agent was at its peak score,
    # i.e., the point after which it made no further progress
    peak_score = final_score
    last_peak_step = None
    for step in trajectory:
        if step["cum_score"] >= peak_score:
            peak_score = step["cum_score"]
            last_peak_step = step

    # Collect a few steps around the stall point for context
    stall_idx = last_peak_step["step"] if last_peak_step else 0
    context_window = [s for s in trajectory if abs(s["step"] - stall_idx) <= 5]

    where = {
        "stall_at_step": stall_idx,
        "agent_score_at_stall": peak_score,
        "next_required_milestone_score": first_missed["score_after"] if first_missed else None,
        "context_steps": context_window,
    }

    what = {
        "correct_action": first_missed["action"] if first_missed else None,
        "score_gain": first_missed["delta"] if first_missed else None,
        "observation_after_correct_action": first_missed["observation_after"] if first_missed else None,
        "walkthrough_step": first_missed["walkthrough_step"] if first_missed else None,
    }

    return {
        "whether": whether,
        "final_score": final_score,
        "max_score": max_score,
        "where": where,
        "what": what,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def annotate_run(game: str, run_dir: Path, output_path: Path, max_episodes: int = 30):
    print(f"Extracting walkthrough milestones for '{game}'...")
    milestones = extract_score_events(game)
    print(f"  Found {len(milestones)} score-giving actions. Max score: {milestones[-1]['score_after']}")
    print(f"  Milestones: {[(m['action'], m['score_after']) for m in milestones]}\n")

    log_files = sorted(run_dir.glob("episode_*.txt"))[:max_episodes]
    if not log_files:
        print(f"No episode log files found in {run_dir}")
        return

    results = {
        "game": game,
        "run_dir": str(run_dir),
        "max_score": milestones[-1]["score_after"] if milestones else 0,
        "milestones": milestones,
        "episodes": [],
    }

    for log_file in log_files:
        ep_num = int(re.search(r"episode_(\d+)", log_file.name).group(1))
        trajectory = parse_episode_log(log_file)
        annotation = annotate_episode(trajectory, milestones)
        annotation["episode"] = ep_num
        annotation["log_file"] = log_file.name
        annotation["n_steps"] = len(trajectory)
        annotation["errors"] = classify_errors(trajectory, milestones)
        results["episodes"].append(annotation)

        status = "✓" if not annotation["whether"] else f"✗ (stall@{annotation.get('where', {}).get('stall_at_step', '?')} score={annotation.get('final_score', '?')})"
        missed_action = annotation.get("what", {}) or {}
        missed = missed_action.get("correct_action", "-")
        err_summary = annotation["errors"]["summary"]["by_type"]
        err_str = ", ".join(f"{k.split('/')[-1]}:{v}" for k, v in err_summary.items()) or "none"
        print(f"  Episode {ep_num:03d}: {status} | first_missed='{missed}' | errors=[{err_str}]")

    # Summary
    n_success = sum(1 for e in results["episodes"] if not e["whether"])
    print(f"\nSummary: {n_success}/{len(results['episodes'])} episodes reached max score.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Annotations saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", required=True, choices=["detective","library","zork1","zork3","balances","temple"])
    parser.add_argument("--run_dir", required=True, type=Path, help="Path to a run's output directory with episode_*.txt files")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path (default: annotations/<game>_annotations.json)")
    parser.add_argument("--max_episodes", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = args.output or Path(__file__).parent / "output" / f"{args.game}_annotations.json"
    annotate_run(args.game, args.run_dir, output, args.max_episodes)
