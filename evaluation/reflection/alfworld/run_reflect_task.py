"""
run_reflect_task.py

Runs the Reflection Task evaluation for a given model on AlfWorld snapshots.
For each snapshot, queries the model with 3 questions:
  Q1 (Detection):    Does this episode have room for improvement?
  Q2 (Localization): Where should the agent improve? (up to 3 step ranges)
  Q3 (Diagnosis):    What went wrong in steps X-Y? (one per core failure)

Usage:
    conda run -n Fraud python run_reflect_task.py \\
        --model claude-sonnet-4-6 \\
        --api anthropic \\
        --games pick_and_place pick_clean \\
        --output_dir output

    conda run -n Fraud python run_reflect_task.py \\
        --model gpt-4.1 \\
        --api openai \\
        --games all \\
        --output_dir output
"""

import argparse
import json
import math
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
from openai import OpenAI

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
FINAL_DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "final_dataset" / "alfworld"

ALL_GAMES = [
    "look_at_obj",
    "pick_and_place",
    "pick_clean",
    "pick_cool",
    "pick_heat",
    "pick_two",
]

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "local")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")


# ── Trajectory formatting ─────────────────────────────────────────────────────

def format_trajectory(trajectory: list[dict]) -> str:
    lines = []
    for step in trajectory:
        obs = step["obs"].replace("\n", " ").strip()
        action = step.get("action", "").strip()
        progress = step.get("progress", 0.0)
        if step["step"] == 0 or not action:
            # Step 0 has no action — only the room description and task instruction
            lines.append(f"[Step {step['step']}] Obs: {obs} | Progress: {progress}")
        else:
            lines.append(
                f"[Step {step['step']}] Obs: {obs} | Action: {action} | Progress: {progress}"
            )
    return "\n".join(lines)


# ── Prompt rendering ──────────────────────────────────────────────────────────

def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def render(template: str, **kwargs) -> str:
    for k, v in kwargs.items():
        template = template.replace("{{" + k + "}}", str(v))
    return template


# ── API calls ─────────────────────────────────────────────────────────────────

def call_anthropic(model: str, system: str, user: str) -> str:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    msg = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return msg.content[0].text


def call_openai(model: str, system: str, user: str) -> str:
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    resp = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content


def call_model(api: str, model: str, system: str, user: str) -> str:
    if api == "anthropic":
        return call_anthropic(model, system, user)
    else:
        return call_openai(model, system, user)


# ── Response parsing ──────────────────────────────────────────────────────────

def parse_json(text: str):
    """Extract JSON from model response (handles <think> tags and markdown code blocks)."""
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1)
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None


def parse_q1(raw: str) -> str | None:
    parsed = parse_json(raw)
    if isinstance(parsed, dict):
        ans = parsed.get("answer", "").lower()
        if ans in ("yes", "no"):
            return ans
    lower = raw.lower()
    if '"yes"' in lower or lower.strip() == "yes":
        return "yes"
    if '"no"' in lower or lower.strip() == "no":
        return "no"
    return None


def parse_q2(raw: str) -> list[list[int]] | None:
    parsed = parse_json(raw)
    if not isinstance(parsed, list):
        return None
    ranges = []
    for item in parsed[:3]:
        if isinstance(item, dict):
            try:
                s = int(item.get("step_start", item.get("start", 0)))
                e = int(item.get("step_end", item.get("end", s)))
                ranges.append([s, e])
            except (TypeError, ValueError):
                continue
    return ranges if ranges else None


def parse_q3(raw: str) -> dict | None:
    parsed = parse_json(raw)
    if isinstance(parsed, dict) and "failure_type" in parsed and "description" in parsed:
        return {"failure_type": str(parsed["failure_type"]), "description": str(parsed["description"])}
    return None


# ── Per-snapshot evaluation ───────────────────────────────────────────────────

def evaluate_snapshot(ep: dict, api: str, model: str, system_prompt: str,
                       q1_tmpl: str, q2_tmpl: str, q3_tmpl: str) -> dict:
    snapshot = ep["snapshot"]
    traj_str = format_trajectory(snapshot["trajectory"])
    game = ep["game"]
    task_type = ep.get("task_type", game)
    final_score = snapshot["final_score"]
    max_score = snapshot["max_score"]

    common_kwargs = dict(game=game, task_type=task_type,
                         final_score=final_score, max_score=max_score,
                         trajectory=traj_str)

    result = {
        "id": ep["id"],
        "game": game,
        "task_type": task_type,
        "source_model": ep["model"],
        "final_score": final_score,
        "max_score": max_score,
    }

    # Q1
    try:
        raw_q1 = call_model(api, model, system_prompt, render(q1_tmpl, **common_kwargs))
        result["q1"] = {"raw": raw_q1, "parsed": parse_q1(raw_q1)}
    except Exception as e:
        result["q1"] = {"raw": None, "parsed": None, "error": str(e)}

    # Q2
    try:
        raw_q2 = call_model(api, model, system_prompt, render(q2_tmpl, **common_kwargs))
        result["q2"] = {"raw": raw_q2, "parsed": parse_q2(raw_q2)}
    except Exception as e:
        result["q2"] = {"raw": None, "parsed": None, "error": str(e)}

    # Q3 — one call per core failure
    core_failures = ep["failure_instances"].get("core_failure", [])
    q3_results = []
    for cf in core_failures:
        where = cf["where"]
        try:
            raw_q3 = call_model(
                api, model, system_prompt,
                render(q3_tmpl, step_start=where[0], step_end=where[1], **common_kwargs),
            )
            q3_results.append({
                "where": where,
                "gt_type": cf["type"],
                "gt_diagnosis": cf.get("diagnosis", ""),
                "raw": raw_q3,
                "parsed": parse_q3(raw_q3),
            })
        except Exception as e:
            q3_results.append({
                "where": where,
                "gt_type": cf["type"],
                "gt_diagnosis": cf.get("diagnosis", ""),
                "raw": None,
                "parsed": None,
                "error": str(e),
            })
    result["q3"] = q3_results

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def load_episodes(games: list[str], sample_frac: float = 1.0, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    episodes = []
    for game in games:
        path = FINAL_DATASET_DIR / game / "snapshots.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        game_eps = data["episodes"]
        if sample_frac < 1.0:
            k = math.ceil(len(game_eps) * sample_frac)
            game_eps = rng.sample(game_eps, k)
        episodes.extend(game_eps)
    return episodes


def has_parse_failure(r: dict) -> bool:
    """Return True if any Q2 or Q3 item failed to parse."""
    if not r["q2"].get("parsed"):
        return True
    if any(not q.get("parsed") for q in r.get("q3", [])):
        return True
    return False


def retry_episode(r: dict, episodes_by_id: dict, api: str, model: str,
                  system_prompt: str, q1_tmpl: str, q2_tmpl: str, q3_tmpl: str) -> dict:
    """Re-run only the failed questions for a result entry."""
    ep = episodes_by_id.get(r["id"])
    if ep is None:
        return r

    snapshot = ep["snapshot"]
    traj_str = format_trajectory(snapshot["trajectory"])
    common_kwargs = dict(game=ep["game"],
                         task_type=ep.get("task_type", ep["game"]),
                         final_score=snapshot["final_score"],
                         max_score=snapshot["max_score"],
                         trajectory=traj_str)

    if not r["q2"].get("parsed"):
        try:
            raw = call_model(api, model, system_prompt, render(q2_tmpl, **common_kwargs))
            r["q2"] = {"raw": raw, "parsed": parse_q2(raw)}
        except Exception as e:
            r["q2"]["error"] = str(e)

    new_q3 = []
    for q in r.get("q3", []):
        if q.get("parsed"):
            new_q3.append(q)
            continue
        where = q["where"]
        try:
            raw = call_model(api, model, system_prompt,
                             render(q3_tmpl, step_start=where[0], step_end=where[1], **common_kwargs))
            q["raw"] = raw
            q["parsed"] = parse_q3(raw)
        except Exception as e:
            q["error"] = str(e)
        new_q3.append(q)
    r["q3"] = new_q3
    return r


def run(args):
    games = ALL_GAMES if args.games == ["all"] else args.games

    system_prompt = load_prompt("system_alfworld.md")
    q1_tmpl = load_prompt("q1_user.md")
    q2_tmpl = load_prompt("q2_user.md")
    q3_tmpl = load_prompt("q3_user.md")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_slug = args.model.replace("/", "_").replace(":", "_")
    output_path = output_dir / f"{model_slug}_results.json"

    # ── Retry mode ────────────────────────────────────────────────────────────
    if args.retry and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)
        results = data["results"]
        episodes = load_episodes(games)
        episodes_by_id = {ep["id"]: ep for ep in episodes}
        to_retry = [r for r in results if has_parse_failure(r)]
        print(f"Retry mode: {len(to_retry)}/{len(results)} episodes have parse failures")

        lock = threading.Lock()
        completed = 0

        def retry_and_save(r):
            nonlocal completed
            new_r = retry_episode(r, episodes_by_id, args.api, args.model,
                                  system_prompt, q1_tmpl, q2_tmpl, q3_tmpl)
            with lock:
                for i, existing in enumerate(results):
                    if existing["id"] == new_r["id"]:
                        results[i] = new_r
                        break
                completed += 1
                q2_ok = bool(new_r["q2"].get("parsed"))
                q3_ok = sum(1 for q in new_r.get("q3", []) if q.get("parsed"))
                print(f"  [{completed}/{len(to_retry)}] {new_r['id']} | Q2={q2_ok} Q3={q3_ok}/{len(new_r.get('q3',[]))}")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump({"model": args.model, "api": args.api,
                               "games": games, "results": results}, f,
                              indent=2, ensure_ascii=False)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(retry_and_save, r) for r in to_retry]
            for fut in futures:
                fut.result()

        print(f"\nRetry done. Results saved to: {output_path}")
        return

    # ── Normal mode ───────────────────────────────────────────────────────────
    episodes = load_episodes(games, sample_frac=args.sample_frac)
    print(f"Loaded {len(episodes)} episodes from {games}")

    results = []
    lock = threading.Lock()
    completed = 0

    def process(ep: dict):
        nonlocal completed
        r = evaluate_snapshot(ep, args.api, args.model, system_prompt,
                              q1_tmpl, q2_tmpl, q3_tmpl)
        with lock:
            results.append(r)
            completed += 1
            q1_ok = r["q1"].get("parsed") or "err"
            q2_n = len(r["q2"].get("parsed") or [])
            q3_n = len(r["q3"])
            print(f"  [{completed}/{len(episodes)}] {r['id']} | Q1={q1_ok} Q2={q2_n}ranges Q3={q3_n}items")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({"model": args.model, "api": args.api,
                           "games": games, "results": results}, f,
                          indent=2, ensure_ascii=False)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process, ep) for ep in episodes]
        for fut in futures:
            fut.result()

    print(f"\nDone. Results saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name (e.g. claude-sonnet-4-6, gpt-4.1)")
    parser.add_argument("--api", choices=["anthropic", "openai"], default="openai",
                        help="API backend to use")
    parser.add_argument("--games", nargs="+", default=["all"],
                        help=f"Games to evaluate (default: all). Valid: {ALL_GAMES}")
    parser.add_argument("--output_dir", default="output",
                        help="Directory to save results")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent requests (default: 8)")
    parser.add_argument("--retry", action="store_true",
                        help="Retry only episodes with parse failures in existing output file")
    parser.add_argument("--sample_frac", type=float, default=1.0,
                        help="Fraction of episodes to sample per game, stratified (default: 1.0)")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
