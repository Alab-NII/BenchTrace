"""
llm_judge.py

LLM-as-judge for Q3 (Diagnosis) on BundledWebShopping snapshots.
For each Q3 item, judge whether the model's description matches the GT diagnosis,
given a context window of trajectory steps around the failure.

Scoring:
  2 — same failure, core behavior consistent
  1 — partially correct: related problem but inaccurate or incomplete
  0 — completely different failure

Usage:
    conda run -n Fraud python llm_judge.py \
        --results output/qwen3-32b_results.json \
        [--judge_model claude-sonnet-4-6] \
        [--workers 10] \
        [--context_window 5]
"""

import argparse
import json
import os
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
FINAL_DATASET_DIR = Path(__file__).resolve().parent.parent.parent / "final_dataset" / "bundled_web_shopping"

SYSTEM_PROMPT = """You are evaluating whether a model's diagnosis of an agent's failure on a BundledWebShopping multi-subtask web shopping task matches the ground truth diagnosis.

You will be given:
1. A trajectory excerpt showing the agent's actions, observations, and progress around the failure
2. The ground truth diagnosis (one sentence)
3. The model's diagnosis (one sentence)

Score the match on a 0–2 scale:
  2 = Same failure: the core problematic behavior is the same, even if worded differently
  1 = Partially correct: describes a related issue but is inaccurate, incomplete, or misattributes the root cause
  0 = Different failure: describes a completely unrelated problem

Focus on whether the *core failure behavior* matches, not surface wording.

Respond with JSON only: {"score": 0|1|2, "reason": "one sentence explanation"}"""


MAX_OBS_CHARS = 1200


def _truncate(text: str, n: int) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 3] + "..."


def format_excerpt(trajectory: list[dict], step_start: int, step_end: int, window: int) -> str:
    if not trajectory:
        return ""
    lo = max(0, step_start - window)
    hi = min(trajectory[-1]["step"], step_end + window)
    lines = []
    for step in trajectory:
        s = step["step"]
        if lo <= s <= hi:
            obs = _truncate(step.get("obs", ""), MAX_OBS_CHARS)
            action = (step.get("action") or "").strip()
            progress = step.get("progress", 0.0)
            subtask = step.get("subtask_idx")
            correct = step.get("correct")
            meta = f"subtask={subtask}"
            if correct is not None:
                meta += f" correct={correct}"
            marker = " ◄" if step_start <= s <= step_end else ""
            if s == 0 or not action:
                lines.append(f"[Step {s}] ({meta}) Obs: {obs} | Progress: {progress}{marker}")
            else:
                lines.append(f"[Step {s}] ({meta}) Obs: {obs} | Action: {action} | Progress: {progress}{marker}")
    return "\n".join(lines)


def call_judge(client: anthropic.Anthropic, model: str,
               excerpt: str, gt: str, pred: str, step_start: int, step_end: int) -> dict | None:
    user = f"""Agent trajectory around steps {step_start}–{step_end} (marked with ◄):

<trajectory_excerpt>
{excerpt}
</trajectory_excerpt>

Ground truth diagnosis: {gt}

Model diagnosis: {pred}"""

    for attempt in range(5):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=256,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user}],
            )
            text = msg.content[0].text
            m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
            if m:
                text = m.group(1)
            parsed = json.loads(text.strip())
            score = int(parsed["score"])
            assert score in (0, 1, 2)
            return {"score": score, "reason": str(parsed.get("reason", ""))}
        except anthropic.RateLimitError:
            if attempt < 4:
                time.sleep(2 ** attempt * 10)
        except Exception as e:
            return {"score": None, "error": str(e)}
    return {"score": None, "error": "rate limit after 5 retries"}


def load_trajectories(games: list[str]) -> dict[str, list[dict]]:
    """Returns {episode_id: trajectory}."""
    traj_map = {}
    for game in games:
        path = FINAL_DATASET_DIR / game / "snapshots.json"
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for ep in data["episodes"]:
            traj_map[ep["id"]] = ep["snapshot"]["trajectory"]
    return traj_map


def run(args):
    with open(args.results, encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]
    games = list({r["game"] for r in results})
    print(f"Loading trajectories for {games}...")
    traj_map = load_trajectories(games)

    items = []
    for ri, r in enumerate(results):
        traj = traj_map.get(r["id"])
        if traj is None:
            continue
        for qi, q in enumerate(r.get("q3", [])):
            if not q.get("parsed"):
                continue
            if "judge" in q and q["judge"].get("score") is not None and not args.redo:
                continue
            items.append((ri, qi, r["id"], q["where"],
                          q["gt_diagnosis"], q["parsed"]["description"], traj))

    print(f"Items to judge: {len(items)}")
    if not items:
        print("Nothing to judge.")
        return

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    lock = threading.Lock()
    completed = 0

    def judge_item(item):
        nonlocal completed
        ri, qi, ep_id, where, gt, pred, traj = item
        excerpt = format_excerpt(traj, where[0], where[1], args.context_window)
        result = call_judge(client, args.judge_model, excerpt, gt, pred, where[0], where[1])

        with lock:
            results[ri]["q3"][qi]["judge"] = result
            completed += 1
            score = result.get("score") if result else "err"
            if completed % 20 == 0 or completed == len(items):
                print(f"  [{completed}/{len(items)}] {ep_id} steps {where} → score={score}")
            if completed % 50 == 0 or completed == len(items):
                with open(args.results, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(judge_item, item) for item in items]
        for fut in futures:
            fut.result()

    with open(args.results, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("\n── Judge Results ──")
    all_scores = [r["q3"][qi].get("judge", {}).get("score")
                  for r in results for qi, q in enumerate(r.get("q3", []))
                  if q.get("parsed") and q.get("judge", {}).get("score") is not None]
    n = len(all_scores)
    for s in (2, 1, 0):
        cnt = all_scores.count(s)
        print(f"  score={s}: {cnt}/{n} ({cnt/n:.1%})" if n else f"  score={s}: 0/0")
    avg = sum(all_scores) / n if n else 0
    print(f"  avg score: {avg:.3f}  (n={n})")

    print("\n── Per-game avg score ──")
    by_game = defaultdict(list)
    for r in results:
        for q in r.get("q3", []):
            s = q.get("judge", {}).get("score")
            if s is not None:
                by_game[r["game"]].append(s)
    for game in sorted(by_game):
        scores = by_game[game]
        print(f"  {game}: {sum(scores)/len(scores):.3f}  (n={len(scores)})")

    print(f"\nResults saved to: {args.results}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to results JSON")
    parser.add_argument("--judge_model", default="claude-sonnet-4-6")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--context_window", type=int, default=5,
                        help="Steps before/after failure range to include (default: 5)")
    parser.add_argument("--redo", action="store_true",
                        help="Re-judge already-judged items")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
