"""
ai_annotate.py

AI-based annotation of Jericho episodes using two AI models (Claude + Gemini).
Reads representative snapshots from JTTL/dataset/<game>/ and uses the
pre-computed trajectory and all_errors fields as input.

Usage:
    conda run -n Fraud python ai_annotate.py \
        --game zork1 \
        [--dataset_dir ../../dataset] \
        [--output_dir output/zork1] \
        [--task jericho]
"""

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import os

import anthropic
from google import genai
from google.genai import types as genai_types

sys.path.insert(0, str(Path(__file__).resolve().parent))
from format_draft import build_user_prompt, format_draft

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"

CLAUDE_MODEL = "claude-sonnet-4-6"
GEMINI_MODEL = "gemini-2.5-flash"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


# ── Prompt loading ───────────────────────────────────────────────────────────

def load_system_prompt(task: str) -> str:
    path = PROMPTS_DIR / f"{task}.md"
    if not path.exists():
        raise FileNotFoundError(f"No prompt found for task '{task}' at {path}")
    return path.read_text(encoding="utf-8")


# ── Dataset loading ──────────────────────────────────────────────────────────

def load_snapshots(game: str, dataset_dir: Path) -> list[dict]:
    """Load all representative snapshots for a game from dataset/<game>/index.json."""
    game_dir = dataset_dir / game
    index_path = game_dir / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"No index.json found at {index_path}")

    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)

    snapshots = []
    for entry in index["entries"]:
        snapshot_path = game_dir / entry["file"]
        with open(snapshot_path, encoding="utf-8") as f:
            snapshots.append(json.load(f))

    return snapshots


# ── AI calls ─────────────────────────────────────────────────────────────────

def call_claude(system_prompt: str, user_prompt: str) -> list[dict]:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return parse_json_response(message.content[0].text)


def call_gemini(system_prompt: str, user_prompt: str) -> list[dict]:
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=4096,
            thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return parse_json_response(response.text)


def parse_json_response(text: str) -> list[dict]:
    """Extract JSON array from model response."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1)
    return json.loads(text.strip())


# ── Per-snapshot annotation ───────────────────────────────────────────────────

def annotate_snapshot(snapshot: dict, system_prompt: str) -> dict:
    """
    Annotate a single representative snapshot with both AI models.
    Reads trajectory and all_errors directly from the snapshot dict.
    """
    snapshot_id = snapshot["id"]
    trajectory = snapshot["trajectory"]

    # all_errors is stored as a flat list in the snapshot
    errors_by_category = {
        "system":    [e for e in snapshot.get("all_errors", []) if e["error_type"].startswith("system")],
        "strategy":  [e for e in snapshot.get("all_errors", []) if e["error_type"].startswith("strategy")],
        "operation": [e for e in snapshot.get("all_errors", []) if e["error_type"].startswith("operation")],
    }

    draft = format_draft(errors_by_category)
    user_prompt = build_user_prompt(trajectory, draft)

    result = {"id": snapshot_id}

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_claude = executor.submit(call_claude, system_prompt, user_prompt)
        future_gemini = executor.submit(call_gemini, system_prompt, user_prompt)

        for future, model_name in [(future_claude, "claude"), (future_gemini, "gemini")]:
            try:
                result[model_name] = future.result()
            except Exception as e:
                result[model_name] = None
                result[f"error_{model_name}"] = str(e)
                print(f"  [{snapshot_id}] {model_name} error: {e}")

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    system_prompt = load_system_prompt(args.task)
    dataset_dir = Path(args.dataset_dir)
    snapshots = load_snapshots(args.game, dataset_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.game}_ai_annotations.json"

    # Load existing results if retrying
    if args.retry_gemini and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            results = json.load(f)
        # Ensure list is long enough
        while len(results["snapshots"]) < len(snapshots):
            results["snapshots"].append(None)
        # Only retry snapshots where gemini failed
        retry_indices = [
            i for i, s in enumerate(results["snapshots"])
            if s is not None and s.get("gemini") is None
        ]
        print(f"Retrying Gemini for {len(retry_indices)} failed snapshots...")
    else:
        results = {
            "game": args.game,
            "task": args.task,
            "models": {"claude": CLAUDE_MODEL, "gemini": GEMINI_MODEL},
            "snapshots": [None] * len(snapshots),
        }
        retry_indices = None

    completed = 0
    lock = __import__("threading").Lock()

    def retry_gemini_and_save(i: int, snapshot: dict):
        nonlocal completed
        sid = snapshot["id"]
        try:
            gemini_result = call_gemini(system_prompt, build_user_prompt(
                snapshot["trajectory"],
                format_draft({
                    "system":    [e for e in snapshot.get("all_errors", []) if e["error_type"].startswith("system")],
                    "strategy":  [e for e in snapshot.get("all_errors", []) if e["error_type"].startswith("strategy")],
                    "operation": [e for e in snapshot.get("all_errors", []) if e["error_type"].startswith("operation")],
                })
            ))
        except Exception as e:
            gemini_result = None
            print(f"  [{sid}] gemini error: {e}")

        with lock:
            results["snapshots"][i]["gemini"] = gemini_result
            results["snapshots"][i].pop("error_gemini", None)
            completed += 1
            n = len(gemini_result or [])
            print(f"  [{completed}] {sid}: gemini={n}")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    def annotate_and_save(i: int, snapshot: dict):
        nonlocal completed
        sid = snapshot["id"]
        result = annotate_snapshot(snapshot, system_prompt)

        with lock:
            results["snapshots"][i] = result
            completed += 1
            n_claude = len(result.get("claude") or [])
            n_gemini = len(result.get("gemini") or [])
            print(f"  [{completed}/{len(snapshots)}] {sid}: claude={n_claude}, gemini={n_gemini}")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    if retry_indices is not None:
        snapshot_map = {s["id"]: s for s in snapshots}
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(retry_gemini_and_save, i, snapshots[i])
                for i in retry_indices
            ]
            for future in futures:
                future.result()
    else:
        print(f"Annotating {len(snapshots)} snapshots for '{args.game}' "
              f"(concurrency={args.workers})...")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(annotate_and_save, i, s)
                       for i, s in enumerate(snapshots)]
            for future in futures:
                future.result()

    print(f"\nDone. Results saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", required=True,
                        choices=["detective", "library", "zork1", "zork3", "balances", "temple"])
    parser.add_argument("--dataset_dir", default=str(DATASET_DIR),
                        help="Path to dataset directory (default: JTTL/dataset)")
    parser.add_argument("--output_dir", default="output",
                        help="Directory to save AI annotation results")
    parser.add_argument("--task", default="jericho",
                        help="Task type, selects prompt from prompts/<task>.md")
    parser.add_argument("--retry_gemini", action="store_true",
                        help="Retry only Gemini-failed snapshots from existing output")
    parser.add_argument("--workers", type=int, default=10,
                        help="Number of snapshots to annotate concurrently (default: 10)")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
