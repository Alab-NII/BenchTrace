"""
ai_annotate.py

AI-based annotation of GroupTravelPlanning episodes using Claude + Gemini.
Reads episode snapshots from GroupTravelPlanning/output/ and annotates failures
using the group_travel_planning.md prompt.

Usage:
    conda run -n Fraud python ai_annotate.py \
        [--snapshots_dir ../output] \
        [--output_dir output] \
        [--framework reflexion] \
        [--model gpt-4.1] \
        [--workers 10] \
        [--retry_claude] \
        [--retry_gemini] \
        [--claude_only]
"""

import argparse
import json
import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anthropic
from google import genai
from google.genai import types as genai_types

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
SNAPSHOTS_DIR = Path(__file__).resolve().parent.parent / "output"

CLAUDE_MODEL = "claude-sonnet-4-6"
GEMINI_MODEL = "gemini-2.5-flash"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


# ── Prompt loading ────────────────────────────────────────────────────────────

def load_system_prompt() -> str:
    path = PROMPTS_DIR / "group_travel_planning.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found at {path}")
    return path.read_text(encoding="utf-8")


# ── Snapshot loading ──────────────────────────────────────────────────────────

def load_snapshots(
    snapshots_dir: Path,
    framework: str | None,
    model: str | None,
) -> list[dict]:
    snapshots = []
    for path in sorted(snapshots_dir.rglob("*.json")):
        if path.name == "summary.json":
            continue
        with open(path, encoding="utf-8") as f:
            snap = json.load(f)
        if snap.get("won", True):
            continue
        if framework and snap.get("framework") != framework:
            continue
        if model and snap.get("model") != model:
            continue
        snapshots.append(snap)
    return snapshots


# ── User prompt ───────────────────────────────────────────────────────────────

def build_user_prompt(snapshot: dict) -> str:
    snapshot_for_prompt = {
        "task_id": snapshot.get("task_id"),
        "task": snapshot.get("task"),
        "n_subtasks": snapshot.get("n_subtasks"),
        "final_progress": snapshot.get("progress"),
        "won": snapshot.get("won"),
        "trajectory": snapshot.get("trajectory", []),
    }
    snapshot_str = json.dumps(snapshot_for_prompt, ensure_ascii=False, indent=2)
    return f"""## Episode Snapshot

<snapshot>
{snapshot_str}
</snapshot>

## Rule-Based Annotation (Draft)

The following failures were automatically detected by a rule-based classifier. Each entry contains `type`, `where`, and a template-generated `why`. Please revise this list:
- Fill in `tier` (core/marginal) and `location` for every entry
- Correct any inaccurate `type`, `where`, or `why` fields
- Remove any false positives
- Add any failures that were missed

<draft_annotation>
[]
</draft_annotation>

Return the revised annotation as a JSON array in the format specified above."""


# ── AI calls ──────────────────────────────────────────────────────────────────

_gemini_sem = threading.Semaphore(1)


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
    with _gemini_sem:
        client = genai.Client(api_key=GEMINI_API_KEY)
        for attempt in range(5):
            try:
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
            except Exception as e:
                if ("429" in str(e) or "503" in str(e)) and attempt < 4:
                    wait = 30 * (attempt + 1)
                    print(f"  [gemini] {str(e)[:20]} retrying in {wait}s (attempt {attempt+1}/5)...")
                    time.sleep(wait)
                else:
                    raise


def parse_json_response(text: str) -> list[dict]:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1)
    return json.loads(text.strip())


# ── Per-snapshot annotation ───────────────────────────────────────────────────

def annotate_snapshot(snapshot: dict, system_prompt: str, claude_only: bool = False) -> dict:
    sid = snapshot["id"]
    user_prompt = build_user_prompt(snapshot)
    result = {"id": sid}

    models = [("claude", call_claude)]
    if not claude_only:
        models.append(("gemini", call_gemini))

    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = {name: executor.submit(fn, system_prompt, user_prompt) for name, fn in models}
        for model_name, future in futures.items():
            try:
                result[model_name] = future.result()
            except Exception as e:
                result[model_name] = None
                result[f"error_{model_name}"] = str(e)
                print(f"  [{sid}] {model_name} error: {e}")

    return result


# ── Atomic write ─────────────────────────────────────────────────────────────

def _atomic_write(path: Path, data: dict):
    dir_ = path.parent
    with tempfile.NamedTemporaryFile("w", dir=dir_, delete=False,
                                    suffix=".tmp", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        tmp = f.name
    os.replace(tmp, path)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    system_prompt = load_system_prompt()
    snapshots_dir = Path(args.snapshots_dir)
    snapshots = load_snapshots(
        snapshots_dir,
        framework=args.framework,
        model=args.model,
    )
    print(f"Loaded {len(snapshots)} failed snapshots to annotate.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tag_parts = [x for x in [args.framework, args.model] if x]
    tag = "_".join(tag_parts) if tag_parts else "all"
    output_path = output_dir / f"group_travel_planning_{tag}_ai_annotations.json"

    retry_model = None
    if (args.retry_gemini or args.retry_claude) and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            results = json.load(f)
        while len(results["snapshots"]) < len(snapshots):
            results["snapshots"].append(None)
        retry_model = "gemini" if args.retry_gemini else "claude"
        retry_indices = [
            i for i, s in enumerate(results["snapshots"])
            if s is not None and s.get(retry_model) is None
        ]
        print(f"Retrying {retry_model} for {len(retry_indices)} failed snapshots...")
    else:
        results = {
            "task": "group_travel_planning",
            "models": {"claude": CLAUDE_MODEL, "gemini": GEMINI_MODEL},
            "snapshots": [None] * len(snapshots),
        }
        retry_indices = None

    completed = 0
    lock = threading.Lock()

    def annotate_and_save(i: int, snapshot: dict):
        nonlocal completed
        sid = snapshot["id"]
        result = annotate_snapshot(snapshot, system_prompt, claude_only=args.claude_only)

        with lock:
            results["snapshots"][i] = result
            completed += 1
            n_claude = len(result.get("claude") or [])
            n_gemini = len(result.get("gemini") or [])
            suffix = f"claude={n_claude}" if args.claude_only else f"claude={n_claude}, gemini={n_gemini}"
            print(f"  [{completed}/{len(snapshots)}] {sid}: {suffix}")
            _atomic_write(output_path, results)

    def retry_model_and_save(i: int, snapshot: dict, model_name: str):
        nonlocal completed
        sid = snapshot["id"]
        caller = call_claude if model_name == "claude" else call_gemini
        try:
            model_result = caller(system_prompt, build_user_prompt(snapshot))
        except Exception as e:
            model_result = None
            print(f"  [{sid}] {model_name} error: {e}")

        with lock:
            if output_path.exists():
                with open(output_path, encoding="utf-8") as f:
                    on_disk = json.load(f)
                results["snapshots"] = on_disk["snapshots"]
            results["snapshots"][i][model_name] = model_result
            results["snapshots"][i].pop(f"error_{model_name}", None)
            completed += 1
            n = len(model_result or [])
            print(f"  [{completed}] {sid}: {model_name}={n}")
            _atomic_write(output_path, results)

    if retry_model is not None:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(retry_model_and_save, i, snapshots[i], retry_model)
                for i in retry_indices
            ]
            for future in futures:
                future.result()
    else:
        print(f"Annotating {len(snapshots)} snapshots (concurrency={args.workers})...")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(annotate_and_save, i, s)
                for i, s in enumerate(snapshots)
            ]
            for future in futures:
                future.result()

    print(f"\nDone. Results saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots_dir", default=str(SNAPSHOTS_DIR))
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--framework", default=None,
                        help="Filter by framework, e.g. reflexion (default: all)")
    parser.add_argument("--model", default=None,
                        help="Filter by model, e.g. gpt-4.1 (default: all)")
    parser.add_argument("--retry_gemini", action="store_true")
    parser.add_argument("--retry_claude", action="store_true")
    parser.add_argument("--claude_only", action="store_true",
                        help="Skip Gemini, annotate with Claude only")
    parser.add_argument("--workers", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
