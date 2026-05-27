"""
annotation_ui.py  —  Human annotation interface for ROGUE failure annotations.

Usage:
    conda run -n Fraud python annotation_ui.py
    # then open http://localhost:5003

Saves results to: output/human/{game}_human.json
"""

import json
import re
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template, request, url_for

# ── Config ────────────────────────────────────────────────────────────────────

BASE = Path(__file__).parent
OUTPUT = BASE / "output"
HUMAN_DIR = OUTPUT / "human"
HUMAN_DIR.mkdir(exist_ok=True)

GAMES = ["detective", "library", "zork1", "zork3", "balances", "temple"]

IOU_THRESHOLD = 0.3

# Map model tag in episode ID → annotations filename suffix
MODEL_ANN = {
    "gpt41": "",           # {game}_annotations.json
    "qwen332b": "_qwen3",  # {game}_annotations_qwen3.json
}

app = Flask(__name__, template_folder="templates_human")


# ── Data helpers ──────────────────────────────────────────────────────────────

def iou(a, b):
    a0, a1 = a
    b0, b1 = b
    overlap = max(0, min(a1, b1) - max(a0, b0) + 1)
    union = (a1 - a0 + 1) + (b1 - b0 + 1) - overlap
    return overlap / union if union > 0 else 0.0


def get_run_dir(game: str, model_tag: str) -> Path | None:
    suffix = MODEL_ANN.get(model_tag, "")
    ann_path = OUTPUT / f"{game}_annotations{suffix}.json"
    if not ann_path.exists():
        return None
    data = json.loads(ann_path.read_text())
    return Path(data["run_dir"])


def find_log_file(run_dir: Path, ep_num: int) -> Path | None:
    pattern = f"episode_{ep_num:03d}_*.txt"
    matches = list(run_dir.glob(pattern))
    return matches[0] if matches else None


def parse_trajectory(log_path: Path) -> list[dict]:
    """Parse episode log into a list of steps with obs/action/inv/score."""
    steps = []
    current = {}
    section = None

    for line in log_path.read_text(errors="replace").splitlines():
        if line.startswith("[STEP]"):
            if current:
                steps.append(current)
            current = {"step": int(line.split()[1]), "obs": "", "action": "", "inv": "", "score": ""}
            section = None
        elif line.startswith("[OBS]"):
            section = "obs"
            current["obs"] = line[5:].strip()
        elif line.startswith("[INV]"):
            section = "inv"
            current["inv"] = line[5:].strip()
        elif line.startswith("[RAW_LLM_OUTPUT]"):
            section = "llm"
            current["llm_raw"] = line[16:].strip()
        elif line.startswith("ACTION:"):
            current["action"] = line[7:].strip()
            section = None
        elif line.startswith("[Your score"):
            m = re.search(r"(\d+)", line)
            if m:
                current["score"] = m.group(1)
        elif line.startswith("----------") or line.startswith("=========="):
            section = None
        elif section == "obs":
            current["obs"] += "\n" + line
        elif section == "inv":
            current["inv"] += "\n" + line

    if current:
        steps.append(current)
    return steps


def build_episode_groups(ep: dict) -> list[dict]:
    """
    Build the list of annotation items for one episode.
    Each item has: group, iou, claude, gemini
    """
    fa_list = ep.get("claude") or []
    fb_list = ep.get("gemini") or []
    items = []

    # Greedy IoU matching at threshold
    used_b = set()
    matched_pairs = []
    for fa in fa_list:
        wa = fa.get("where", [0, 0])
        bi, bs = -1, 0.0
        for j, fb in enumerate(fb_list):
            if j in used_b:
                continue
            s = iou(wa, fb.get("where", [0, 0]))
            if s > bs:
                bs, bi = s, j
        if bs >= IOU_THRESHOLD and bi >= 0:
            matched_pairs.append((fa, fb_list[bi], bs))
            used_b.add(bi)

    matched_a_ids = {id(m[0]) for m in matched_pairs}
    unm_a = [fa for fa in fa_list if id(fa) not in matched_a_ids]
    unm_b = [fb for j, fb in enumerate(fb_list) if j not in used_b]

    # Group 1: matched pairs — only where ≥1 is core
    for fa, fb, score in matched_pairs:
        if fa.get("tier") == "core" or fb.get("tier") == "core":
            items.append({"group": "matched", "iou": round(score, 3), "claude": fa, "gemini": fb})

    # Group 2: low-confidence pairs (0 < IoU < 0.3, ≥1 core)
    used_low_b = set()
    for i, fa in enumerate(unm_a):
        for j, fb in enumerate(unm_b):
            if j in used_low_b:
                continue
            s = iou(fa.get("where", [0, 0]), fb.get("where", [0, 0]))
            if 0 < s < IOU_THRESHOLD:
                if fa.get("tier") == "core" or fb.get("tier") == "core":
                    items.append({"group": "low_conf", "iou": round(s, 3), "claude": fa, "gemini": fb})
                    used_low_b.add(j)
                    break

    # Group 3: Claude-only core
    low_used_a = {id(item["claude"]) for item in items if item["group"] == "low_conf" and item.get("claude")}
    for fa in unm_a:
        if id(fa) not in low_used_a and fa.get("tier") == "core":
            items.append({"group": "claude_only", "iou": None, "claude": fa, "gemini": None})

    # Group 4: Gemini-only core
    low_used_b_ids = {id(item["gemini"]) for item in items if item["group"] == "low_conf" and item.get("gemini")}
    for fb in unm_b:
        if id(fb) not in low_used_b_ids and fb.get("tier") == "core":
            items.append({"group": "gemini_only", "iou": None, "claude": None, "gemini": fb})

    # Fallback: if no items after core filtering, show ALL failures regardless of tier
    # (both annotators may have mislabeled everything as marginal)
    if not items:
        for fa, fb, score in matched_pairs:
            items.append({"group": "matched", "iou": round(score, 3), "claude": fa, "gemini": fb, "fallback": True})
        used_low_b2 = set()
        for i, fa in enumerate(unm_a):
            for j, fb in enumerate(unm_b):
                if j in used_low_b2:
                    continue
                s = iou(fa.get("where", [0, 0]), fb.get("where", [0, 0]))
                if 0 < s < IOU_THRESHOLD:
                    items.append({"group": "low_conf", "iou": round(s, 3), "claude": fa, "gemini": fb, "fallback": True})
                    used_low_b2.add(j)
                    break
        low_used_a2 = {id(item["claude"]) for item in items if item.get("fallback") and item.get("claude")}
        for fa in unm_a:
            if id(fa) not in low_used_a2:
                items.append({"group": "claude_only", "iou": None, "claude": fa, "gemini": None, "fallback": True})
        low_used_b2_ids = {id(item["gemini"]) for item in items if item.get("fallback") and item.get("gemini")}
        for fb in unm_b:
            if id(fb) not in low_used_b2_ids:
                items.append({"group": "gemini_only", "iou": None, "claude": None, "gemini": fb, "fallback": True})

    return items


def load_ai_data(game: str) -> dict:
    """Load ai_annotations.json for a game."""
    path = OUTPUT / f"{game}_ai_annotations.json"
    return json.loads(path.read_text())


def load_human(game: str) -> dict:
    path = HUMAN_DIR / f"{game}_human.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"game": game, "episodes": {}}


def save_human(game: str, data: dict):
    data["updated_at"] = datetime.now().isoformat()
    path = HUMAN_DIR / f"{game}_human.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    games_info = []
    for game in GAMES:
        ai = load_ai_data(game)
        human = load_human(game)
        total = len(ai["snapshots"])
        done = sum(1 for ep_id, ep_data in human["episodes"].items() if ep_data.get("completed"))
        games_info.append({"name": game, "total": total, "done": done})
    return render_template("index.html", games=games_info)


@app.route("/game/<game>")
def game_view(game):
    ai = load_ai_data(game)
    human = load_human(game)
    episodes = []
    for ep in ai["snapshots"]:
        ep_id = ep["id"]
        items = build_episode_groups(ep)
        n_items = len(items)
        ep_human = human["episodes"].get(ep_id, {})
        n_done = sum(
            1 for item_data in ep_human.get("items", {}).values()
            if item_data.get("keep") is not None
        )
        episodes.append({
            "id": ep_id,
            "n_items": n_items,
            "n_done": n_done,
            "completed": ep_human.get("completed", False),
        })
    return render_template("game.html", game=game, episodes=episodes)


@app.route("/game/<game>/episode/<ep_id>")
def episode_view(game, ep_id):
    ai = load_ai_data(game)
    human = load_human(game)

    # Find this episode
    ep = next((e for e in ai["snapshots"] if e["id"] == ep_id), None)
    if ep is None:
        return f"Episode {ep_id} not found", 404

    # Get episode list for prev/next nav
    ep_ids = [e["id"] for e in ai["snapshots"]]
    ep_idx = ep_ids.index(ep_id)
    prev_id = ep_ids[ep_idx - 1] if ep_idx > 0 else None
    next_id = ep_ids[ep_idx + 1] if ep_idx < len(ep_ids) - 1 else None

    # Build items
    items = build_episode_groups(ep)

    # Load trajectory
    model_tag = re.search(r"_(gpt41|qwen332b)_", ep_id)
    model_tag = model_tag.group(1) if model_tag else "gpt41"
    ep_num = int(ep_id.rsplit("_", 1)[-1])
    run_dir = get_run_dir(game, model_tag)
    trajectory = []
    if run_dir:
        log_file = find_log_file(run_dir, ep_num)
        if log_file:
            trajectory = parse_trajectory(log_file)

    # Load existing human annotations for this episode
    ep_human = human["episodes"].get(ep_id, {"items": {}, "completed": False})
    # Merge saved state into items
    for i, item in enumerate(items):
        key = str(i)
        saved = ep_human.get("items", {}).get(key, {})
        item["human"] = saved

    return render_template(
        "episode.html",
        game=game,
        ep_id=ep_id,
        ep_idx=ep_idx,
        total_eps=len(ep_ids),
        prev_id=prev_id,
        next_id=next_id,
        items=items,
        trajectory=trajectory,
        completed=ep_human.get("completed", False),
    )


@app.route("/api/save_item", methods=["POST"])
def save_item():
    data = request.get_json()
    game = data["game"]
    ep_id = data["ep_id"]
    item_idx = str(data["item_idx"])
    human_data = data["human"]  # {keep, diagnosis_choice, diagnosis_custom}

    human = load_human(game)
    if ep_id not in human["episodes"]:
        human["episodes"][ep_id] = {"items": {}, "completed": False}
    human["episodes"][ep_id]["items"][item_idx] = human_data
    save_human(game, human)
    return jsonify({"ok": True})


@app.route("/api/mark_complete", methods=["POST"])
def mark_complete():
    data = request.get_json()
    game = data["game"]
    ep_id = data["ep_id"]
    human = load_human(game)
    if ep_id not in human["episodes"]:
        human["episodes"][ep_id] = {"items": {}, "completed": False}
    human["episodes"][ep_id]["completed"] = True
    save_human(game, human)
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004, debug=True)
