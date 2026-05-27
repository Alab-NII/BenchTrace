"""
format_draft.py

Converts rule-based classify_errors() output into the draft annotation format
expected by the AI annotator prompt, and formats episode trajectories into
structured step lists for the snapshot.
"""

import json
import re
from pathlib import Path


# ── Snapshot formatting ──────────────────────────────────────────────────────

def format_snapshot(trajectory: list[dict]) -> str:
    """
    Format a parsed trajectory into a compact JSON string for the AI prompt.
    Each step: {step, obs, action, reward, cum_score}
    """
    return json.dumps(trajectory, ensure_ascii=False, indent=2)


# ── Draft annotation formatting ──────────────────────────────────────────────

def _infer_where(error: dict) -> list[int]:
    """
    Infer the step range [first, last] from a rule-based error dict.
    Uses evidence fields where available, falls back to [step, step].
    """
    evidence = error.get("evidence", {})
    step = error.get("step", 0)

    # strategy/loop has segment_steps: [start, end]
    seg = evidence.get("segment_steps")
    if seg and len(seg) == 2:
        return [seg[0], seg[1]]

    return [step, step]


def _make_failure_id(error_type: str, step: int) -> str:
    subtype = error_type.split("/")[-1]
    return f"{subtype}_step{step}"


def format_draft(errors_by_category: dict) -> list[dict]:
    """
    Convert the output of classify_errors() into a draft annotation list
    for the AI annotator.

    Input: the 'errors' field from annotate_episodes output, structured as:
        {"system": [...], "strategy": [...], "operation": [...], "summary": {...}}

    Output: list of draft failure dicts with fields:
        failure_id, type, where, why
    (tier and location are left for the AI to fill in)
    """
    draft = []
    for category in ("system", "strategy", "operation"):
        for error in errors_by_category.get(category, []):
            draft.append({
                "failure_id": _make_failure_id(error["error_type"], error["step"]),
                "type": error["error_type"],
                "tier": "",
                "location": "",
                "why": error["description"],
                "where": _infer_where(error),
            })

    # Deduplicate by (type, where) — rule-based may fire multiple times for same step
    seen = set()
    deduped = []
    for item in draft:
        key = (item["type"], tuple(item["where"]))
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    # Sort by first step for readability
    deduped.sort(key=lambda x: x["where"][0])
    return deduped


# ── User prompt assembly ─────────────────────────────────────────────────────

def build_user_prompt(trajectory: list[dict], draft: list[dict]) -> str:
    snapshot_str = format_snapshot(trajectory)
    draft_str = json.dumps(draft, ensure_ascii=False, indent=2)
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
{draft_str}
</draft_annotation>

Return the revised annotation as a JSON array in the format specified above."""
