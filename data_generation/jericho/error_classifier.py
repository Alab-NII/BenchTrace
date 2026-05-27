"""
error_classifier.py

Rule-based classifier for LLM agent errors in Jericho text adventure games.
Classifies in-episode errors into three categories:

  1. System errors    — output format violations
  2. Strategy errors  — route inefficiency, wrong destination, loop/cycle
  3. Operation errors — perception error, decision error, feedback blindness

Usage (standalone):
    from error_classifier import classify_errors
    errors = classify_errors(trajectory, milestones)

Each trajectory step dict:
    {step, obs, action, reward, cum_score}

Each milestone dict (from annotate_episodes.extract_score_events):
    {walkthrough_step, action, score_before, score_after, delta, observation_after}
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


# ── Room name extraction ────────────────────────────────────────────────────

# Jericho games render room headers as  "** Room Name **"  or  "<<Room Name>>"
_ROOM_PATTERNS = [
    re.compile(r"\*\*\s*(.+?)\s*\*\*"),       # ** Room Name **
    re.compile(r"<<\s*(.+?)\s*>>"),            # <<Room Name>>
    re.compile(r"^([A-Z][A-Za-z '\-]+)$", re.M),  # Title-cased first line fallback
]

def extract_room_name(obs: str) -> Optional[str]:
    """Return the room name from an observation string, or None."""
    for pat in _ROOM_PATTERNS:
        m = pat.search(obs)
        if m:
            name = m.group(1).strip()
            if 3 < len(name) < 60:
                return name
    return None


# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class ErrorEvent:
    error_type: str          # e.g. "system/format", "strategy/loop"
    step: int                # trajectory step where error is detected
    description: str
    evidence: dict = field(default_factory=dict)


# ── 1. System errors ────────────────────────────────────────────────────────

# Actions that look like raw LLM output artefacts
_NON_ACTION_PATTERNS = [
    re.compile(r"^\s*$"),                           # empty
    re.compile(r"(?i)^(sorry|i (can't|cannot)|as an ai)", re.I),
    re.compile(r"\n"),                              # multi-line (should be single command)
    re.compile(r"^[^a-zA-Z]"),                     # starts with non-letter (JSON/code leak)
    re.compile(r".{120,}"),                         # suspiciously long
]

def detect_system_errors(trajectory: list[dict]) -> list[ErrorEvent]:
    """Detect output format errors in agent actions."""
    errors = []
    for step in trajectory:
        action = step.get("action", "")
        for pat in _NON_ACTION_PATTERNS:
            if pat.search(action):
                errors.append(ErrorEvent(
                    error_type="system/format",
                    step=step["step"],
                    description=f"Malformed action: {repr(action[:80])}",
                    evidence={"action": action},
                ))
                break
    return errors


# ── 2. Strategy errors ───────────────────────────────────────────────────────

def detect_loops(
    trajectory: list[dict],
    min_repeats: int = 2,
) -> list[ErrorEvent]:
    """
    Detect loop segments: contiguous stretches with no score gain where at least
    one (room, action) pair repeats >= min_repeats times.

    The trajectory is first split at every score-gain event into no-progress
    segments. Each segment is checked independently, yielding at most ONE
    ErrorEvent per segment — eliminating sliding-window double-counting.
    """
    if not trajectory:
        return []

    errors = []
    rooms = [extract_room_name(s["obs"]) or "?" for s in trajectory]

    # Split trajectory into no-progress segments at score-gain boundaries
    segments: list[list[int]] = []  # each entry = list of trajectory indices
    current: list[int] = [0]
    for idx in range(1, len(trajectory)):
        if trajectory[idx]["cum_score"] > trajectory[idx - 1]["cum_score"]:
            segments.append(current)
            current = [idx]
        else:
            current.append(idx)
    segments.append(current)

    for seg_indices in segments:
        if len(seg_indices) < min_repeats:
            continue

        pairs = [(rooms[idx], trajectory[idx]["action"].lower().strip())
                 for idx in seg_indices]
        counts = Counter(pairs)
        flagged = [(pair, cnt) for pair, cnt in counts.items() if cnt >= min_repeats]

        if not flagged:
            continue

        worst = max(flagged, key=lambda x: x[1])
        start_step = trajectory[seg_indices[0]]["step"]
        end_step   = trajectory[seg_indices[-1]]["step"]
        errors.append(ErrorEvent(
            error_type="strategy/loop",
            step=start_step,
            description=(
                f"Loop segment steps {start_step}–{end_step} ({len(seg_indices)} steps): "
                f"'{worst[0][1]}' in room '{worst[0][0]}' repeated {worst[1]}x"
            ),
            evidence={
                "room": worst[0][0],
                "repeated_action": worst[0][1],
                "repeat_count": worst[1],
                "segment_length": len(seg_indices),
                "segment_steps": [start_step, end_step],
                "all_repeated_pairs": {
                    f"{p[0]}|{p[1]}": c for p, c in flagged
                },
            },
        ))

    return errors


def detect_route_inefficiency(
    trajectory: list[dict],
    milestones: list[dict],
    threshold_ratio: float = 2.0,
) -> list[ErrorEvent]:
    """
    Detect when the agent takes far more steps than the walkthrough to reach a
    score milestone.

    threshold_ratio: flag if agent_steps > threshold_ratio * walkthrough_steps
    between two consecutive milestones.
    """
    if len(milestones) < 2:
        return []

    errors = []
    total_steps = trajectory[-1]["step"] if trajectory else 0

    # Map score → first trajectory step index that reached it
    score_to_step: dict[float, int] = {}
    for step in trajectory:
        s = step["cum_score"]
        if s not in score_to_step:
            score_to_step[s] = step["step"]

    for i in range(1, len(milestones)):
        prev_m = milestones[i - 1]
        curr_m = milestones[i]

        wt_steps = curr_m["walkthrough_step"] - prev_m["walkthrough_step"]
        if wt_steps <= 0:
            continue

        agent_reach_prev = score_to_step.get(prev_m["score_after"])
        if agent_reach_prev is None:
            continue  # Agent never even reached the first milestone — not measurable

        agent_reach_curr = score_to_step.get(curr_m["score_after"])

        if agent_reach_curr is not None:
            # Agent reached both milestones
            agent_steps = agent_reach_curr - agent_reach_prev
            reached_second = True
        else:
            # Agent reached the first but never the second:
            # count all remaining steps as spent failing to reach curr_m
            agent_steps = total_steps - agent_reach_prev
            reached_second = False

        if agent_steps > threshold_ratio * wt_steps:
            errors.append(ErrorEvent(
                error_type="strategy/route_inefficiency",
                step=agent_reach_prev,
                description=(
                    f"Agent used {agent_steps} steps after score {prev_m['score_after']} "
                    f"{'to reach' if reached_second else 'without reaching'} "
                    f"score {curr_m['score_after']} "
                    f"(walkthrough: {wt_steps} steps, ratio {agent_steps/wt_steps:.1f}x)"
                ),
                evidence={
                    "from_score": prev_m["score_after"],
                    "to_score": curr_m["score_after"],
                    "reached_second_milestone": reached_second,
                    "agent_steps": agent_steps,
                    "walkthrough_steps": wt_steps,
                    "ratio": round(agent_steps / wt_steps, 2),
                },
            ))

    return errors




# ── 3. Operation errors ──────────────────────────────────────────────────────

# Patterns that suggest the game gave negative / informative feedback
_NEGATIVE_FEEDBACK_PATTERNS = [
    re.compile(r"(?i)(you can'?t|that's not|doesn'?t work|nothing happen|no effect|i don'?t (see|understand)|what\?)"),
    re.compile(r"(?i)(already|can'?t go that way|not here|not possible)"),
]

def _has_negative_feedback(obs: str) -> bool:
    return any(p.search(obs) for p in _NEGATIVE_FEEDBACK_PATTERNS)


def detect_feedback_blindness(
    trajectory: list[dict],
    repeat_window: int = None,
) -> list[ErrorEvent]:
    """
    Detect when the agent repeats an action after receiving explicit negative
    feedback for it within the last `repeat_window` steps.
    """
    errors = []
    n = len(trajectory)
    window = repeat_window if repeat_window is not None else n  # None = global

    for i in range(1, n):
        curr_action = trajectory[i]["action"].lower().strip()
        # Look back within window
        for j in range(max(0, i - window), i):
            if trajectory[j]["action"].lower().strip() != curr_action:
                continue
            # Check if there was negative feedback in the observation that
            # followed step j
            if j + 1 < n and _has_negative_feedback(trajectory[j + 1]["obs"]):
                errors.append(ErrorEvent(
                    error_type="operation/feedback_blindness",
                    step=trajectory[i]["step"],
                    description=(
                        f"Action '{curr_action}' repeated at step {trajectory[i]['step']} "
                        f"despite negative feedback at step {trajectory[j+1]['step']}"
                    ),
                    evidence={
                        "repeated_action": curr_action,
                        "first_attempt_step": trajectory[j]["step"],
                        "negative_feedback_step": trajectory[j + 1]["step"],
                        "negative_feedback_obs": trajectory[j + 1]["obs"][:200],
                        "repeat_step": trajectory[i]["step"],
                    },
                ))
                break

    return errors


def detect_perception_errors(
    trajectory: list[dict],
    milestones: list[dict],
) -> list[ErrorEvent]:
    """
    Detect when the agent is in the correct room for a milestone but fails to
    take the required action, suggesting it did not perceive the relevant object
    or affordance.

    Heuristic: the agent is in the milestone's target room (from observation_after)
    but never issues the required action while in that room, yet also fails to
    reach the required score.
    """
    errors = []
    final_score = trajectory[-1]["cum_score"] if trajectory else 0

    for m in milestones:
        if m["score_after"] <= final_score:
            continue  # Agent did reach this milestone

        target_room = extract_room_name(m["observation_after"])
        if target_room is None:
            continue

        correct_action = m["action"].lower().strip()
        # Canonical direction for correct_action (None if not a direction)
        correct_dir = _action_to_direction(correct_action)

        # Find steps where agent was in the target room
        in_room_steps = [
            s for s in trajectory
            if (extract_room_name(s["obs"]) or "").lower() == target_room.lower()
        ]

        if not in_room_steps:
            continue  # wrong-destination error, not perception

        # Agent was in the right room — did it try the correct action?
        # Normalize direction aliases: 'north' == 'n', 'west' == 'w', etc.
        def _action_matches(action: str) -> bool:
            a = action.lower().strip()
            if a == correct_action:
                return True
            if correct_dir and _action_to_direction(a) == correct_dir:
                return True
            return False

        tried = any(_action_matches(s["action"]) for s in in_room_steps)

        if not tried:
            # step = last visit to the room (final missed opportunity)
            errors.append(ErrorEvent(
                error_type="operation/perception_error",
                step=in_room_steps[-1]["step"],
                description=(
                    f"Agent entered room '{target_room}' but never tried "
                    f"'{correct_action}' (needed for score {m['score_after']})"
                ),
                evidence={
                    "required_room": target_room,
                    "correct_action": correct_action,
                    "last_visit_step": in_room_steps[-1]["step"],
                    "agent_actions_in_room": [s["action"] for s in in_room_steps],
                    "required_score": m["score_after"],
                },
            ))

    return errors


def detect_decision_errors(
    trajectory: list[dict],
    milestones: list[dict],
) -> list[ErrorEvent]:
    """
    Detect when the agent tries the correct action in the correct room but at
    the wrong time (preconditions not met), or tries a semantically close but
    incorrect variant.

    Heuristic: agent issued an action containing the key verb of the milestone
    action while in the milestone room, but the action didn't score points.
    """
    errors = []
    final_score = trajectory[-1]["cum_score"] if trajectory else 0

    for m in milestones:
        if m["score_after"] <= final_score:
            continue

        target_room = extract_room_name(m["observation_after"])
        if target_room is None:
            continue

        correct_action = m["action"].lower().strip()
        correct_dir    = _action_to_direction(correct_action)
        # For direction actions the "verb" concept doesn't apply — skip verb matching
        is_direction_action = correct_dir is not None
        correct_verb = correct_action.split()[0] if correct_action and not is_direction_action else ""

        in_room_steps = [
            s for s in trajectory
            if (extract_room_name(s["obs"]) or "").lower() == target_room.lower()
        ]

        if not in_room_steps:
            continue

        def _action_matches_correct(action: str) -> bool:
            a = action.lower().strip()
            if a == correct_action:
                return True
            if correct_dir and _action_to_direction(a) == correct_dir:
                return True
            return False

        # Agent tried correct action (exact or direction-equivalent) but scored 0
        # → precondition failure (decision error)
        wrong_timing_steps = [
            s for s in in_room_steps
            if _action_matches_correct(s["action"]) and s["reward"] == 0
        ]
        if wrong_timing_steps:
            errors.append(ErrorEvent(
                error_type="operation/decision_error",
                step=wrong_timing_steps[0]["step"],
                description=(
                    f"Agent tried correct action '{correct_action}' in room "
                    f"'{target_room}' but scored 0 — precondition likely unmet"
                ),
                evidence={
                    "correct_action": correct_action,
                    "room": target_room,
                    "steps": [s["step"] for s in wrong_timing_steps],
                    "required_score": m["score_after"],
                },
            ))
            continue

        # Agent used same verb but different object (wrong variant)
        # Only applies to non-direction actions (e.g. "take X" vs "take Y")
        if correct_verb:
            wrong_variant_steps = [
                s for s in in_room_steps
                if s["action"].lower().startswith(correct_verb)
                and not _action_matches_correct(s["action"])
                and s["reward"] == 0
            ]
            if wrong_variant_steps:
                errors.append(ErrorEvent(
                    error_type="operation/decision_error",
                    step=wrong_variant_steps[0]["step"],
                    description=(
                        f"Agent used verb '{correct_verb}' in room '{target_room}' "
                        f"but with wrong object (needed: '{correct_action}')"
                    ),
                    evidence={
                        "correct_action": correct_action,
                        "agent_variants": [s["action"] for s in wrong_variant_steps[:5]],
                        "room": target_room,
                        "required_score": m["score_after"],
                    },
                ))

    return errors


# ── Exit / direction parsing ─────────────────────────────────────────────────

# Canonical direction → all surface forms
_DIR_CANON: dict[str, set[str]] = {
    "north":     {"north", "n"},
    "south":     {"south", "s"},
    "east":      {"east",  "e"},
    "west":      {"west",  "w"},
    "up":        {"up",    "u"},
    "down":      {"down",  "d"},
    "northeast": {"northeast", "ne"},
    "northwest": {"northwest", "nw"},
    "southeast": {"southeast", "se"},
    "southwest": {"southwest", "sw"},
}
_SURFACE_TO_CANON: dict[str, str] = {
    s: canon for canon, surfaces in _DIR_CANON.items() for s in surfaces
}
_DIR_WORDS_RE = (
    r"(?:north(?:east|west)?|south(?:east|west)?|east|west|up|down|ne|nw|se|sw)"
)

# Patterns that introduce exit directions in Jericho observations
_EXIT_LIST_PATTERNS = [
    # Structured: "Obvious exits: North, East"
    re.compile(rf"(?i)obvious exits?[:\s]+(.+?)(?:\n|\.|\Z)"),
    re.compile(rf"(?i)you can (?:also )?go[:\s]+(.+?)(?:\n|\.|\Z)"),
    re.compile(rf"(?i)\bexits?[:\s]+([a-zA-Z ,/]+?)(?:\n|\.|\Z)"),
    # Narrative instruction: "go north or west", "go east"
    re.compile(
        rf"(?i)\bgo\s+((?:{_DIR_WORDS_RE}(?:[,\s]+(?:or\s+)?)?)+)",
    ),
    # Physical feature pointing to a direction: "door/passage/path to the north"
    re.compile(
        rf"(?i)(?:door|passage|path|corridor|tunnel|opening|exit|stairway|stairs?)"
        rf"\s+(?:to\s+the\s+|leading\s+(?:to\s+the\s+)?)({_DIR_WORDS_RE})\b"
    ),
]

# Navigation verb prefixes to strip when recognising direction actions
_NAV_PREFIX = re.compile(r"^(?:go|walk|move|run|travel|head|proceed)\s+")


def parse_exits(obs: str) -> set[str]:
    """Return canonical direction names mentioned as available exits in obs."""
    exits: set[str] = set()
    for pat in _EXIT_LIST_PATTERNS:
        for m in pat.finditer(obs):
            text = m.group(1).lower()
            for tok in re.split(r"[,\s/]+", text):
                tok = tok.strip(".,;")
                if tok in _SURFACE_TO_CANON:
                    exits.add(_SURFACE_TO_CANON[tok])
    return exits


def _action_to_direction(action: str) -> Optional[str]:
    """Return canonical direction if action is (or starts with) a direction word."""
    a = action.lower().strip()
    if a in _SURFACE_TO_CANON:
        return _SURFACE_TO_CANON[a]
    # Strip navigation prefix: "go north" → "north"
    stripped = _NAV_PREFIX.sub("", a).strip()
    return _SURFACE_TO_CANON.get(stripped)


def detect_unexplored_destination(
    trajectory: list[dict],
    min_visits: int = 2,
) -> list[ErrorEvent]:
    """
    Detect when the agent was explicitly prompted about a direction or destination
    (via exit listings in observations) but never explored it.

    Only fires when the observation explicitly hints at an exit — if no hint was
    given, the unexplored direction is not counted.

    For each room visited >= min_visits times:
      - collect all exit directions mentioned in observations
      - collect all directions the agent actually tried from that room
      - flag directions that were hinted but never tried
    """
    # room -> {"exits": set, "tried": set, "visits": int,
    #          "dir_first_hint": {direction: step}}
    room_data: dict[str, dict] = {}

    for step in trajectory:
        room = extract_room_name(step["obs"]) or "?"
        if room == "?":
            continue

        exits     = parse_exits(step["obs"])
        tried_dir = _action_to_direction(step["action"])

        if room not in room_data:
            room_data[room] = {
                "exits": set(),
                "tried": set(),
                "visits": 0,
                "dir_first_hint": {},   # direction -> first step it appeared in obs
            }
        # Record first hint step for each newly seen direction
        for d in exits:
            if d not in room_data[room]["dir_first_hint"]:
                room_data[room]["dir_first_hint"][d] = step["step"]
        room_data[room]["exits"].update(exits)
        if tried_dir:
            room_data[room]["tried"].add(tried_dir)
        room_data[room]["visits"] += 1

    errors = []
    for room, data in room_data.items():
        if data["visits"] < min_visits:
            continue
        if not data["exits"]:
            continue
        missed = data["exits"] - data["tried"]
        if missed:
            # step = earliest hint step among the unexplored directions
            hint_step = min(
                data["dir_first_hint"].get(d, 0) for d in missed
            )
            errors.append(ErrorEvent(
                error_type="strategy/unexplored_destination",
                step=hint_step,
                description=(
                    f"Room '{room}': directions {sorted(missed)} hinted in obs but never tried "
                    f"({data['visits']} visit(s))"
                ),
                evidence={
                    "room": room,
                    "available_exits": sorted(data["exits"]),
                    "tried_directions": sorted(data["tried"]),
                    "unexplored": sorted(missed),
                    "first_hint_step": hint_step,
                    "first_hint_per_dir": {d: data["dir_first_hint"].get(d, 0)
                                           for d in sorted(missed)},
                    "visits": data["visits"],
                },
            ))

    return errors


# ── Top-level classifier ─────────────────────────────────────────────────────

def classify_errors(
    trajectory: list[dict],
    milestones: list[dict],
) -> dict:
    """
    Run all error detectors and return a structured result.

    Returns:
        {
          "system":    [ErrorEvent, ...],
          "strategy":  [ErrorEvent, ...],
          "operation": [ErrorEvent, ...],
          "summary": {
              "total": int,
              "by_type": {"system/format": int, ...},
          }
        }
    """
    system_errors    = detect_system_errors(trajectory)
    strategy_errors  = (
        detect_loops(trajectory)
        + detect_route_inefficiency(trajectory, milestones)
        + detect_unexplored_destination(trajectory)
    )
    operation_errors = (
        detect_feedback_blindness(trajectory)
        + detect_perception_errors(trajectory, milestones)
        + detect_decision_errors(trajectory, milestones)
    )

    all_errors = system_errors + strategy_errors + operation_errors
    type_counts: dict[str, int] = {}
    for e in all_errors:
        type_counts[e.error_type] = type_counts.get(e.error_type, 0) + 1

    return {
        "system":    [_event_to_dict(e) for e in system_errors],
        "strategy":  [_event_to_dict(e) for e in strategy_errors],
        "operation": [_event_to_dict(e) for e in operation_errors],
        "summary": {
            "total": len(all_errors),
            "by_type": type_counts,
        },
    }


def _event_to_dict(e: ErrorEvent) -> dict:
    return {
        "error_type":  e.error_type,
        "step":        e.step,
        "description": e.description,
        "evidence":    e.evidence,
    }
