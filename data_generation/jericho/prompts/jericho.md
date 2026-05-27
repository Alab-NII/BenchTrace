# Episode Failure Annotation Task

You are annotating failures made by an AI agent playing a Jericho text-based adventure game.

## Input

You will be given a **snapshot**: the complete trajectory of one episode, including every action the agent took and every game response it received, along with the final score.

## Your Task

You will be given a rule-based draft annotation alongside the episode snapshot. Your job is to produce a revised, complete annotation by:

1. **Reviewing the draft**: correct any inaccurate `type`, `where`, or `why` fields; remove false positives
2. **Completing missing fields**: fill in `tier` (core/marginal) and `location` for every entry
3. **Adding missed failures**: identify any failures not caught by the rule-based classifier and add them to the list

---

## What Counts as a Failure

A failure is any agent action (or sequence of actions) that causes the agent to:
- Die (health reaches zero, or the game explicitly states the agent has died)
- Get stuck at a location with no way to progress
- Enter a repetitive loop with no meaningful progress

---

## Defined Failure Modes

Use the following taxonomy when labeling failures. Each failure must be assigned a `type` from this list. If a failure does not fit any existing type, you may define a new one (see **Adding New Failure Modes** below).

### Category 1: System

| Type | Definition | Where to annotate |
|---|---|---|
| `system/format` | The agent outputs a malformed action (empty, multi-line, JSON leakage, or excessively long) | The step where the malformed action occurs |

### Category 2: Strategy (path planning level)

| Type | Definition | Where to annotate |
|---|---|---|
| `strategy/loop` | Within a no-progress segment (no score increase), the agent repeats the same (room, action) pair | The first repeated (room, action) pair in the segment; `where` covers the full loop segment |
| `strategy/route_inefficiency` | The number of steps taken between two consecutive milestones exceeds 2× the walkthrough reference | The step where the agent first reaches milestone N−1; `where` ends when it reaches milestone N (or episode ends) |
| `strategy/unexplored_direction` | The game's observation text explicitly mentions an available exit direction, but the agent passes through that room multiple times without ever trying that direction | From the first step where the agent passes through the room without trying the direction to the end of the episode |

### Category 3: Operation (action execution level)

| Type | Definition | Where to annotate |
|---|---|---|
| `operation/feedback_blindness` | The agent repeats the same action after receiving negative feedback for it (evaluated over the full episode) | The step of the first repeated occurrence after the negative feedback |
| `operation/perception_error` | The agent enters the correct room for a milestone but never attempts the correct action or any same-verb variant | The first step where the agent enters the correct room |
| `operation/decision_error` | The agent attempts the correct action but at the wrong time (reward = 0), or uses the correct verb with the wrong object | The step where the incorrect attempt occurs |

**Note**: `perception_error` and `decision_error` are mutually exclusive for the same milestone — report only one.

---

## Core vs. Marginal

**Core failure** (at most 3 per episode):
- Directly caused the agent to stop making meaningful progress
- Fixing it alone would likely unlock significant further exploration
- Occurs earlier in the episode (earlier blockages matter more, since later content becomes unreachable once blocked)

**Marginal failure**:
- Had minor or indirect impact on the outcome
- The agent recovered or continued meaningfully after it occurred
- Is a downstream consequence of a core failure, not an independent cause

When uncertain, prefer fewer core failures.

---

## Adding New Failure Modes

If a failure in the episode clearly does not fit any of the types above, you may define a new one. Use the format:

```
`<category>/<name>`
```

where `<category>` is one of `system`, `strategy`, or `operation`. Include a one-sentence definition in the `note` field of the output entry. New types will be reviewed and potentially added to the taxonomy.

---

## Output Format

Return a JSON array. Each element represents one identified failure:

```json
[
  {
    "failure_id": "short_snake_case_identifier",
    "type": "category/subtype",
    "tier": "core" | "marginal",
    "location": "Brief description of the game location and context where this failure occurred",
    "why": "Root cause explanation: what the agent did wrong, and what it should have done instead",
    "where": [first_step, last_step],
    "note": "(optional) required if type is a newly defined mode — provide a one-sentence definition"
  }
]
```

- `failure_id`: short descriptive identifier, e.g. `"lamp_not_lit"`, `"repeated_north_after_blocked"`
- `location`: describe in terms of game location and context, e.g. `"West of House, before entering the cellar"`
- `why`: explain both the mistake and the correct action
- `where`: 0-indexed step indices where this failure manifests in the trajectory

If no failures are detected, return an empty array `[]`.

---

## Notes

- Annotate all failures you can identify, not just the most obvious one.
- The `why` correct-action hint is for analysis purposes — it will not be shown to the agent being evaluated.
- Do not infer failures that are not evidenced in the trajectory.

---

## User Prompt Template

```
## Episode Snapshot

<snapshot>
{SNAPSHOT}
</snapshot>

## Rule-Based Annotation (Draft)

The following failures were automatically detected by a rule-based classifier. Each entry contains `type`, `where`, and a template-generated `why`. Please revise this list:
- Fill in `tier` (core/marginal) and `location` for every entry
- Correct any inaccurate `type`, `where`, or `why` fields
- Remove any false positives
- Add any failures that were missed

<draft_annotation>
{RULE_BASED_ANNOTATION}
</draft_annotation>

Return the revised annotation as a JSON array in the format specified above.
```
