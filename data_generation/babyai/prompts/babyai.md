# Episode Failure Annotation Task

You are annotating failures made by an AI agent completing navigation and manipulation tasks in BabyAI (a text-based 2D grid-world environment).

## Input

You will be given a **snapshot**: the complete trajectory of one episode, including every action the agent took and every observation it received, along with the final progress score.

## Your Task

You will be given a rule-based draft annotation alongside the episode snapshot. Your job is to produce a revised, complete annotation by:

1. **Reviewing the draft**: correct any inaccurate `type`, `where`, or `why` fields; remove false positives
2. **Completing missing fields**: fill in `tier` (core/marginal) and `location` for every entry
3. **Adding missed failures**: identify any failures not caught by the rule-based classifier and add them to the list

---

## BabyAI Environment Basics

### Grid mechanics

- The agent occupies a cell in a 2D grid and has a **facing direction** (north / south / east / west)
- To reach an object, the agent must: (1) **turn** to align with the object, (2) **move forward** until directly adjacent
- Interaction (pick up / toggle) requires the target to be **exactly 1 step directly in front**, with **zero lateral offset**
- Walls block movement silently — `move forward` against a wall leaves the agent in place and the observation unchanged

### Observations

Each observation provides:
- The current **facing direction** (north / south / east / west)
- A list of **visible objects** with **relative egocentric positions** — two phrasings appear across episodes, both mean the same thing:
  - Compact: `"blue door (closed) (5 steps ahead, 2 to the right)"`
  - Verbose: `"There is a blue closed door 1 5 steps in front of you and 2 steps to your right."`
- Whether the agent is carrying an object

**Interpretation**: an object with any lateral offset ("M steps to the right/left") is NOT reachable by moving forward — the agent must first turn to align, then approach.

### Task type

The task type is encoded in the `Mission:` line of every observation (e.g., `Mission: open a red door`). Use this to determine the required action sequence from the table below. Some episodes may not have an explicit `task_type` metadata field — always infer from the Mission text.

### Action set

| Action | Effect |
|---|---|
| `turn left` | Rotate 90° counter-clockwise; position unchanged |
| `turn right` | Rotate 90° clockwise; position unchanged |
| `move forward` | Advance 1 cell in facing direction (no-op if wall is in front) |
| `pick up` | Pick up the object 1 step directly in front; fails silently if not aligned |
| `drop` | Drop the currently carried object 1 step in front |
| `toggle` | Open/close a door or activate a switch 1 step directly in front; fails silently if not adjacent |
| `done` | Declare task complete |

### Field of view

Cone-shaped; the agent cannot see through walls or behind itself. Objects in other rooms are invisible until the agent enters that room.

### Progress

- Ranges from 0.0 to 1.0
- Partial credit is given for completing sub-goals (e.g., picking up the key before unlocking the door)
- A progress of 0.95 with `n_steps` at the episode limit typically means the agent ran out of steps without completing the final action
- **Step limits**: varies by task level (64 or 128 steps)

### Task types and required action sequences

Use the `Mission:` line to map to a task type: "go to" → GoTo, "open" → Open, "pick up" → Pickup, "put" → PutNextLocal, "unlock" → UnlockLocal.

| Task type | Required sequence |
|---|---|
| **GoTo** (GoToRedBall, GoToDoor, GoToObjMaze, etc.) | Navigate until the agent is **adjacent** to the target object (win by proximity — do NOT toggle) |
| **Open / OpenDoor / OpenDoorColor** | Navigate to the target door → `toggle` (must be 1 step in front, facing it) |
| **Pickup / PickupLoc / UnblockPickup** | Navigate to target object → `pick up` (must be 1 step in front, facing it); UnblockPickup may require moving a blocking object first |
| **PutNextLocal** | Pick up object A → navigate to object B → `drop` adjacent to B |
| **UnlockLocal / GoToImpUnlock** | Find key → `pick up` key → navigate to locked door → `toggle` to unlock → navigate to target |
| **OpenDoorsOrderN4** | Open multiple doors in the specified sequence; each requires adjacent toggle |

**Critical**: GoTo tasks are won by **being adjacent** to the target, not by toggling it. Toggling a door in a GoTo task is a wrong-action error.

---

## What Counts as a Failure

A failure is any agent action (or sequence of actions) that causes the agent to:
- Exhaust the step budget without completing the task (progress < 1.0)
- Enter a repetitive pattern with no meaningful progress increase

---

## Defined Failure Modes

Use the following taxonomy when labeling failures. Each failure must be assigned a `type` from this list. If a failure does not fit any existing type, you may define a new one (see **Adding New Failure Modes** below).

### Category 1: System

| Type | Definition | Where to annotate |
|---|---|---|
| `system/format` | The agent outputs a malformed action (empty string, multi-line text, JSON/code leakage, or a string that is not a valid BabyAI action) | The step where the malformed action occurs |

### Category 2: Strategy (multi-step planning level)

| Type | Definition | Where to annotate |
|---|---|---|
| `strategy/wall_running` | The agent repeatedly executes `move forward` despite the observation remaining unchanged across multiple consecutive steps, indicating it is blocked against a wall, and never attempts to turn | From the first step where the observation stops changing to the end of the no-progress segment |
| `strategy/lateral_target_ignore` | The target object is persistently visible in observations with a non-zero lateral offset ("M steps to your right/left"), but the agent keeps moving forward without turning to align — the offset does not decrease over many steps | From the first step where the lateral target is visible and forward movement begins, to the end of the no-progress segment |
| `strategy/loop` | The agent cycles through a repeating sequence of (facing direction, action) pairs — such as alternating `turn left` / `turn right`, or repeatedly taking the same turn-and-move circuit — without reducing its distance to the target | The first step of the repeating pattern; `where` covers the full loop segment |
| `strategy/room_boundary_deadlock` | In a multi-room task, the target is in a different room, but the agent never exits the starting room (never successfully traverses a door) and makes no progress toward the target | From the first step to the end of the episode |

### Category 3: Operation (single-step execution level)

| Type | Definition | Where to annotate |
|---|---|---|
| `operation/pickup_misalignment` | The agent issues `pick up` but the target object is not exactly 1 step directly in front (either the lateral offset is non-zero, or the distance is > 1); the pick-up fails silently and the agent does not diagnose the failure | The step where the misaligned pick-up occurs |
| `operation/toggle_misalignment` | The agent issues `toggle` but the target door or switch is not exactly 1 step directly in front; the toggle fails silently | The step where the misaligned toggle occurs |
| `operation/wrong_action_type` | The agent uses the wrong action category for the task goal: (a) toggling a door in a GoTo task (win condition is adjacency, not opening); (b) issuing `done` before the task is complete; (c) issuing `pick up` when the task requires navigation only | The step where the wrong action is issued |
| `operation/wrong_target_attribute` | The agent navigates to and attempts to interact with an object of the wrong color or type (e.g., approaching a blue door when the task requires opening the red door, or picking up a grey ball when the task requires a red ball) | The step where the agent first attempts to interact with the wrong object |

**Note**: `strategy/wall_running` and `strategy/lateral_target_ignore` are the two most common BabyAI failure modes and often co-occur. Annotate both if both are present; use `tier` to distinguish which had greater impact.

---

## Core vs. Marginal

**Core failure** (at most 3 per episode):
- Directly caused the agent to stop making meaningful progress
- Fixing it alone would likely enable significant further progress
- Occurs earlier in the episode (earlier blockages matter more)

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
    "location": "Brief description of the agent's position and context where this failure occurred",
    "why": "Root cause explanation: what the agent did wrong, and what it should have done instead",
    "where": [first_step, last_step],
    "note": "(optional) required if type is a newly defined mode — provide a one-sentence definition"
  }
]
```

- `failure_id`: short descriptive identifier, e.g. `"wall_run_north"`, `"toggle_wrong_door"`, `"ball_lateral_ignored"`
- `location`: describe in terms of agent facing and visible objects, e.g. `"facing east, red ball visible 5 steps ahead and 2 to the right"`, `"starting room, locked grey door blocking north exit"`
- `why`: explain both the mistake and the correct action — reference specific observations (object positions, facing direction) from the trajectory
- `where`: 0-indexed step indices where this failure manifests in the trajectory

If no failures are detected, return an empty array `[]`.

---

## Notes

- In BabyAI, **failed actions are silent** — a blocked `move forward`, a misaligned `pick up`, or a distant `toggle` all produce an unchanged observation. This is the primary diagnostic signal: if the observation does not change after an action, the action had no effect.
- An object with non-zero lateral offset ("M steps to your right/left") cannot be picked up or toggled from the current position regardless of forward distance.
- Do not infer failures that are not evidenced in the trajectory.
- The `why` correct-action hint is for analysis purposes — it will not be shown to the agent being evaluated.

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
