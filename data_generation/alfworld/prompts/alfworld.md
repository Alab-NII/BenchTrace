# Episode Failure Annotation Task

You are annotating failures made by an AI agent completing household tasks in AlfWorld (a text-based embodied environment).

## Input

You will be given a **snapshot**: the complete trajectory of one episode, including every action the agent took, every game response it received, the list of admissible commands available at each step, and the final progress score.

## Your Task

You will be given a rule-based draft annotation alongside the episode snapshot. Your job is to produce a revised, complete annotation by:

1. **Reviewing the draft**: correct any inaccurate `type`, `where`, or `why` fields; remove false positives
2. **Completing missing fields**: fill in `tier` (core/marginal) and `location` for every entry
3. **Adding missed failures**: identify any failures not caught by the rule-based classifier and add them to the list

---

## AlfWorld Environment Basics

- **Navigation**: `go to <location>` (e.g., `go to drawer 1`, `go to microwave 1`)
- **Object interaction**: `take <obj> from <location>`, `move <obj> to <receptacle>`, `open/close <location>`, `examine <location>`
- **Transformation** (task-specific): `heat <obj> with <appliance>`, `cool <obj> with <appliance>`, `clean <obj> with <appliance>`, `use desklamp 1`
- **Admissible commands** are provided at every step — the agent always knows which exact actions are valid
- **Progress** ranges from 0.0 to 1.0; partial credit is given for completing sub-goals
- **Step limit**: 50 steps per episode

### Task Types and Required Action Sequences

| Task type | Required sequence |
|---|---|
| `pick_and_place_simple` | find object → pick up → go to destination → place |
| `pick_two_obj_and_place` | find object 1 → pick up → place; find object 2 → pick up → place |
| `pick_heat_then_place_in_recep` | find object → pick up → go to microwave → `heat obj with microwave` → go to destination → place |
| `pick_cool_then_place_in_recep` | find object → pick up → go to fridge → `cool obj with fridge` → go to destination → place |
| `pick_clean_then_place_in_recep` | find object → pick up → go to sinkbasin → `clean obj with sinkbasin` → go to destination → place |
| `look_at_obj_in_light` | find desklamp → `use desklamp 1`; find target object → pick up → `use desklamp 1` (while holding object) |

---

## What Counts as a Failure

A failure is any agent action (or sequence of actions) that causes the agent to:
- Exhaust the 50-step budget without completing the task (progress < 1.0)
- Enter a repetitive loop with no meaningful progress increase

---

## Defined Failure Modes

Use the following taxonomy when labeling failures. Each failure must be assigned a `type` from this list. If a failure does not fit any existing type, you may define a new one (see **Adding New Failure Modes** below).

### Category 1: System

| Type | Definition | Where to annotate |
|---|---|---|
| `system/format` | The agent outputs a malformed action (empty, multi-line, JSON leakage, excessively long, or starting with a non-letter character) | The step where the malformed action occurs |

### Category 2: Strategy (multi-step planning level)

| Type | Definition | Where to annotate |
|---|---|---|
| `strategy/loop` | Within a no-progress segment (progress does not increase), the agent repeats the same (location, action) pair | The first repeated (location, action) pair in the segment; `where` covers the full loop segment |
| `strategy/exhaustive_rescan` | The agent repeatedly revisits and re-examines locations it has already fully searched, without committing to any object it previously found | From the first redundant revisit to the end of the no-progress segment |
| `strategy/missing_destination` | The agent holds the correct object (or has completed a processing step) but never navigates to the required destination location during the relevant phase | From the step the agent acquires the object/completes processing to the end of the episode |
| `strategy/missing_processing_step` | The agent's plan entirely omits the required transformation step (`heat`/`cool`/`clean`/`use desklamp`) — the agent never attempts the command even when holding the object near the correct appliance | The step where the agent first has the object and is near the appliance (missed opportunity); `where` extends to the end of the episode |
| `strategy/goal_misinterpretation` | The agent treats the task as a different task type from the very beginning of the episode (e.g., executing pick-and-place behavior for a `look_at_obj_in_light` task) | Step 0 to end of episode |

### Category 3: Operation (single-step execution level)

| Type | Definition | Where to annotate |
|---|---|---|
| `operation/feedback_blindness` | The agent repeats the same action after receiving an explicit negative or null response for it in the immediately preceding observation | The step of the first repeated occurrence after the negative feedback |
| `operation/wrong_object_pick` | The agent picks up an object that is semantically similar to but categorically different from the required target (e.g., picking `spraybottle` when the task requires `perfume`, or `cup` when the task requires `egg`) | The step where the wrong pick action occurs |
| `operation/move_command_avoidance` | The agent is at the correct destination location with the target object in inventory, the admissible commands explicitly include `move <obj> to <receptacle>`, but the agent issues a passive command (`look`, `examine`, `go to`) instead | The step where the move command is available but not taken |

**Note**: `strategy/missing_processing_step` and `strategy/goal_misinterpretation` are mutually exclusive — if the agent misidentifies the task from the start, use `goal_misinterpretation` rather than `missing_processing_step`.

---

## Core vs. Marginal

**Core failure** (at most 3 per episode):
- Directly caused the agent to stop making meaningful progress
- Fixing it alone would likely unlock significant further progress
- Occurs earlier in the episode (earlier blockages matter more, since later sub-goals become unreachable once blocked)

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
    "location": "Brief description of the in-game location and context where this failure occurred",
    "why": "Root cause explanation: what the agent did wrong, and what it should have done instead",
    "where": [first_step, last_step],
    "note": "(optional) required if type is a newly defined mode — provide a one-sentence definition"
  }
]
```

- `failure_id`: short descriptive identifier, e.g. `"heat_step_skipped"`, `"repeated_examine_drawer"`, `"wrong_spray_picked"`
- `location`: describe in terms of the in-game location and task context, e.g. `"at microwave 1 with egg 2 in inventory"`, `"scanning cabinet 1–5 after failing to find soapbottle"`
- `why`: explain both the mistake and the correct action, referencing specific objects and locations from the trajectory
- `where`: 0-indexed step indices where this failure manifests in the trajectory

If no failures are detected, return an empty array `[]`.

---

## Notes

- Annotate all failures you can identify, not just the most obvious one.
- When admissible commands at a step contain the correct action and the agent ignores it, this is strong evidence for an operation-level failure.
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
