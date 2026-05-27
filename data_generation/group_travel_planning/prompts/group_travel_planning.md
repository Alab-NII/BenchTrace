# Episode Failure Annotation Task

You are annotating failures made by an AI agent completing group travel planning tasks in GroupTravelPlanning (a multi-traveler itinerary generation environment).

## Input

You will be given a **snapshot**: the complete trajectory of one episode, including every action the agent took and every observation it received, along with the final progress score.

## Your Task

You will be given a rule-based draft annotation alongside the episode snapshot. Your job is to produce a revised, complete annotation by:

1. **Reviewing the draft**: correct any inaccurate `type`, `where`, or `why` fields; remove false positives
2. **Completing missing fields**: fill in `tier` (core/marginal) and `location` for every entry
3. **Adding missed failures**: identify any failures not caught by the rule-based classifier and add them to the list

---

## GroupTravelPlanning Environment Basics

### Task structure

The agent must generate personalized daily itineraries for N travelers (N = 5–8) who are joining a group trip. Each traveler corresponds to one **subtask**, presented in order.

A **base itinerary** for the group's lead traveler is shown at every step — it specifies destinations, restaurants, hotels, and attractions by name only, with **no prices or ratings**.

### Subtask observations and actions

- Each subtask observation contains: (1) the base itinerary, (2) the current traveler's constraints
- The agent generates a free-text daily plan for that traveler
- Previous travelers' plans from the **same episode** are injected into the prompt as `=== Memory: Previous Travelers This Episode ===` (not visible in the `obs` field, but present in what the agent receives — infer from prior action steps)
- Reflections from **previous failed episodes** are injected as `=== Lessons from Previous Attempts ===`
- **Trajectory format:** even-numbered steps (0, 2, 4...) are observations with `action=None`; odd-numbered steps (1, 3, 5...) contain the agent's generated plan

### Constraint types

Each traveler's request may include:

| Constraint type | Example |
|---|---|
| **Food type** | "must include Chinese and Japanese cuisine" |
| **Price range** | "breakfast budget: $10–$20 per person" |
| **Rating range** | "restaurant rated above 4.2" |
| **JOIN** | "use the same restaurant as Traveler 2 for Day 3 dinner" |
| **RELATION** | "accommodation rated higher than Traveler 1's Day 2 hotel" |
| **RELATION (arithmetic)** | "dinner price at least $20 less than Traveler 3's second-day dinner" |

JOIN and RELATION constraints require the agent to **read and reference prior travelers' plans from memory**. The base itinerary contains no prices or ratings — the agent must use world knowledge or the memory text to infer these values.

### Progress scoring

- `progress = average constraint satisfaction rate across all N subtasks`
- Each subtask is scored independently by an LLM judge: `subtask_progress = n_satisfied / n_total`
- `won = True` when `progress >= 1.0` (all travelers' constraints fully satisfied)
- A subtask with `subtask_progress < 1.0` always has at least one unmet constraint

### Impossible RELATION constraints

Some RELATION constraints are arithmetically impossible because the referenced traveler's plan hallucinated a very low price (e.g., $12 for a dinner), making a "at least $20 less" constraint require a negative price. This is a recurring structural issue. Both models encounter it; Qwen3-32B tends to loop indefinitely, while GPT-4.1 tends to commit to an out-of-range value with rationalization.

---

## What Counts as a Failure

A failure is any agent action (or absence of action) that causes the agent to:
- Produce a plan that does not satisfy one or more explicit traveler constraints (subtask_progress < 1.0)
- Fail to produce any plan for a subtask (output_truncation)
- Make a reasoning error that leads to a constraint violation across multiple subtasks

---

## Defined Failure Modes

Use the following taxonomy when labeling failures. Each failure must be assigned a `type` from this list. If a failure does not fit any existing type, you may define a new one (see **Adding New Failure Modes** below).

### Category 1: System

| Type | Definition | Where to annotate |
|---|---|---|
| `system/output_truncation` | The agent's `<think>` block opens but never closes and no plan is produced — the action contains only internal monologue without an itinerary. Caused by the model entering a reasoning loop (typically triggered by an impossible RELATION constraint) or hitting a token limit before generating output. | The step where the truncated action occurs |

### Category 2: Strategy (multi-step planning level)

| Type | Definition | Where to annotate |
|---|---|---|
| `strategy/reflection_drift` | After a failed episode, the reflection correctly names the failing traveler but not the root cause. The next episode fixes that traveler's subtask while inadvertently breaking a different one. The failure migrates across travelers from episode to episode without converging. Observable within a snapshot when: (1) prior-episode reflections in the prompt name traveler X as the problem, and (2) the current episode fails on a different traveler Y for an unrelated reason. | The step of the newly failing subtask where the drift becomes apparent |

### Category 3: Operation (single-step decision level)

| Type | Definition | Where to annotate |
|---|---|---|
| `operation/cross_traveler_reference_error` | A JOIN or RELATION constraint requires matching a value from a previous traveler's plan (price, rating, restaurant name, room type). The agent either ignores the memory context and independently halluciniates a different value, or misreads the memory and derives the wrong reference — leading to an inconsistent or incorrect plan. | The step where the wrong reference leads to a constraint violation |
| `operation/constraint_range_violation` | The agent explicitly computes the valid range for a price or rating constraint, recognizes that its chosen venue is outside that range, but commits to the violation anyway — typically with a rationalization note ("closest available option", "adjusted to realistic price"). | The step where the out-of-range choice is committed |
| `operation/cuisine_type_mismatch` | The agent assigns an incorrect cuisine category to a venue in order to satisfy a food type constraint — e.g., claiming a beignet stand serves Mexican food, or omitting a required cuisine while listing only some of the required types. | The step where the misclassified venue is chosen |
| `operation/accommodation_type_mismatch` | When matching a "same room type as Traveler X" constraint, the agent substitutes a different accommodation category (e.g., "Entire Apartment" instead of "Private Room", "Suite" instead of "King Room") and argues equivalence. | The step where the mismatched room type is chosen |

**Note**: `system/output_truncation` and `operation/constraint_range_violation` often have the same root cause (an arithmetically impossible RELATION constraint). Annotate `system/output_truncation` when no plan is produced (Qwen loop); annotate `operation/constraint_range_violation` when a plan is produced but contains an acknowledged out-of-range value (GPT rationalization). The `why` field should identify the impossible arithmetic in both cases.

---

## Core vs. Marginal

**Core failure** (at most 3 per episode):
- Directly caused the agent to submit a plan with one or more unsatisfied constraints
- Fixing it alone would likely bring `subtask_progress` to 1.0 for that traveler
- Earlier subtask failures are more impactful (they can corrupt memory context for later travelers)

**Marginal failure**:
- Had minor or indirect impact on the outcome
- The constraint was borderline (e.g., the agent's price was $1 outside the stated range and there is genuine ambiguity about the exact bound)
- Is a downstream consequence of a core failure (e.g., a JOIN constraint fails because the referenced traveler's plan was wrong)

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
    "location": "Brief description of which subtask and traveler this failure occurred at",
    "why": "Root cause explanation: what the agent inferred wrong, and what it should have done instead",
    "where": [first_step, last_step],
    "note": "(optional) required if type is a newly defined mode — provide a one-sentence definition"
  }
]
```

- `failure_id`: short descriptive identifier, e.g. `"brenda_impossible_price"`, `"room_type_substitution_subtask3"`, `"qwen_loop_traveler2"`
- `location`: describe in terms of subtask number and traveler name/index, e.g. `"subtask 3 (Brenda), Day 2 dinner constraint"`, `"subtask 4 (Nathan), accommodation type matching"`
- `why`: explain the constraint that was violated, what the agent did, and what it should have done — reference specific values from the trajectory (e.g., "Thomas's dinner price hallucinated as $12; Brenda needs at least $20 less, requiring ≤ -$8 which is impossible")
- `where`: 0-indexed step indices — for a single subtask failure use `[step, step]`; for drift spanning multiple steps use `[first_step, last_step]`

If no failures are detected, return an empty array `[]`.

---

## Notes

- **Memory context is not in the `obs` field.** The agent receives previous travelers' plans via a memory prompt injection, not recorded in `obs`. To understand what memory the agent had for subtask N, read the action fields of subtasks 0 through N-1 in the same episode.
- **The base itinerary contains no prices or ratings.** Any price or rating cited by the agent for a restaurant or hotel is hallucinated from world knowledge. This is expected behavior — but when a RELATION constraint references another traveler's price, both the reference and the derived price must be consistent.
- **Impossible RELATION constraints are a known dataset issue.** When you see a RELATION arithmetic constraint that would require a negative or near-zero price, the agent cannot satisfy it correctly. Focus your annotation on how the agent responds (output_truncation vs. constraint_range_violation), not on the constraint itself.
- **Qwen3-32B has `<think>` blocks.** All of Qwen's reasoning appears inside `<think>...</think>` before the plan. If `</think>` is absent and no plan follows, annotate `system/output_truncation`. If `</think>` is present and a plan follows, evaluate the plan for operation-level failures.
- **For GPT-4.1**, the dominant failure is `operation/constraint_range_violation` — the model tends to commit to a slightly out-of-range value with inline acknowledgment rather than looping.
- **`strategy/reflection_drift`** requires observing the reflection text in the prompt (under `=== Lessons from Previous Attempts ===`): if the reflection names traveler X as the problem but the current episode fails on traveler Y instead, annotate drift.
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
