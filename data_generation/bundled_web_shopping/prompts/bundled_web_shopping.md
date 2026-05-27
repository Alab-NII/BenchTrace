# Episode Failure Annotation Task

You are annotating failures made by an AI agent completing bundled product selection tasks in BundledWebShopping (a multi-step compatible-item selection environment).

## Input

You will be given a **snapshot**: the complete trajectory of one episode, including every action the agent took and every observation it received, along with the final progress score.

## Your Task

You will be given a rule-based draft annotation alongside the episode snapshot. Your job is to produce a revised, complete annotation by:

1. **Reviewing the draft**: correct any inaccurate `type`, `where`, or `why` fields; remove false positives
2. **Completing missing fields**: fill in `tier` (core/marginal) and `location` for every entry
3. **Adding missed failures**: identify any failures not caught by the rule-based classifier and add them to the list

---

## BundledWebShopping Environment Basics

### Task structure

The agent must select N products (typically N=6) to form a compatible bundle within a budget. Each product corresponds to one **subtask**, presented in order. The agent selects one option (A/B/C/...) per subtask.

**Example bundle types:** baking kit, skincare routine, home theater system, skincare set, grocery bundle.

### Observations and actions

- The observation is **fixed throughout the episode** — it is the full task description including all subtasks and their available options, shown at every step
- The agent reads the obs and produces a selection action for each subtask
- **Trajectory format:** odd-numbered steps (1, 3, 5...) contain the agent's reasoning and selection; even-numbered steps (0, 2, 4...) have `action=None`
- **Step limit:** typically 12 steps (6 subtasks × 2)

### Required action format

Each subtask action must contain a letter selection in the form `Selection: [X]` or `[X]` (where X is one of A/B/C/...). Reasoning text may precede the selection. No selection means the subtask scores 0.

### Compatibility cascade

Each option has a compatibility type (e.g., Gel, Hydrating, Leather, OLED). The **compatibility rules** specify which types pair well and which to avoid across subtasks:

```
"Gel pairs well with Astringent; Hydrating pairs well with Alcohol Free;
Gel avoids Alcohol Free; Hydrating avoids Astringent"
```

The type of the item selected in subtask N constrains which items are valid in subtasks N+1 through N+k. The agent must:
1. Correctly identify the type of the previously selected item from its text description
2. Apply the cascade rules to determine which options are compatible
3. Within compatible options, satisfy the subtask-level goal (highest-priced / lowest-priced / highest-rated)

### Price and rating visibility

**Prices and ratings are never shown in the observation.** The agent can only infer them from brand names, product descriptions, and world knowledge. The ground-truth price/rating is a hidden value used for scoring.

### Progress scoring

- `progress = (number of correctly selected subtasks) / N`
- Each subtask is scored independently: the agent's selection letter must match the ground-truth correct answer
- **A wrong selection does not block subsequent subtask attempts** — the agent can still respond to all N subtasks regardless of earlier errors
- Progress advances by 1/N per correctly answered subtask

### Goal vs. Preference

Each subtask may have both a **Goal** (e.g., "Buy the lowest-priced one") and a **Preference** (e.g., "Pick the highest-rated option among those compatible"). When they conflict, the Goal takes priority.

---

## What Counts as a Failure

A failure is any agent action (or absence of action) that causes the agent to:
- Submit an incorrect selection for a subtask (progress does not advance)
- Fail to submit any valid selection for a subtask
- Make a reasoning error that leads to a wrong letter choice

---

## Defined Failure Modes

Use the following taxonomy when labeling failures. Each failure must be assigned a `type` from this list. If a failure does not fit any existing type, you may define a new one (see **Adding New Failure Modes** below).

### Category 1: System

| Type | Definition | Where to annotate |
|---|---|---|
| `system/output_truncation` | The agent's action is cut off mid-reasoning (e.g., `<think>` block without closing `</think>`, reasoning that ends abruptly) with no `Selection: [X]` produced — caused by the model hitting a token limit before generating a final answer | The step where the truncated action occurs |
| `system/format` | The agent's action does not contain any recognizable letter selection (e.g., outputs free text describing products but no `[X]`, outputs `[✗]` or `[None]` as the selection letter, or outputs a number instead of a letter) | The step where the malformed action occurs |

### Category 2: Strategy (multi-step reasoning level)

| Type | Definition | Where to annotate |
|---|---|---|
| `strategy/wrong_cascade` | The agent correctly identifies the goal of the current subtask but applies the compatibility rules using an incorrect type label for a previously selected item — leading to a wrong cascade assumption that propagates through subsequent subtasks | The step where the wrong type inference first occurs; `where` extends through the last subtask affected |
| `strategy/cascade_propagation` | Starting from a wrong selection in an earlier subtask, the agent consistently builds subsequent reasoning on that wrong premise — the agent's logic is internally consistent but grounded in an incorrect earlier choice | From the first subtask that follows a wrong selection, to the last subtask of the episode |
| `strategy/goal_preference_conflict` | The agent explicitly prioritizes the subtask-level **Preference** over the **Goal** when they conflict in direction (e.g., chooses highest-priced when Goal says lowest-priced, citing the Preference as justification) | The step where the conflicting choice is made |

### Category 3: Operation (single-step decision level)

| Type | Definition | Where to annotate |
|---|---|---|
| `operation/unobservable_ranking` | The agent must select the highest-priced, lowest-priced, or highest-rated option, but prices/ratings are not shown — the agent makes an incorrect inference from brand name or product description and picks the wrong item | The step where the wrong ranking inference is made |
| `operation/type_misidentification` | The agent misidentifies the compatibility type of a previously selected item (e.g., labeling a "hydrating cleansing bar" as "gel") when applying cascade rules for the current subtask selection | The step where the misidentification leads to a wrong selection |
| `operation/semantic_refusal` | The agent determines that no available option fits the subtask label (e.g., "none of these are actual frostings") and refuses to select any option, even though the task's semantic mapping is broader than the literal label | The step where the refusal occurs |
| `operation/within_set_ranking_error` | The agent correctly identifies the compatible subset of options (right type) but picks the wrong item within that subset due to incorrect price/rating ranking — the compatibility reasoning is correct but the final letter choice is wrong | The step where the wrong item within the correct compatible set is chosen |

**Note**: `strategy/wrong_cascade` and `operation/type_misidentification` are closely related — use `type_misidentification` when the error is contained to the current step's type inference; use `wrong_cascade` when the misidentified type then drives incorrect compatibility reasoning in subsequent subtasks.

---

## Core vs. Marginal

**Core failure** (at most 3 per episode):
- Directly caused the agent to submit a wrong selection
- Fixing it alone would likely correct that subtask and potentially unblock subsequent subtasks
- Earlier subtask errors are more impactful (they trigger cascade propagation)

**Marginal failure**:
- Had minor or indirect impact on the outcome
- The agent's selection happened to be wrong due to inherent ambiguity (e.g., two equally plausible options with hidden price difference) rather than a reasoning error
- Is a downstream consequence of a core cascade failure, not an independent cause

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
    "location": "Brief description of which subtask and what the agent was trying to select",
    "why": "Root cause explanation: what the agent inferred wrong, and what it should have done instead",
    "where": [first_step, last_step],
    "note": "(optional) required if type is a newly defined mode — provide a one-sentence definition"
  }
]
```

- `failure_id`: short descriptive identifier, e.g. `"think_truncated_step3"`, `"frosting_refusal"`, `"hydrating_typed_as_gel"`
- `location`: describe in terms of subtask number and product category, e.g. `"subtask 2 (frosting selection), baking bundle"`, `"subtask 4 (content source), electronics bundle"`
- `why`: explain both the reasoning error and what the correct inference should have been, citing specific options and compatibility rules from the obs where relevant
- `where`: 0-indexed step indices — for a single subtask error, use `[step, step]`; for cascade propagation spanning multiple subtasks, use `[first_affected_step, last_subtask_step]`

If no failures are detected, return an empty array `[]`.

---

## Notes

- **`system/output_truncation` is overwhelmingly the most common failure for Qwen/Qwen3-32B** (>88% of its action steps). When annotating Qwen episodes, check every odd-numbered step for whether `</think>` and `Selection: [X]` are present before looking for reasoning errors.
- **For GPT-4.1 episodes**, the dominant failure is `operation/unobservable_ranking` at subtask 1 (prices/ratings are hidden), followed by `strategy/cascade_propagation` from that initial error.
- **A wrong selection does not prevent the agent from answering later subtasks.** Annotate each subtask's failure independently; use `strategy/cascade_propagation` only when the agent's *reasoning* at a later step explicitly (and incorrectly) references a wrong earlier choice.
- **`operation/semantic_refusal`** (outputting `[✗]` or `[None]`) should be annotated as `system/format` if the output contains no letter, or `operation/semantic_refusal` if the agent explicitly reasons that no option is valid.
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
