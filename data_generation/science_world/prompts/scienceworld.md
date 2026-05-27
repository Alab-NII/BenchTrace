# Episode Failure Annotation Task

You are annotating failures made by an AI agent completing science experiment tasks in ScienceWorld (a text-based multi-room environment for scientific reasoning).

## Input

You will be given a **snapshot**: the complete trajectory of one episode, including every action the agent took, every game response it received, and the final progress score.

## Your Task

You will be given a rule-based draft annotation alongside the episode snapshot. Your job is to produce a revised, complete annotation by:

1. **Reviewing the draft**: correct any inaccurate `type`, `where`, or `why` fields; remove false positives
2. **Completing missing fields**: fill in `tier` (core/marginal) and `location` for every entry
3. **Adding missed failures**: identify any failures not caught by the rule-based classifier and add them to the list

---

## ScienceWorld Environment Basics

### Rooms and navigation

The environment contains multiple rooms: living room, hallway, kitchen, bedroom, greenhouse, workshop, art studio, bathroom, outside. The agent starts in a random room. Navigation: `go door to <room>` (door must first be opened with `open door to <room>`).

### Action space

Free-form natural language commands. Common valid forms include:
- Navigation: `open door to kitchen`, `go door to kitchen`, `look around`
- Object interaction: `pick up <object>`, `put <object> in <container>`, `move <object> to <location>`, `open <container>`, `look in <container>`, `look at <object>`
- Appliance control: `turn on stove`, `turn off stove`, `turn on oven`, `open oven`
- Instruments: `move thermometer to stove`, `activate stopwatch`, `deactivate stopwatch`
- Electrical: `connect <wire> terminal 1 to <component> anode` (see FM-3 for exact syntax)
- Experiment: `mix <container>`, `focus on <object>`, `reset task`, `wait`

Invalid actions return: `"Unknown action. Type 'help' for a list of actions"` or `"No known action matches that input."`

### Disambiguation protocol

When a command matches multiple objects, the environment returns:
```
Ambiguous request: Please enter the number for the action you intended (or blank to cancel):
0: pick up apple (in bowl, in kitchen)
1: pick up apple (in fridge, in kitchen)
```
The agent **must** respond with a bare digit (`0`, `1`, etc.) in the **very next action**. Any other response (free text, `look around`, blank) cancels the disambiguation state and the digit becomes invalid.

### Progress

Ranges 0.0 to 1.0. Partial credit is given for completing sub-goals (reaching the correct room, finding the target substance, completing intermediate steps). **Step limit**: 100 steps.

### Task types and required action sequences

| Task type | Required sequence |
|---|---|
| `melt` | Find target substance → pick up → place in heat source (stove/oven) → turn on heat source → wait until melted |
| `boil` | Find target liquid → place in metal pot → put on stove → turn on stove → wait until boiling |
| `grow-plant` | Go to greenhouse → find seed → plant in soil/flower pot → water repeatedly with jug → apply fertilizer → wait for growth |
| `power-component` | Go to workshop → wire battery to switch to component using correct anode/cathode/terminal syntax → activate switch |
| `measure-melting-point-known-substance` | Find target substance → place in metal pot → put on stove → move thermometer to stove → turn on stove → wait → read temperature at melting point → `focus on` correct box |
| `find-living-thing` | Locate a living organism (plant in pot or bee) → pick up → move to correct colored box → `focus on <organism>` |
| `chemistry-mix` | Find both chemical substances → place in **metal pot** → `mix metal pot` → observe product |
| `use-thermometer` | Pick up thermometer → `use thermometer on <substance>` → read temperature → `focus on` correct color box |
| `inclined-plane-friction-named-surfaces` | Move block onto inclined plane with surface A → `wait` → `look at inclined plane` to read % slid → repeat for surface B → `focus on` surface with less sliding (higher friction) |
| `test-conductivity-of-unknown-substances` | Go to workshop → wire circuit → test each substance by inserting in circuit → observe whether lamp lights → `focus on` correct box |

---

## What Counts as a Failure

A failure is any agent action (or sequence of actions) that causes the agent to:
- Exhaust the 100-step budget without completing the task (progress < 1.0)
- Enter a repetitive pattern with no meaningful progress increase

---

## Defined Failure Modes

Use the following taxonomy when labeling failures. Each failure must be assigned a `type` from this list. If a failure does not fit any existing type, you may define a new one (see **Adding New Failure Modes** below).

### Category 1: System

| Type | Definition | Where to annotate |
|---|---|---|
| `system/format` | The agent outputs a malformed action that the environment cannot parse (empty string, multi-line text, partial commands truncated mid-word) | The step where the malformed action occurs |

### Category 2: Strategy (multi-step planning level)

| Type | Definition | Where to annotate |
|---|---|---|
| `strategy/look_around_loop` | The agent repeatedly issues `look around` as its primary action across many consecutive steps, making no experimental progress — not triggered by a specific failure but emerging as a planning breakdown | From the first step where `look around` begins repeating without intervening productive actions, to the end of the no-progress segment |
| `strategy/wrong_room_deadlock` | The agent spends the majority of the episode in a room irrelevant to the task (e.g., attempting electrical wiring actions in the kitchen for a `melt` task, or searching for living things in the workshop), never navigating to the task-relevant room | From the first step to the end of the episode |
| `strategy/reset_overuse` | The agent issues `reset task` multiple times in succession (5+) without attempting any new approach between resets, burning step budget | The first step of the first reset in the overuse sequence; `where` covers the full reset sequence |
| `strategy/missing_experimental_step` | The agent's plan entirely omits a required sub-step specific to the task type: applying fertilizer in `grow-plant`, placing the thermometer on the heat source in `measure-melting-point`, using `focus on` after placement in `find-living-thing`, issuing `mix metal pot` in `chemistry-mix` | The last step before the step limit where the missing action could still have been taken |
| `strategy/wrong_measurement_method` | The agent uses an incorrect measurement approach for the task (e.g., using the stopwatch to time block sliding on an inclined plane instead of reading the `% down the plane` percentage; measuring ambient temperature instead of moving the thermometer to the heat source) | From the first step where the wrong method is applied to the end of the no-progress segment |

### Category 3: Operation (single-step execution level)

| Type | Definition | Where to annotate |
|---|---|---|
| `operation/disambiguation_failure` | When the environment returns an `"Ambiguous request"` prompt, the agent does not respond with a bare digit in the very next action — either responding with free text, issuing `look around` first, or sending the digit too late after the state has been cancelled | The step immediately following the `"Ambiguous request"` observation where the wrong response is given |
| `operation/electrical_wiring_syntax` | The agent attempts to wire an electrical circuit but uses invalid syntax (e.g., `connect battery to switch`, `connect wire to battery anode` without specifying the wire's terminal endpoint) instead of the required form `connect <wire> terminal N to <component> anode/cathode` | The step where the malformed wiring command is issued |
| `operation/wrong_mix_container` | In a `chemistry-mix` task, the agent places ingredients in a non-mixing container (bowl, glass cup, jar) and attempts `mix <container>` — only `mix metal pot` produces a result | The step where the mix command on the wrong container is issued |
| `operation/wrong_object_identified` | The agent picks up and focuses on an object that is not the required target: (a) a non-living object (potato, orange) for `find-living-thing`; (b) a substance of the wrong type for `melt`/`boil`/`chemistry-mix` | The step where the agent picks up or focuses on the wrong object |
| `operation/placement_syntax_error` | The agent uses invalid placement syntax (`put down X in Y`, `put X in orange`) when the correct form is `move X to Y` or `put X in Y` (without "down"; full container name required) | The step where the invalid placement command is issued |

**Note**: `strategy/missing_experimental_step` and `strategy/wrong_measurement_method` are mutually exclusive — if the agent uses entirely the wrong method (e.g., stopwatch timing), use `wrong_measurement_method`; if the agent uses the right method but omits a specific sub-step (e.g., forgets to `focus on` at the end), use `missing_experimental_step`.

---

## Core vs. Marginal

**Core failure** (at most 3 per episode):
- Directly caused the agent to stop making meaningful progress
- Fixing it alone would likely unlock significant further progress
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
    "location": "Brief description of the room and experimental context where this failure occurred",
    "why": "Root cause explanation: what the agent did wrong, and what it should have done instead",
    "where": [first_step, last_step],
    "note": "(optional) required if type is a newly defined mode — provide a one-sentence definition"
  }
]
```

- `failure_id`: short descriptive identifier, e.g. `"look_loop_kitchen"`, `"ambig_no_digit"`, `"mix_bowl_not_pot"`
- `location`: describe in terms of room and experimental state, e.g. `"kitchen, marshmallow in cupboard not yet retrieved"`, `"workshop, attempting circuit wiring with wrong syntax"`
- `why`: explain both the mistake and the correct action, referencing specific observations or actions from the trajectory
- `where`: 0-indexed step indices where this failure manifests in the trajectory

If no failures are detected, return an empty array `[]`.

---

## Notes

- **The most common failure** (55% of all episodes, 95% of Qwen3-32B) is `strategy/look_around_loop`. When annotating, check early whether `look around` dominates the action distribution before looking for other failures.
- **Disambiguation failures** are common and subtle: check every step following an `"Ambiguous request"` observation — the correct response is only a single digit.
- **Invalid actions are not always failures**: a single `Unknown action` response followed by a corrected command is not worth annotating. Only flag when the agent repeats invalid actions without learning from feedback.
- **`reset task` is a valid action** used once to restart goal tracking; annotate only when 5+ consecutive resets occur with no intervening productive actions.
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
