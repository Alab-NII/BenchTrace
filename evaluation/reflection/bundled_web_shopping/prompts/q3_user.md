Here is the trajectory of an agent on BundledWebShopping category **{{game}}** ({{task_type}}). The agent achieved a final score of **{{final_score}} / {{max_score}}**.

<trajectory>
{{trajectory}}
</trajectory>

The agent performed poorly in steps **{{step_start}}–{{step_end}}**. In one sentence, describe what went wrong in this specific step range.

Also classify the failure using exactly one of these types:
- `system/output_truncation`: agent's action is cut off mid-reasoning (e.g., `<think>` block without closing `</think>`, reasoning that ends abruptly) with no `Selection: [X]` produced — caused by hitting a token limit before generating a final answer
- `system/format`: agent's action does not contain any recognizable letter selection (free text without `[X]`, outputs `[✗]` / `[None]`, or outputs a number instead of a letter)
- `strategy/wrong_cascade`: agent correctly identifies the goal of the current subtask but applies the compatibility rules using an incorrect type label for a previously selected item — leading to a wrong cascade assumption that propagates through subsequent subtasks
- `strategy/cascade_propagation`: starting from a wrong selection in an earlier subtask, the agent consistently builds subsequent reasoning on that wrong premise — the agent's logic is internally consistent but grounded in an incorrect earlier choice
- `strategy/goal_preference_conflict`: agent explicitly prioritizes the subtask-level **Preference** over the **Goal** when they conflict (e.g., chooses highest-priced when the Goal says lowest-priced, citing the Preference as justification)
- `operation/unobservable_ranking`: agent must select the highest/lowest-priced or highest-rated option but prices/ratings are not shown — the agent makes an incorrect inference from brand name or product description and picks the wrong item
- `operation/type_misidentification`: agent misidentifies the compatibility type of a previously selected item (e.g., labels a "hydrating cleansing bar" as "gel") when applying cascade rules for the current subtask
- `operation/semantic_refusal`: agent decides that no available option fits the subtask label (e.g., "none of these are actual frostings") and refuses to select any option, even though the task's semantic mapping is broader than the literal label
- `operation/within_set_ranking_error`: agent correctly identifies the compatible subset (right type) but picks the wrong item within that subset due to an incorrect price/rating ranking — compatibility reasoning is correct but the final letter choice is wrong
- `unknown`: the failure does not clearly fit any of the above

Respond with a JSON object only, no other text:
{"failure_type": "...", "description": "..."}
