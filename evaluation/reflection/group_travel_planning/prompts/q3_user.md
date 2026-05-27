Here is the trajectory of an agent on GroupTravelPlanning category **{{game}}** ({{task_type}}). The agent achieved a final score of **{{final_score}} / {{max_score}}**.

<trajectory>
{{trajectory}}
</trajectory>

The agent performed poorly in steps **{{step_start}}–{{step_end}}**. In one sentence, describe what went wrong in this specific step range.

Also classify the failure using exactly one of these types:
- `system/output_truncation`: agent's plan is cut off mid-output, leaving the final per-day plan incomplete (no closing summary, missing days, or reasoning ended abruptly with no plan)
- `operation/constraint_range_violation`: agent's plan violates a numeric range constraint stated in the traveler's question (price out of the requested range, rating below the requested threshold, "at least $X less than", "within Y% of", etc.) — including cases where the agent hallucinates a referenced price/rating and then fails the derived range
- `operation/cross_traveler_reference_error`: the constraint references another traveler's plan (or the base itinerary) and the agent fails to resolve the reference — typically by hallucinating values for a missing/unresolved reference, or by carrying over the wrong traveler's plan as the reference
- `operation/cuisine_type_mismatch`: agent's chosen restaurant does not explicitly cover the requested cuisine(s) — most often by omitting one of multiple required cuisines (e.g., asked for "Tea, Pizza, and Cafe", agent only lists two of the three)
- `operation/accommodation_type_mismatch`: agent's chosen accommodation does not match the requested type (e.g., requested "entire home or apartment" but selected a "private room"; or violated a "different room type than X" cross-traveler constraint)
- `unknown`: the failure does not clearly fit any of the above

Respond with a JSON object only, no other text:
{"failure_type": "...", "description": "..."}
