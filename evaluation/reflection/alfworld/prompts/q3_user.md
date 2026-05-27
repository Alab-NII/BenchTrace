Here is the trajectory of an agent on AlfWorld task type **{{game}}** ({{task_type}}). The agent achieved a final score of **{{final_score}} / {{max_score}}**.

<trajectory>
{{trajectory}}
</trajectory>

The agent performed poorly in steps **{{step_start}}–{{step_end}}**. In one sentence, describe what went wrong in this specific step range.

Also classify the failure using exactly one of these types:
- `system/format`: agent issued a malformed action (empty, multi-line, JSON leakage, excessively long, or starting with a non-letter character)
- `strategy/loop`: within a no-progress segment, the agent repeated the same (location, action) pair
- `strategy/exhaustive_rescan`: agent repeatedly revisited and re-examined receptacles it had already fully searched, without committing to any object it had previously found
- `strategy/missing_destination`: agent was holding the correct object (or had completed a processing step) but never navigated to the required destination receptacle
- `strategy/missing_processing_step`: agent's plan entirely omitted a required transformation step (`heat`/`cool`/`clean`/`use desklamp`) — the agent never attempted the command even when holding the object near the correct appliance
- `strategy/goal_misinterpretation`: agent treated the task as a different task type from the start (e.g., executing pick-and-place behavior on a `look_at_obj_in_light` task)
- `operation/feedback_blindness`: agent repeated the same action after receiving an explicit negative or null response in the immediately preceding observation
- `operation/wrong_object_pick`: agent picked up an object that is semantically similar to but categorically different from the target (e.g., `spraybottle` instead of `perfume`, `cup` instead of `egg`)
- `operation/move_command_avoidance`: agent was at the correct destination with the target object in inventory, the `move <obj> to <receptacle>` command was admissible, but the agent issued a passive command (`look`, `examine`, `go to`) instead
- `unknown`: the failure does not clearly fit any of the above

Respond with a JSON object only, no other text:
{"failure_type": "...", "description": "..."}
