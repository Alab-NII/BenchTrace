Here is the trajectory of an agent playing BabyAI level **{{game}}** (task type: {{task_type}}). The agent achieved a final score of **{{final_score}} / {{max_score}}**.

<trajectory>
{{trajectory}}
</trajectory>

The agent performed poorly in steps **{{step_start}}–{{step_end}}**. In one sentence, describe what went wrong in this specific step range.

Also classify the failure using exactly one of these types:
- `system/format`: agent outputs a malformed action (empty, multi-line, JSON/code leakage, or a string that is not a valid BabyAI action)
- `strategy/wall_running`: agent repeatedly executes `move forward` despite the observation remaining unchanged across multiple consecutive steps (blocked against a wall) and never attempts to turn
- `strategy/lateral_target_ignore`: the target is persistently visible with a non-zero lateral offset (`<M> steps to your right/left`), but the agent keeps moving forward without turning to align — the offset does not decrease over many steps
- `strategy/loop`: the agent cycles through a repeating sequence of (facing direction, action) pairs (e.g., alternating `turn left` / `turn right`, or repeatedly taking the same turn-and-move circuit) without reducing its distance to the target
- `strategy/room_boundary_deadlock`: in a multi-room task, the target is in a different room, but the agent never exits the starting room (never successfully traverses a door)
- `strategy/wrong_object_interaction`: agent commits to interacting with the wrong object (wrong color/type) for multiple steps, e.g., navigates to and picks up the wrong-colored ball or opens the wrong door
- `operation/pickup_misalignment`: agent issues `pick up` but the target object is not exactly 1 step directly in front (lateral offset is non-zero, or distance > 1) — pickup fails silently
- `operation/toggle_misalignment`: agent issues `toggle` but the target door/switch is not exactly 1 step directly in front — toggle fails silently
- `operation/wrong_action_type`: agent uses the wrong action category for the goal (e.g., toggling a door for a `GoTo` task whose win condition is mere adjacency; issuing `done` before completion; issuing `pick up` on a navigation-only task)
- `operation/wrong_target_attribute`: agent navigates to or attempts to interact with an object of the wrong color or type (e.g., approaching a blue door when the task requires the red door)
- `unknown`: the failure does not clearly fit any of the above

Respond with a JSON object only, no other text:
{"failure_type": "...", "description": "..."}
