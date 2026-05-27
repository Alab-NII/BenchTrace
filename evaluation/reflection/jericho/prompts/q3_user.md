Here is the trajectory of an agent playing **{{game}}**. The agent achieved a final score of **{{final_score}} / {{max_score}}** points.

<trajectory>
{{trajectory}}
</trajectory>

The agent performed poorly in steps **{{step_start}}–{{step_end}}**. In one sentence, describe what went wrong in this specific step range.

Also classify the failure using exactly one of these types:
- `system/format`: agent issued syntactically malformed or unrecognized commands
- `strategy/loop`: agent repeated the same (location, action) pairs without progress
- `strategy/route_inefficiency`: agent took an unnecessarily long or redundant path
- `strategy/unexplored_direction`: agent failed to explore an available direction or exit
- `operation/feedback_blindness`: agent ignored explicit negative feedback and repeated the same ineffective action
- `operation/perception_error`: agent failed to notice or correctly interpret a key object or piece of information
- `operation/decision_error`: agent made a logically incorrect decision given available information
- `unknown`: the failure does not clearly fit any of the above

Respond with a JSON object only, no other text:
{"failure_type": "...", "description": "..."}
