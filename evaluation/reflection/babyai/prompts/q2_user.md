Here is the trajectory of an agent playing BabyAI level **{{game}}** (task type: {{task_type}}). The agent achieved a final score of **{{final_score}} / {{max_score}}**.

<trajectory>
{{trajectory}}
</trajectory>

Identify up to 3 step ranges where improving the agent's behavior would most effectively boost its performance. Focus on the most impactful issues (e.g., running into a wall, ignoring a target with a lateral offset, looping turns, picking up/toggling the wrong object, never exiting the starting room).

Respond with a JSON array only, no other text. Each element must have "step_start" and "step_end" (inclusive step numbers). Example:
[{"step_start": 2, "step_end": 12}, {"step_start": 30, "step_end": 64}]
