Here is the trajectory of an agent on AlfWorld task type **{{game}}** ({{task_type}}). The agent achieved a final score of **{{final_score}} / {{max_score}}**.

<trajectory>
{{trajectory}}
</trajectory>

Identify up to 3 step ranges where improving the agent's behavior would most effectively boost its performance. Focus on the most impactful issues (e.g., looping in already-searched receptacles, failing to navigate to the required destination, missing a required heat/cool/clean step, picking the wrong object).

Respond with a JSON array only, no other text. Each element must have "step_start" and "step_end" (inclusive step numbers). Example:
[{"step_start": 5, "step_end": 18}, {"step_start": 30, "step_end": 50}]
