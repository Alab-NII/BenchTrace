Here is the trajectory of an agent on ScienceWorld task **{{game}}** (task type: {{task_type}}). The agent achieved a final score of **{{final_score}} / {{max_score}}**.

<trajectory>
{{trajectory}}
</trajectory>

Identify up to 3 step ranges where improving the agent's behavior would most effectively boost its performance. Focus on the most impactful issues (e.g., looping on `look around`, getting stuck in the wrong room, omitting a required experimental sub-step like `mix metal pot` or `focus on`, malformed wiring/placement syntax, picking up the wrong object).

Respond with a JSON array only, no other text. Each element must have "step_start" and "step_end" (inclusive step numbers). Example:
[{"step_start": 2, "step_end": 50}, {"step_start": 70, "step_end": 100}]
