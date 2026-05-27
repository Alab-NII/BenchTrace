Here is the trajectory of an agent playing **{{game}}**. The agent achieved a final score of **{{final_score}} / {{max_score}}** points.

<trajectory>
{{trajectory}}
</trajectory>

Identify up to 3 step ranges where improving the agent's behavior would most effectively boost its performance. Focus on the most impactful issues.

Respond with a JSON array only, no other text. Each element must have "step_start" and "step_end" (inclusive step numbers). Example:
[{"step_start": 10, "step_end": 25}, {"step_start": 60, "step_end": 70}]
