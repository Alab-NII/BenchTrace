Here is the trajectory of an agent on BundledWebShopping category **{{game}}** ({{task_type}}). The agent achieved a final score of **{{final_score}} / {{max_score}}**.

<trajectory>
{{trajectory}}
</trajectory>

Identify up to 3 step ranges where improving the agent's behavior would most effectively boost its performance. Focus on the most impactful issues (e.g., output truncation that prevented producing `Selection: [X]`, an unobservable-attribute ranking error in an early subtask that cascades through later subtasks, type misidentification that breaks compatibility, or refusing to select any option).

Each step in the trajectory is annotated with `subtask=<idx>`; failures are usually localized to specific subtask boundaries.

Respond with a JSON array only, no other text. Each element must have "step_start" and "step_end" (inclusive step numbers). Example:
[{"step_start": 1, "step_end": 1}, {"step_start": 5, "step_end": 11}]
