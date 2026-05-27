Here is the trajectory of an agent on GroupTravelPlanning task **{{game}}** ({{task_type}}). The agent achieved a final score of **{{final_score}} / {{max_score}}**.

<trajectory>
{{trajectory}}
</trajectory>

Identify up to 3 step ranges where improving the agent's behavior would most effectively boost its performance. Focus on the most impactful issues (e.g., a traveler whose explicit constraints were ignored, a JOIN/RELATION constraint to the base itinerary that was missed, repeated identical plans across travelers, or plans that hallucinate activities not in the itinerary).

Each step is annotated with `subtask=<idx>`; failures are usually localized to specific traveler boundaries. Note that even-indexed steps contain the question (no agent plan) and odd-indexed steps contain the agent's plan.

Respond with a JSON array only, no other text. Each element must have "step_start" and "step_end" (inclusive step numbers). Example:
[{"step_start": 1, "step_end": 1}, {"step_start": 5, "step_end": 7}]
