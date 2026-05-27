Here is the trajectory of an agent on ScienceWorld task **{{game}}** (task type: {{task_type}}). The agent achieved a final score of **{{final_score}} / {{max_score}}**.

<trajectory>
{{trajectory}}
</trajectory>

The agent performed poorly in steps **{{step_start}}–{{step_end}}**. In one sentence, describe what went wrong in this specific step range.

Also classify the failure using exactly one of these types:
- `system/format`: agent outputs a malformed action that the environment cannot parse (empty string, multi-line text, partial commands truncated mid-word)
- `strategy/look_around_loop`: agent repeatedly issues `look around` as its primary action across many consecutive steps, making no experimental progress (a planning breakdown rather than a reaction to a specific failure)
- `strategy/wrong_room_deadlock`: agent spends most of the episode in a room irrelevant to the task (e.g., trying to wire a circuit in the kitchen, or searching for living things in the workshop), never navigating to the task-relevant room
- `strategy/reset_overuse`: agent issues `reset task` 5+ times in succession without trying any new approach in between
- `strategy/missing_experimental_step`: agent's plan entirely omits a required sub-step specific to the task type (applying fertilizer in `grow-plant`, placing the thermometer on the heat source in `measure-melting-point`, calling `focus on` after placement in `find-living-thing`, issuing `mix metal pot` in `chemistry-mix`)
- `strategy/wrong_measurement_method`: agent uses an incorrect measurement approach (e.g., using the stopwatch to time block sliding instead of reading the inclined-plane percentage; measuring ambient temperature instead of moving the thermometer to the heat source)
- `operation/disambiguation_failure`: when the env returns "Ambiguous request", the agent does not respond with a bare digit on the very next action (responds with free text, issues `look around` first, or sends the digit too late)
- `operation/electrical_wiring_syntax`: agent attempts to wire a circuit but uses invalid syntax (e.g., `connect battery to switch`) instead of the required `connect <wire> terminal N to <component> anode/cathode`
- `operation/wrong_mix_container`: in `chemistry-mix`, agent places ingredients in a non-mixing container (bowl, glass cup, jar) and tries `mix <container>` — only `mix metal pot` produces a result
- `operation/wrong_object_identified`: agent picks up / focuses on an object that is not the required target (e.g., a non-living object for `find-living-thing`; a substance of the wrong type for `melt`/`boil`/`chemistry-mix`)
- `operation/placement_syntax_error`: agent uses invalid placement syntax (`put down X in Y`, `put X in orange`) instead of `move X to Y` or `put X in Y` with the full container name
- `unknown`: the failure does not clearly fit any of the above

Respond with a JSON object only, no other text:
{"failure_type": "...", "description": "..."}
