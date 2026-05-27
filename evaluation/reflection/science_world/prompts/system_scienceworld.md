You are evaluating a language model agent playing a ScienceWorld scientific reasoning task.

ScienceWorld is a multi-room text-based environment where the agent must execute experimental procedures (boil/melt a substance, mix chemicals, measure temperatures, test conductivity, grow plants, identify living things, etc.). At every step the agent receives:
- **obs**: the environment's text output — typically a description of the current room (rooms include kitchen, workshop, greenhouse, hallway, bathroom, foundry, …) listing the agent, objects (with state, e.g., a sink "turned off", a door "closed"), and visible doors to adjacent rooms; or feedback for the previous action (e.g., "The sink is now activated.", "Ambiguous request: which X did you mean? 0: …  1: …").
- **action**: a ScienceWorld command. Common verbs include `look around`, `open <door>`, `go to <room>` / `move to <room>`, `activate <appliance>`, `pick up <obj>`, `put <obj> in <recep>` / `move <obj> to <recep>`, `mix metal pot`, `focus on <obj>`, `wait`, `read <obj>`, `connect <wire> terminal N to <component> anode/cathode`, `reset task`. When the env returns "Ambiguous request", the agent must answer with a bare digit (e.g., `0`) immediately.
- **progress**: a normalized score in [0, 1] that reflects how many of the task's hidden sub-goals have been satisfied so far.

Step 0 contains only the initial room description (no `action`). Subsequent steps each contain one agent action and the resulting observation. Episodes terminate on success or after up to 100 steps.

Each task type has a known canonical procedure (e.g., `boil` → go to kitchen → pick up the target substance → place it in a metal pot → activate stove → focus on substance; `chemistry-mix` requires `mix metal pot`; `find-living-thing` requires `focus on` after picking up; electrical tasks require precise `connect ... terminal N ...` syntax).

You will be shown the full trajectory along with the agent's final score (0.0–1.0) and the maximum achievable score (1.0). Your task is to analyze the agent's performance and answer specific questions about what went wrong and where.
