You are evaluating a language model agent playing an AlfWorld (TextWorld-ALFRED) household task.

In each step of the episode, the agent receives:
- **obs**: the environment's text output (description of the current location, objects, or feedback for the previous action)
- **action**: the command the agent issued in response (e.g., `go to <recep>`, `open <recep>`, `take <obj> from <recep>`, `move <obj> to <recep>`, `heat <obj> with <appliance>`, `cool <obj> with <appliance>`, `clean <obj> with <appliance>`, `use <obj>`, `look`)
- **progress**: a normalized completion score in [0, 1] reflecting how much of the task the agent has accomplished so far

Step 0 contains the initial room description and the task instruction in the form `Your task is to: <goal>` (e.g., `put some soapbottle on cabinet`, `heat some egg and put it in the diningtable`). The task instruction has no `action`. Subsequent steps each contain one agent action and the resulting observation.

Each episode terminates either when the agent succeeds (`final_score == max_score == 1.0`) or after up to 50 steps. The agent earns partial credit for sub-goals (e.g., reaching the right location, picking up the correct object, performing required transformations like heat/cool/clean).

You will be shown the full trajectory along with the agent's final score (0.0–1.0) and the maximum achievable score (1.0). Your task is to analyze the agent's performance and answer specific questions about what went wrong and where.
