# Hybrid controller (gate) rules

A hybrid agent (`policy.kind: hybrid`) asks the language model for a tool
call and executes it unless one of the following holds; the heuristic policy
of the same role and adaptation level then chooses the action instead.

1. **Illegal**: the proposed tool is not among the legal hints derived from
   the current observation (`force_legal_hints: true`).
2. **Stall**: a stall counter is incremented when the proposal had to be
   repaired or replaced (parse failure, invalid parameters), when the model
   asks for reconnaissance for the third consecutive time while hosts are
   known, or when it asks to harvest credentials without a service foothold;
   it is decremented otherwise. The gate fires when the counter reaches
   `stall_limit` (1 in every reported campaign cell).
3. **Progress (attacker only)**: after 3 executed model steps without a
   service foothold or credentials, reconnaissance/wait/probe/harvest
   proposals are replaced; after 6 executed model steps without any
   foothold, every proposal is replaced.

If the Ollama backend fails after its retries, the heuristic acts for that
step and the step is counted as a gate intervention.

The first attacker action of each episode (when no host is known) is a
reconnaissance step issued without querying the model and counted as a
model step.

Per-episode counters (`hybrid_gate_stats` in each episode record):
`llm_steps`, `heuristic_steps`, `gate_illegal`, `gate_stall`,
`gate_progress`, `llm_repairs`. The gate fraction reported in the paper is
`heuristic_steps / (llm_steps + heuristic_steps)` for the attacker, pooled
over the episodes of a run.

## Attribution ablation (phase 2b)

- Model only: `policy.kind: llm` (no gate); invalid proposals are still
  repaired from the observation or replaced by the first legal hint.
- Heuristic only: `policy.kind: heuristic` for both roles.

The exact system prompts are in `repro/prompts/`.
