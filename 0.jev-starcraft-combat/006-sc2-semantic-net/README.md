# 006 — Jev semantic net

Episode 006 tests Jev as a semantic hidden layer rather than as a commander.

Episode 005 used:

    battlefield -> discrete commander plan/target -> per-Marine actions

Episode 006 uses:

    battlefield -> semantic probabilities -> per-Marine actions

The arena, blackboard, reflexes, confidence gating, stutter primitive, timing, and soldier
action pool stay inherited from episode 005. The semantic policy removes the discrete
squad plan and Jev-selected priority target.

The first Jev call emits six independent Noul activations in [0,1]:
baneling_pressure, clumping_danger, encirclement_risk, focus_fire_opportunity,
retreat_pressure, and formation_instability.

The second call sees those shared activations plus the original battlefield/local Marine
state and independently chooses an action for every living Marine. Activations are
perceptions, not commands.

## Hypothesis

A multidimensional probabilistic semantic bottleneck can preserve useful shared battlefield
information better than one discrete commander plan while still allowing Marines to react
differently to the same global situation.

This episode deliberately does not train Jev or learn the semantic nodes. First establish
whether the architecture is useful.

## Policies

- attack_move
- random
- stutter_all
- jev_commander
- jev_commander_stutter
- jev_semantic_net

The key comparison is jev_commander_stutter vs jev_semantic_net. They share the same arena, micro primitives, blackboard, reflexes and confidence gating. The semantic policy exposes all nine Marine actions directly (including stim) and rewrites the two commander-dependent criteria (focus_bane and bait) in terms of perceptions/local state, because it has no squad plan or commander-owned stim decision.

## Run

    uv sync
    uv run pytest
    uv run arena run --policy jev_semantic_net --runs 20
    uv run arena evaluate

For a matched comparison, also run the other policies with the same run count/seed setup.
Each semantic decision is written to decisions.jsonl under semantic_activations so the
hidden layer can be inspected after the experiment.

## What to inspect beyond win rate

Look at Marines alive, survivor HP, enemies killed, fight duration, latency/cost, action
mix, and whether semantic activations vary meaningfully through the trajectory. A 20/20
result alone is not enough because episode 005 showed that unconditional stutter already
solves this scenario very well.
