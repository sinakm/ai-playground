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

## Results (round 6a)

Scenario (inherited from 005): 12 Marines vs 10 Banelings + 10 Zerglings (20 Zerg),
20 paused fights per policy (seeds 0-19), 60 s fight cap, decisions every 12 game
loops (0.54 s). Score = Zerg killed if any Marine survives, else 0. `attack_move`,
`random` and `jev_commander` were not re-run in this round.

| Policy | Runs | Wins | Score | Marines alive | Survivor HP | Enemies killed | Fight s | Median ms | p90 ms | Cost USD |
|---|---|---|---|---|---|---|---|---|---|---|
| stutter_all | 20 | 20 | 20 | 8.85 | 51.30 | 20 | 11.39 | - | - | 0.00 |
| jev_commander_stutter | 20 | 20 | 20 | 7.85 | 33.70 | 20 | 10.00 | 361.90 | 456.00 | 0.23 |
| jev_semantic_net | 20 | 16 | 16 | 5.25 | 46.60 | 19.05 | 14.89 | 366.10 | 454.80 | 0.28 |

The hypothesis did not hold in this scenario: removing the commander plan and
feeding Marines six shared perceptions instead lost 4 of 20 fights (seeds 0, 1,
9, 16; all Marines dead after 13-17 kills), kept fewer Marines alive and took
longer than both the commander and the no-AI `stutter_all` baseline.

## What the semantic layer did

Activations were not constant (565 decisions):

| Perception | p10 | median | p90 | std |
|---|---|---|---|---|
| baneling_pressure | 0.03 | 0.63 | 0.95 | 0.39 |
| clumping_danger | 0.04 | 0.48 | 0.85 | 0.32 |
| encirclement_risk | 0.20 | 0.46 | 0.73 | 0.20 |
| focus_fire_opportunity | 0.03 | 0.61 | 0.74 | 0.29 |
| retreat_pressure | 0.43 | 0.82 | 0.89 | 0.18 |
| formation_instability | 0.21 | 0.66 | 0.86 | 0.23 |

`retreat_pressure` stayed high most of the fight and carried little signal.

Marine actions, share of all Marine-steps:

| | Jev raw choice (semantic) | Executed (semantic) | Jev raw choice (commander) | Executed (commander) |
|---|---|---|---|---|
| attack | 55.3% | 36.7% | 39.9% | 35.5% |
| kite | 20.3% | 13.8% | 8.0% | 8.4% |
| stutter | 7.6% | 35.2% | 3.7% | 19.8% |
| split | 5.7% | 0.2% | 6.0% | 13.8% |
| focus_bane | 2.8% | 0% | 29.9% | 10.8% |
| retreat | 5.0% | 9.0% | 4.8% | 7.7% |
| cover_ally | 3.4% | 0.8% | 7.6% | 2.8% |
| low-confidence Marines | 55.3% | | 40.2% | |

Without a commander, the pre-contact split and the commander's focus-fire target
disappeared (executed split 0.2% vs 13.8%, focus_bane 0% vs 10.8%). More than half
of the Marine answers fell under the 0.5 confidence gate and were replaced by the
code default, which is why executed stutter (35%) is far above Jev's own stutter
choice (7.6%).

## Cost and latency

Two sequential calls per decision (6 Noul perceptions, then up to 12 Marine
Choices): median 366.1 ms, p90 454.8 ms, inside the 535.7 ms budget, 0 late
decisions, 0 API errors. 20 fights: 6,664,049 input tokens, $0.28.

## Video

`runs/jev_semantic_net-20261003-171522-s100/showcase.mp4` (realtime take 1 of 3,
win, 6 Marines alive, 14.1 s). Not uploaded yet. Realtime takes are not
reproducible from paused seeds; of three takes, two won and one lost. Takes 2 and 3
have no audio (`soundcard` `Error 0x800401f0` on the second recording in one
process).

## Limits

- 20 fights per policy, one scenario. `stutter_all` already wins 20/20 here, so the
  scenario can't show a gain from better judgment, only losses.
- The comparison changes two things at once: the information layer (perceptions vs
  plan) and the actions that depended on the plan (pre-split, commander target).
- The showcase title card still says "one squad decision every 0.54 s"; the
  semantic net makes two calls per decision.

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
