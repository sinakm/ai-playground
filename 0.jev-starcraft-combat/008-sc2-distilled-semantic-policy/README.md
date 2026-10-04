# 008 — Distilled Jev semantic policy

Episode 008 replaces the weak terminal-reward RL experiment from 007 with teacher distillation.

The successful `jev_commander_stutter` policy controls the battle. At the same decision step,
a frozen Jev perception call observes the identical state. After code reflexes are applied,
the episode stores:

    Jev global semantics
    + Jev local semantics for each Marine
    + action that actually executed
    + teacher squad-stim decision

That gives aligned supervised examples instead of asking one terminal reward to explain
hundreds of Marine decisions.

## Architecture

Jev is still not trained.

One batched perception call produces:

- 7 global scalar perceptions:
  - baneling_pressure
  - clumping_danger
  - encirclement_risk
  - focus_fire_opportunity
  - retreat_pressure
  - formation_instability
  - stim_opportunity
- 4 local scalar perceptions per living Marine:
  - personal_danger
  - isolation
  - escape_pressure
  - firing_opportunity
- one referential perception: the Baneling currently posing the strongest threat

For Marine i:

    x_i = [7 global activations, 4 local activations]

The numerical student is:

    11 inputs -> 16 tanh hidden units -> 8 action logits -> softmax

Stim is deliberately not one of those eight actions. The commander can stim and still issue
a movement/combat order in the same decision window, so episode 008 gives stim its own learned
global head:

    7 global inputs -> sigmoid(stim_now)

The eight per-Marine actions are:

    kite, split, attack, retreat, focus_bane, cover_ally, bait, stutter

When the student chooses `focus_bane`, it uses Jev's referential priority-Baneling perception.
This fixes the entity-identity loss exposed in episode 006.

## Why not continue the 007 weights?

The latest 007 checkpoint is preserved in episode 007. After 40 fights its action probabilities
were still nearly uniform and the argmax policy mostly collapsed to `bait`. The architecture
also mixed squad stim into a mutually exclusive per-Marine action head. Episode 008 intentionally
starts a corrected supervised student rather than treating those weak RL weights as useful
initialization.

## Results so far (round 8a)

50 teacher fights (49 wins) -> 8,860 samples -> student validation accuracy 0.60 (majority 0.32;
focus_bane and cover_ally recall 0, stim head no better than "never stim"). Student without the
commander, seeds 0-19, T=1.0: **16/20 wins**, 4.7 Marines alive, 156 ms median, $0.13 for 20
fights. Same win rate as 006, below the teacher. Details and next steps:
[`../notes/2026-10-03-ep008-findings.md`](../notes/2026-10-03-ep008-findings.md).

## 1. Collect teacher demonstrations

Delete the old dataset first if you want a clean replicate:

    rm data/teacher.jsonl

Then run the successful commander while the semantic Jev call observes:

    uv sync
    uv run pytest
    uv run arena run \
      --policy jev_teacher_collect \
      --runs 50 \
      --seed-base 2000 \
      --dataset data/teacher.jsonl

The teacher still owns the battle. The perception call is only an observer. Each JSONL row is
one decision step and stores the executed labels after reflexes, not merely the commander's raw
choice.

Start with 50 fights. If class coverage is weak, extend to 100.

## 2. Train locally

No StarCraft and no Jev calls are needed for this step:

    uv run arena train-bc \
      --dataset data/teacher.jsonl \
      --weights models/distilled_policy.json \
      --epochs 120

The trainer splits by whole episode seed, not random decision rows, so validation trajectories
are held out. It reports action accuracy/loss, stim accuracy/loss, class counts, sample count,
and agreement between Jev's referential Baneling perception and the commander's chosen target.

## 3. Evaluate without the commander

The student now sees only Jev perception plus its learned numerical weights:

    uv run arena run \
      --policy jev_distilled_semantic \
      --runs 20 \
      --seed-base 0 \
      --weights models/distilled_policy.json \
      --temperature 1.0

Then:

    uv run arena evaluate

Temperature 1.0 is the primary evaluation because the teacher itself is not a deterministic
argmax classifier and episode 007 showed that near-tied logits make argmax misleading.
`--temperature 0` is useful as a diagnostic, but should not be the only reported result.

## What would count as success?

The strongest result is not beating StarCraft with a large model. It is compression:

    variable battlefield
        -> one frozen Jev perception call
        -> 11 semantic values per Marine + one referential target
        -> a few hundred learned numerical parameters
        -> coordinated behavior

If the distilled student approaches the 005 commander's 20/20 performance, then the commander
has effectively taught a very small numerical control network how to act over Jev semantic
perception.

Only after that should we reintroduce RL/self-play as fine-tuning rather than asking RL to
discover competent behavior from random weights.
