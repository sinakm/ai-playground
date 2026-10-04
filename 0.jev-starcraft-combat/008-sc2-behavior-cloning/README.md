# 008 — Behavior cloning over Jev perception

Episode 007 showed that the perception layer is useful but terminal-reward REINFORCE from
random weights is a poor way to learn the control policy. After 40 fights the weights had moved
only about 2%, the policy stayed close to uniform, and argmax collapsed mostly to `bait`.

Episode 008 changes the learning problem:

    strong commander teacher
             │
             ├── controls StarCraft
             │
    same state ──> frozen Jev perception
                      │
             matched semantic features
                      │
                      + teacher executed action
                      ↓
               behavior cloning
                      ↓
             small numerical policy

The episode-007 weights remain preserved in the 007 folder. They are **not** used by default
here because they are nearly random and biased toward `bait`. You can explicitly pass an
episode-008-compatible initialization to `arena clone --init-weights ...`, but the normal
experiment starts fresh.

## Architecture

Jev makes one batched perception call per decision window:

- 6 global Noul activations:
  - baneling_pressure
  - clumping_danger
  - encirclement_risk
  - focus_fire_opportunity
  - retreat_pressure
  - formation_instability
- 4 local Noul activations for every living Marine:
  - personal_danger
  - isolation
  - escape_pressure
  - firing_opportunity
- 1 optional referential Choice:
  - which candidate Baneling is the most immediate shared threat?

For Marine i the trainable input is still 10 numbers:

    x_i = [6 global activations, 4 local activations]

The action network is:

    10 -> 16 tanh -> 8 action logits -> softmax

The 8 mutually-exclusive Marine actions are:

    kite, split, attack, retreat, focus_bane, cover_ally, bait, stutter

Stim is now a **separate global binary head** over the 6 global activations. This matters
because the successful commander can stim and still issue a movement/fire action in the same
decision window; a single 9-way action head could not faithfully imitate that behavior.

The referential Baneling target is also kept separate from the scalar vector. If the learned
policy chooses `focus_bane`, the Jev perception target is passed to the existing targeting code.

## Step 1 — collect matched teacher traces

`jev_teacher_collect` uses the successful commander + stutter policy to control the fight while
the new Jev perception layer observes the **same battlefield state**. The commander behavior is
unchanged; perception is passive and only adds logging.

Use paused runs for collection:

    uv sync
    uv run pytest

    uv run arena run \
      --policy jev_teacher_collect \
      --runs 50 \
      --seed-base 2000

Every decision log now contains:

- global semantic activations
- local activations per Marine
- Jev's perceived priority Baneling
- commander's actual priority Baneling
- commander's raw Marine choices
- executed Marine actions after plan/reflex rules
- commander stim decision

This makes the training pairs matched rather than trying to align unrelated trajectories.

## Step 2 — behavior clone offline

No StarCraft and no Jev calls are needed for this step:

    uv run arena clone

Defaults:

- fresh MLP initialization
- 150 epochs
- learning rate 0.03
- 80/20 split by **whole fight**, not random Marine rows
- mild inverse-frequency weighting (power 0.5) so rare but important actions such as
  `split` and `focus_bane` are not ignored

Outputs:

    models/semantic_policy_bc.json
    models/clone_metrics.json

The metrics include training/validation action accuracy, stim accuracy, teacher win rate, and
agreement between Jev's perceived priority target and the commander's chosen target.

The held-out split is by battle to avoid leaking adjacent states from the same trajectory into
both train and validation sets.

## Step 3 — evaluate the cloned policy

Deterministic argmax:

    uv run arena run \
      --policy jev_trainable_semantic \
      --weights models/semantic_policy_bc.json \
      --runs 20 \
      --seed-base 0

Also test stochastic evaluation, because a cloned policy can represent a multi-modal teacher
distribution that argmax collapses:

    uv run arena run \
      --policy jev_trainable_semantic \
      --weights models/semantic_policy_bc.json \
      --sample-eval \
      --temperature 0.7 \
      --runs 20 \
      --seed-base 100

Then:

    uv run arena evaluate

## Step 4 — only then consider RL fine-tuning

If behavior cloning produces a competent policy, copy the cloned checkpoint and fine-tune that
copy with `--train`. Do not ask RL to discover competent behavior from random weights again.

## What this experiment answers

The key question is no longer whether a random tiny net can solve StarCraft from one terminal
reward. It is:

> Can a few hundred learned numerical weights reproduce a strong hierarchical commander when
> their only observation is Jev's compact semantic perception?

A positive result would show that a variable-sized battlefield can be compressed by Jev into a
small semantic representation from which a conventional trainable network can recover useful
coordinated behavior.
