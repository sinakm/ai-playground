# 009 — Relational Jev semantic policy

Episode 009 tests the hypothesis exposed by 008: the student was not mainly limited by
network capacity; it was missing semantic relationships.

In 008, the student learned movement actions well but had zero held-out recall for
`focus_bane` and `cover_ally`. It knew properties of the battlefield and each Marine, and
Jev identified the commander's priority Baneling 97% of the time, but the input did not say how
a particular Marine related to that target or to a threatened teammate.

009 keeps Jev frozen and adds semantic **edge features**.

## Architecture

One batched Jev perception call produces:

### 7 global node features
- baneling_pressure
- clumping_danger
- encirclement_risk
- focus_fire_opportunity
- retreat_pressure
- formation_instability
- stim_opportunity

### 4 local Marine node features
- personal_danger
- isolation
- escape_pressure
- firing_opportunity

### 5 relational / edge features per Marine
- priority_target_shootable
- priority_target_threatens_ally
- should_focus_priority_target
- ally_needs_cover
- can_cover_ally

### 1 referential output
- the Baneling Jev considers the current highest-priority threat

For Marine i:

    x_i = [
        7 global node activations,
        4 local node activations,
        5 relational edge activations
    ]

The student remains deliberately tiny:

    16 inputs -> 16 tanh hidden units -> 8 action logits -> softmax

The same weights are shared by every Marine.

Stim remains a separate learned head because Stim can happen in the same window as another
Marine action:

    7 global inputs -> sigmoid(stim_now)

The eight mutually exclusive Marine actions are:

    kite, split, attack, retreat, focus_bane, cover_ally, bait, stutter

When the student predicts `focus_bane`, the action executor uses Jev's referential priority
target.

## Why this is a cleaner test

008 already showed that a larger or more heavily trained generic MLP would not solve missing
information. Its action distribution looked similar to the teacher, but rare coordinated actions
occurred at the wrong times.

009 changes the representation rather than increasing model size.

It also uses class-balanced supervised learning:
- inverse-sqrt weighting for rare Marine actions;
- positive-class weighting for squad Stim;
- per-class recall and balanced accuracy in the training report.

The goal is specifically to test whether relational semantic features recover
`focus_bane` and `cover_ally` timing.

## 1. Collect relational teacher demonstrations

Start with a clean dataset:

    rm -f data/relational_teacher.jsonl

Then run the successful commander while the relational Jev perception layer observes the same
battlefield:

    uv sync
    uv run pytest

    uv run arena run \
      --policy jev_relational_teacher \
      --runs 50 \
      --seed-base 3000 \
      --dataset data/relational_teacher.jsonl

The commander still owns the battle. Labels are the actions that actually execute after code
reflexes.

## 2. Train the student locally

No StarCraft and no Jev calls are used here:

    uv run arena train-bc \
      --dataset data/relational_teacher.jsonl \
      --weights models/relational_policy.json \
      --epochs 120

The report includes:
- held-out action accuracy;
- balanced action accuracy;
- recall for every action;
- Stim positive/negative recall;
- class weights;
- Jev-vs-commander priority-target agreement.

The most important metrics for this episode are held-out recall for `focus_bane` and
`cover_ally`.

## 3. Evaluate without the commander

    uv run arena run \
      --policy jev_relational_student \
      --runs 20 \
      --seed-base 0 \
      --weights models/relational_policy.json \
      --temperature 1.0

    uv run arena evaluate

Also run an argmax diagnostic:

    uv run arena run \
      --policy jev_relational_student \
      --runs 10 \
      --seed-base 100 \
      --weights models/relational_policy.json \
      --temperature 0

Do not mix those diagnostic runs into the primary results table.

## Success criteria

The first gate is representational, before StarCraft win rate:

1. `focus_bane` held-out recall is materially above 0.
2. `cover_ally` held-out recall is materially above 0.
3. Stim positive recall improves over 008's effectively-never-stim head.
4. Priority-target agreement remains high.

Then evaluate control quality. The target is to move meaningfully above 008's 16/20 student
toward the commander's 20/20-level behavior while retaining the one-call semantic perception
architecture.

If this works, the useful abstraction is no longer just "Jev as sigmoid neurons." It is:

    semantic node activations
        +
    semantic edge activations
        +
    referential entity outputs
        ->
    tiny trainable numerical controller

That is much closer to a semantic graph neural network.
