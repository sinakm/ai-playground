# 007 — Trainable Jev semantic policy

Episode 007 makes the numerical part real.

Jev is frozen and used only as perception. One batched Jev call emits six global semantic
activations plus four local activations for each living Marine. Jev does not choose the
Marine action.

For Marine i:

    x_i = [6 global Jev activations, 4 local Jev activations]

A tiny local network then chooses the action:

    10 inputs -> 16 tanh hidden units -> 9 action logits -> softmax

The same network weights are shared by every Marine. This is intentional: a Marine's
behavior differs because its local semantic activations differ, not because it owns a
separate model.

Global features:
- baneling_pressure
- clumping_danger
- encirclement_risk
- focus_fire_opportunity
- retreat_pressure
- formation_instability

Local features:
- personal_danger
- isolation
- escape_pressure
- firing_opportunity

The nine outputs are the existing Marine actions: kite, split, attack, stim, retreat,
focus_bane, cover_ally, bait, stutter.

## Training

The first implementation uses simple episodic REINFORCE. During training the policy samples
from its softmax. At the end of each fight it receives:

    reward =
        enemies_killed if at least one Marine survives else 0
        + 0.25 * Marines_alive
        + 0.01 * surviving_HP

The original score remains dominant; survival/HP only separates successful policies that
otherwise all kill 20 enemies.

Weights persist between fights in models/semantic_policy.json. Jev itself is never updated.

Start with a small training run:

    uv sync
    uv run pytest
    uv run arena run --policy jev_trainable_semantic --runs 20 --train --seed-base 1000

If the mechanics look sane, continue to roughly 100-200 training fights in batches. The
same weights file is loaded and updated after every battle:

    uv run arena run --policy jev_trainable_semantic --runs 20 --train --seed-base 1020

For evaluation, omit --train. That freezes the saved weights and uses argmax actions:

    uv run arena run --policy jev_trainable_semantic --runs 20 --seed-base 0
    uv run arena evaluate

Use a different file with --weights if you want independent training replicates.

## What this tests

006 asked whether shared semantic activations were useful prompt context for another Jev
decision. 007 asks a cleaner question:

    variable battlefield
        -> frozen Jev semantic sensors
        -> fixed 10-D vector per Marine
        -> trainable numerical weights
        -> action

This is deliberately a tiny model. If it cannot learn a useful policy, the first suspects
should be representation, reward, exploration, or credit assignment—not insufficient model
capacity.

## Important caveats

This first trainer uses one terminal reward for all Marine decisions in a battle. Credit
assignment is therefore crude. It is a proof-of-learning experiment, not a claim that
REINFORCE is the best optimizer.

The scenario is also still easy for scripted stutter_all. Compare not only win rate but
Marines alive, survivor HP, fight time, action distribution, and learning across training
checkpoints.
