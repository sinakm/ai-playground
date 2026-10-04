# 009 — Jev relational semantic graph

Episode 009 targets the failure exposed by 008: the 11 scalar node features predicted movement
well, but they did not contain enough information to time coordinated actions such as
`focus_bane` and `cover_ally`.

The change is deliberately representational. Jev remains frozen and the numerical student remains
tiny. We add semantic **edges** between a Marine and the important entities around it.

## Architecture

Perception is now two batched Jev calls.

Call 1 — semantic nodes:

- 7 global activations
- 4 local activations per Marine
- one referential priority-Baneling choice

Call 2 — semantic edges:

The exact Baneling selected by call 1 is inserted back into the state as
`semantic_priority_target`. For every living Marine Jev evaluates:

- `target_engagement_value`
  - would this Marine materially help by attacking that exact target now?
- `target_threat_to_local_group`
  - is that target an immediate threat to this Marine or nearby teammates?
- `ally_needs_cover`
  - does one of this Marine's nearby teammates need covering fire?
- `cover_effectiveness`
  - can this Marine actually provide useful cover without creating disproportionate danger?

So Marine i now receives:

    7 global node features
    + 4 local node features
    + 4 relational edge features
    = 15 numeric inputs

The student remains:

    15 -> 16 tanh -> 8 action logits -> softmax

Stim is still a separate decision-level head:

    7 global inputs -> sigmoid(stim_now)

The action head remains shared by every Marine.

## Why two Jev calls?

The second call must reason about the exact referential Baneling chosen by the first call.
Trying to ask "which target?" and "should Marine 7 engage that target?" in one independent
question batch risks the edge question silently reasoning about a different Baneling.

The tradeoff is latency: 009 should be slower than 008's ~156 ms student, but it should still be
well below the old commander stack if both batched perception calls stay near the historical Jev
latency.

## Teacher data

Episode 008's committed dataset cannot be reused for the primary 009 experiment because it has no
relational activations. Collect a fresh aligned dataset:

    cd 0.jev-starcraft-combat/009-sc2-relational-semantic-graph
    uv sync
    uv run pytest

    uv run arena run \
      --policy jev_teacher_collect \
      --runs 50 \
      --seed-base 3000 \
      --dataset data/teacher_relational.jsonl

The successful commander still controls the fight. Jev node/edge perception only observes.
Labels are the actions that actually execute after reflexes.

## Train locally

Primary run: do not rebalance action classes initially. That makes the cleanest comparison with
008 and tests whether the missing information really was relational.

    uv run arena train-bc \
      --dataset data/teacher_relational.jsonl \
      --weights models/relational_policy.json \
      --epochs 120 \
      --class-balance none

The trainer now reports per-class validation recall directly. The key measurements are
`focus_bane` and `cover_ally`.

Stim is trained once per battlefield decision rather than duplicated once per living Marine.
Because stim is rare, its binary loss uses positive-class weighting by default. Disable that only
for an ablation with `--no-balance-stim`.

If the relational representation helps but a rare class still has poor recall, a secondary
training ablation is available:

    uv run arena train-bc \
      --dataset data/teacher_relational.jsonl \
      --weights models/relational_policy-balanced.json \
      --epochs 120 \
      --class-balance sqrt

Do not treat the balanced run as the primary comparison.

## Evaluate without the commander

    uv run arena run \
      --policy jev_distilled_semantic \
      --runs 20 \
      --seed-base 0 \
      --weights models/relational_policy.json \
      --temperature 1.0

    uv run arena evaluate

Also run an argmax diagnostic:

    uv run arena run \
      --policy jev_distilled_semantic \
      --runs 10 \
      --seed-base 100 \
      --weights models/relational_policy.json \
      --temperature 0

Keep those runs separate before aggregation if you do not want them mixed.

## Success criteria

Episode 008 validation recall:

- kite: 0.90
- split: 0.91
- attack: 0.69
- retreat: 0.34
- focus_bane: 0.00
- cover_ally: 0.00
- stutter: 0.56

The main 009 representation test is not merely overall accuracy. It is:

1. `focus_bane` recall becomes materially non-zero on held-out episode seeds.
2. `cover_ally` recall becomes materially non-zero.
3. End-to-end win rate/survival improves beyond 008's 16/20 without restoring the commander.
4. Jev's referential target remains highly aligned with the teacher target.
5. Student inference remains inside the ~536 ms decision budget.

If those happen, the useful abstraction is no longer just "Jev as sigmoid neurons." It is closer
to a semantic graph: Jev supplies node state, entity references and edge activations; a tiny
numerical controller learns the policy on top.
