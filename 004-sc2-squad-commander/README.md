# 004 — SC2 squad commander

Same Marines-vs-Zerg fight as [003](../003-sc2-marine-micro/), with a new
control structure: a Jev (`jev-1.13.0`) commander first picks a squad plan
and a priority Baneling target (call 1), then every living Marine answers
one of eight actions, seeing that plan and a shared "blackboard" of nearby
teammates (call 2). Does adding a planning layer and squad coordination beat
per-Marine-only control? See [001's glossary](../001-sc2-combat-arena/README.md#glossary)
for term definitions.

## Video

Release: https://github.com/sinakm/ai-playground/releases/tag/ep004

- `ep004-commander-4e-win.mp4` (round 4e) — win: 5 Marines alive, 20/20
  Zerg killed, 10.0 s, 0 late decisions.
- `ep004-commander-4c-loss.mp4` (round 4c) — loss: 10 kills in 7.7 s.

Download form: `https://github.com/sinakm/ai-playground/releases/download/ep004/<file>`

## Results

10 fights per policy per round unless noted, 60 s fight cap. Score = Zerg
killed if any Marine survives, else 0.

### Round 4a — commander + blackboard + support actions (partial run)

Scenario: 12 Marines vs 14 Banelings + 10 Zerglings (24 Zerg).
`attack_move` had 1 run, not 10, in this round.

| Policy | Runs | Wins | Score | Win rate | Marines alive | Enemies killed | Fight s | Median ms | p90 ms | Cost USD |
|---|---|---|---|---|---|---|---|---|---|---|
| attack_move | 1 | 0 | 0 | 0.00 | 0 | 9 | 3.48 | - | - | 0.00 |
| random | 10 | 0 | 0 | 0.00 | 0 | 17.60 | 20.58 | - | - | 0.00 |
| jev_commander | 10 | 0 | 0 | 0.00 | 0 | 8.50 | 7.20 | 332.30 | 421.50 | 0.06 |

### Round 4b — role facts added to state, plan-driven roles, focus cap 4

Scenario: 12 Marines vs 14 Banelings + 10 Zerglings (24 Zerg).

| Policy | Runs | Wins | Score | Win rate | Marines alive | Enemies killed | Fight s | Median ms | p90 ms | Cost USD |
|---|---|---|---|---|---|---|---|---|---|---|
| attack_move | 10 | 0 | 0 | 0.00 | 0 | 9.20 | 3.61 | - | - | 0.00 |
| random | 10 | 0 | 0 | 0.00 | 0 | 17.60 | 20.58 | - | - | 0.00 |
| jev_marine | 10 | 0 | 0 | 0.00 | 0 | 12.50 | 16.92 | 174.50 | 271.40 | 0.04 |
| jev_commander | 10 | 0 | 0 | 0.00 | 0 | 11.60 | 18.66 | 365.00 | 471.70 | 0.11 |

### Round 4c — back to 10 Banelings; same code as 4b

Scenario: 12 Marines vs 10 Banelings + 10 Zerglings (20 Zerg).

| Policy | Runs | Wins | Score | Win rate | Marines alive | Enemies killed | Fight s | Median ms | p90 ms | Cost USD |
|---|---|---|---|---|---|---|---|---|---|---|
| attack_move | 10 | 0 | 0 | 0.00 | 0 | 9.50 | 4.00 | - | - | 0.00 |
| random | 10 | 6 | 13.90 | 0.60 | 2.50 | 18.50 | 25.33 | - | - | 0.00 |
| jev_marine | 10 | 5 | 10 | 0.50 | 1.90 | 16.50 | 21.19 | 153.50 | 193.10 | 0.06 |
| jev_commander | 10 | 0 | 0 | 0.00 | 0 | 12.20 | 12.95 | 319.70 | 386.40 | 0.08 |

### Round 4d — kite reflex, `pre_split` plan, commander `stim_now`, Zerg targets nearest Marine

Scenario: 12 Marines vs 10 Banelings + 10 Zerglings (20 Zerg).

| Policy | Runs | Wins | Score | Win rate | Marines alive | Enemies killed | Fight s | Median ms | p90 ms | Cost USD |
|---|---|---|---|---|---|---|---|---|---|---|
| attack_move | 10 | 0 | 0 | 0.00 | 0 | 9.20 | 3.83 | - | - | 0.00 |
| random | 10 | 6 | 12 | 0.60 | 2.30 | 18.20 | 22.04 | - | - | 0.00 |
| jev_marine | 10 | 0 | 0 | 0.00 | 0 | 11.40 | 17.10 | 155.50 | 198.70 | 0.04 |
| jev_commander | 10 | 1 | 2 | 0.10 | 0.30 | 13.60 | 10.22 | 318.00 | 398.00 | 0.07 |

### Round 4e — Zergling swarm reflex (regroup when 3+ within 2 cells), `cover_ally` blocked near Banelings

Scenario: 12 Marines vs 10 Banelings + 10 Zerglings (20 Zerg).

| Policy | Runs | Wins | Score | Win rate | Marines alive | Enemies killed | Fight s | Median ms | p90 ms | Cost USD |
|---|---|---|---|---|---|---|---|---|---|---|
| attack_move | 10 | 0 | 0 | 0.00 | 0 | 9.20 | 3.83 | - | - | 0.00 |
| random | 10 | 6 | 12 | 0.60 | 2.50 | 17.90 | 21.32 | - | - | 0.00 |
| jev_marine | 10 | 0 | 0 | 0.00 | 0 | 11.40 | 17.10 | 155.50 | 198.70 | 0.04 |
| jev_commander | 10 | 2 | 4 | 0.20 | 0.60 | 15.20 | 11.48 | 318.80 | 393.50 | 0.08 |

## What Jev decided and why

`jev_commander` calls Jev twice per step: a commander picks one of five
squad plans (`focus_banes`, `bait_and_split`, `pre_split`, `hold_and_shoot`,
`fall_back`) and a priority Baneling, then every Marine picks one of up to
eight actions, told the plan and a blackboard of nearby teammates.

Round 4a's plan choice (`bait_and_split` 57% of decisions) went nowhere:
Marines never actually chose `bait` (0 times) because the state had no fact
telling a Marine it was the one closest to the Banelings. Round 4b added
that role fact and made roles plan-driven, and 4c reused the same code on
an easier scenario (10 Banelings instead of 14); executed actions became
`attack` 49%, `kite` 19%, `focus_bane` 13%, `bait` 9%.

Round 4d added a hard-coded kite reflex (step away whenever a Baneling is
within 2.5 cells, overriding whatever Jev chose) after round 4c showed
commander Marines only kited next to a Baneling 57% of the time — solo
`jev_marine` kited in that situation 91% of the time — because following
the plan pulled them into bait and focus-fire roles instead. The reflex cut
"danger moments" (a Marine in kite range) from 137 to 49 across the round.
Round 4d also fixed the scripted Zerg to chase the nearest Marine instead
of the squad's center; before this fix, `bait` could never work as
designed. That same fix made the fight harder for solo `jev_marine`, whose
kiting now drags Zerglings along behind it: its win rate dropped from 5/10
(round 4c) to 0/10 (round 4d).

Round 4e added a second reflex — regroup when 3 or more Zerglings are
within 2 cells — and blocked `cover_ally` near Banelings. `jev_commander`
reached its best result of the series: 2/10 wins, 15.20 kills, with 158
reflex overrides recorded across the 10 fights. Across all five rounds,
commander kills rose 8.5 → 11.6 → 12.2 → 13.6 → 15.2 and wins went from
0/10 to 2/10, but random (which mixes stim, split and retreat by chance)
still killed more on average (17.9 in round 4e).

## Cost and latency

The decision interval is 12 game loops ≈ 0.54 s (535.7 ms budget) for both
`jev_marine` and `jev_commander` throughout this episode. `jev_commander`'s
reported latency is the sum of its two calls (commander + soldiers); in
round 4e that was median 318.8 ms, p90 393.5 ms — inside budget.
`jev_marine`'s single call stayed lower: median 155.5 ms, p90 198.7 ms.
`late_decisions` is 0 across every round in `summary.json`. 10 fights of
`jev_commander` cost $0.06–$0.11 depending on the round; `jev_marine` cost
$0.04–$0.06.

## Limits and failures

- `jev_commander` never out-killed random on raw kills in any round; its
  best win rate (2/10, round 4e) still trails random's typical 6/10.
- Round 4a's `bait` plan was untestable: the scripted Zerg targeted the
  squad's center, so no Marine's "closest to the Banelings" fact was ever
  true in a way that made baiting happen. This was a scenario bug, not a
  model failure, and was only fixed in round 4d.
- Round 4d's fix (Zerg chases the nearest Marine) broke solo `jev_marine`,
  whose win rate dropped from 5/10 (round 4c) to 0/10 (round 4d and 4e),
  because its kiting now drags Zerglings toward itself instead of the
  group.
- Rounds 4d and 4e rely on two hard-coded reflexes (a kite-away trigger
  within 2.5 cells, and a regroup trigger when 3+ Zerglings are within 2
  cells) that override whatever action Jev chose. `retreat_to_squad` is an
  executed-only action never offered to Jev at all. The recorded fights in
  these rounds are Jev's plan plus code-side safety reflexes, not Jev
  acting alone — 158 reflex overrides were recorded in round 4e's 10
  fights.
- The client retries once on a transient TypeSafe error (HTTP 520,
  connection reset, timeout, or rate limit) — see `tests/test_api_errors.py`
  — but `api_errors` is 0 in every round's `summary.json`; none were hit
  during these recorded runs.
- More options did not automatically help: the 8-action commander scheme
  killed fewer Zerg than plain `attack_move` in round 4a until role facts
  and code-side roles existed to back the extra choices.
- Small sample: 10 fights per policy per round (1 for `attack_move` in
  round 4a), one map, one seed range.
- Audio capture in recorded runs (`soundcard`) is Windows-only.
- The scripted Zerg opponent is driven directly by the bot via
  `debug_control_enemy`, not the game's own AI.

## How to run

1. Install [Battle.net](https://battle.net) and free StarCraft II (Windows
   or Mac). Set graphics to Low.
2. `uv run python scripts/setup_maps.py` — downloads Blizzard's map pack and
   installs `Flat128.SC2Map`, accepting the [AI and Machine Learning
   License](https://blzdistsc2-a.akamaihd.net/AI_AND_MACHINE_LEARNING_LICENSE.html).
3. `cp .env.example .env` and set `TYPESAFE_API_KEY`.
4. If `uv` fails to install with a hardlink error (`os error 396`), run
   `export UV_LINK_MODE=copy` (PowerShell: `$env:UV_LINK_MODE = "copy"`)
   first.
5. Policies in this episode: `attack_move`, `random`, `jev_marine`,
   `jev_commander`.
   - `uv run arena run --policy <name> --runs 10`
   - `uv run arena run --policy jev_commander --record` (captures video +
     audio of one fight; implies `--realtime`)
   - `uv run arena evaluate` (aggregates `runs/` into `results/`)
   - `uv run arena render --run runs/<run-dir>` (renders the showcase video
     for one recorded run)
6. `uv run pytest` runs the unit tests (the one test marked `sc2` launches
   the game and is skipped by default).

## Credits and licenses

- StarCraft II and the map pack are Blizzard Entertainment's, used here
  under the [AI and Machine Learning
  License](https://blzdistsc2-a.akamaihd.net/AI_AND_MACHINE_LEARNING_LICENSE.html).
- Jev is TypeSafe's System One model, used under TypeSafe's Jev API terms.
- [python-sc2](https://github.com/BurnySc2/python-sc2) (`burnysc2`), MIT.
- This episode's code is MIT, see the repo `LICENSE`.
