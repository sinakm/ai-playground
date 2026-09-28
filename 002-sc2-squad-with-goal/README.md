# 002 — SC2 squad with a goal

Same fight as [001](../001-sc2-combat-arena/): TypeSafe's Jev (`jev-1.13.0`)
picks one squad-wide action for 12 Terran Marines against Zerg Banelings and
Zerglings. This episode gives Jev an explicit goal and score in the prompt,
and asks whether telling Jev *when* to use each action (instead of only
*what* it does) changes its behavior. See [001's glossary](../001-sc2-combat-arena/README.md#glossary)
for term definitions.

## Video

Release: https://github.com/sinakm/ai-playground/releases/tag/ep002

- `ep002-jev-squad-win.mp4` — Jev's win: 6 Marines alive, 16/16 Zerg killed,
  17.1 s, 0 late decisions.

Download form: `https://github.com/sinakm/ai-playground/releases/download/ep002/<file>`

## Results

Scenario: 12 Marines vs 6 Banelings + 10 Zerglings (16 Zerg), 10 fights per
policy, 60 s fight cap. Score = Zerg killed if any Marine survives, else 0.

### Round 2a — goal + score added, `spread` now kites, still 0.18 s steps

`jev_squad` had 8 runs, not 10, in this round.

| Policy | Runs | Wins | Score | Win rate | Marines alive | Enemies killed | Fight s | Median ms | p90 ms | Cost USD |
|---|---|---|---|---|---|---|---|---|---|---|
| attack_move | 10 | 0 | 0 | 0.00 | 0 | 9 | 4.75 | - | - | 0.00 |
| random | 10 | 6 | 9.60 | 0.60 | 2.70 | 12.80 | 16.08 | - | - | 0.00 |
| jev_squad | 8 | 0 | 2.50 | 0.00 | 9.75 | 2.50 | 60.00 | 149.80 | 202.90 | 0.23 |

### Round 2b — decision steps slowed to 0.54 s, criteria say *when* to act

| Policy | Runs | Wins | Score | Win rate | Marines alive | Enemies killed | Fight s | Median ms | p90 ms | Cost USD |
|---|---|---|---|---|---|---|---|---|---|---|
| attack_move | 10 | 0 | 0 | 0.00 | 0 | 9.30 | 5.08 | - | - | 0.00 |
| random | 10 | 7 | 11.20 | 0.70 | 4.30 | 13.10 | 8.92 | - | - | 0.00 |
| jev_squad | 10 | 4 | 7.90 | 0.40 | 3.90 | 11.20 | 36.30 | 147.60 | 186.20 | 0.05 |

## What Jev decided and why

In round 2a, `spread` was rewritten to actually kite (step away, then keep
attacking), and the state gained "danger" fields. It made no difference:
Jev still picked `spread` 100% of the time, and its step-away move order
was overwritten by the next decision before the attack step could run,
because decisions still came every 0.18 s. Jev again survived (9.75 Marines
alive) without killing (2.50 kills), 0/8 wins.

Round 2b made two changes at once: decisions slowed to every 0.54 s (giving
each move-then-attack action time to finish), and the action criteria were
rewritten to say *when* to use each one instead of only what it does — e.g.
`spread` became "Banelings are within 3 cells of the squad and marines are
clumped." That produced a real mix for the first time: `spread` 69%,
`clump` 19%, `attack` 12%. Jev won 4/10 fights, its best result across both
rounds and the first time it out-won the do-nothing `attack_move` baseline.

## Cost and latency

Both rounds ran at a decision budget under 12 loops. Round 2a still used
4 game loops ≈ 0.18 s per step (178.6 ms budget); Jev's median latency was
149.8 ms, p90 202.9 ms — p90 exceeded budget. Round 2b moved to 12 loops ≈
0.54 s per step (535.7 ms budget); median 147.6 ms, p90 186.2 ms, both well
inside budget, and `late_decisions` was 0 in both rounds per `summary.json`.
10 fights of `jev_squad` cost $0.23 in round 2a and $0.05 in round 2b (fewer
tokens per call once the prompt got more specific).

## Limits and failures

- Random still beat Jev on raw kills and score in both rounds (round 2b:
  random scored 11.20 vs Jev's 7.90).
- Round 2a shows that better state fields alone don't change behavior if
  the decision cadence undoes the action; the fix (round 2b) needed both a
  slower cadence and criteria that state *when* to act.
- Small sample: 8–10 fights per policy per round, one map, one seed range.
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
5. Policies in this episode: `attack_move`, `random`, `jev_squad`.
   - `uv run arena run --policy <name> --runs 10`
   - `uv run arena run --policy jev_squad --record` (captures video + audio
     of one fight; implies `--realtime`)
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
