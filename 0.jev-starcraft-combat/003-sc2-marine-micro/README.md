# 003 — SC2 marine micro

Same Marines-vs-Zerg fight as [002](../002-sc2-squad-with-goal/), but Jev
(`jev-1.13.0`) now answers one question per living Marine in a single API
call, instead of one question for the whole squad. Does per-Marine control
beat squad-wide control? See [001's glossary](../001-sc2-combat-arena/README.md#glossary)
for term definitions.

## Video

Release: https://github.com/sinakm/ai-playground/releases/tag/ep003

- `ep003-jev-per-marine-win.mp4` (round 3a) — Jev per-Marine win: 4 Marines
  alive, 16/16 Zerg killed, 12.8 s, 0 late decisions.

Download form: `https://github.com/sinakm/ai-playground/releases/download/ep003/<file>`

## Results

10 fights per policy per round, 60 s fight cap. Score = Zerg killed if any
Marine survives, else 0. `random` in this episode picks a random action per
Marine (same control granularity as `jev_marine`), not a random squad
action.

### Round 3a — per-Marine questions added (5 relative actions)

Scenario: 12 Marines vs 6 Banelings + 10 Zerglings (16 Zerg).

| Policy | Runs | Wins | Score | Win rate | Marines alive | Enemies killed | Fight s | Median ms | p90 ms | Cost USD |
|---|---|---|---|---|---|---|---|---|---|---|
| attack_move | 10 | 0 | 0 | 0.00 | 0 | 9.30 | 5.08 | - | - | 0.00 |
| random | 10 | 10 | 16 | 1.00 | 7.10 | 16 | 13.18 | - | - | 0.00 |
| jev_squad | 10 | 2 | 6.80 | 0.20 | 5.70 | 7.80 | 46.78 | 139.50 | 178.70 | 0.08 |
| jev_marine | 10 | 6 | 9.60 | 0.60 | 2.20 | 14.10 | 18.38 | 147.30 | 190.40 | 0.05 |

### Round 3b — harder scenario (14 Banelings), stronger stim wording

Scenario: 12 Marines vs 14 Banelings + 10 Zerglings (24 Zerg).

| Policy | Runs | Wins | Score | Win rate | Marines alive | Enemies killed | Fight s | Median ms | p90 ms | Cost USD |
|---|---|---|---|---|---|---|---|---|---|---|
| attack_move | 10 | 0 | 0 | 0.00 | 0 | 9.20 | 3.61 | - | - | 0.00 |
| random | 10 | 3 | 13.50 | 0.30 | 1.50 | 21.30 | 37.41 | - | - | 0.00 |
| jev_squad | 10 | 1 | 11.70 | 0.10 | 4.60 | 11.70 | 56.22 | 158.60 | 321.80 | 0.09 |
| jev_marine | 10 | 0 | 0 | 0.00 | 0 | 11.60 | 14.12 | 166.80 | 1116.10 | 0.04 |

## What Jev decided and why

`jev_marine` asks one question per living Marine — `kite`, `split`,
`attack`, `stim`, or `retreat` — in one call, so different Marines can take
different actions on the same step (Marine 3 kites while Marine 7 attacks).
In round 3a its action mix was `attack` 49%, `kite` 33%, `retreat` 9%,
`split` 8%, `stim` 0%. Per-Marine control clearly beat squad control on
this fight: `jev_marine` won 6/10 vs `jev_squad`'s 2/10, though random
(also per-unit, but truly random) won all 10.

Round 3b made the fight harder (14 Banelings instead of 6) and strengthened
the stim wording in the criteria text. `jev_marine`'s mix shifted to `kite`
49%, `attack` 37%, `stim` 1%, but it won 0/10 — the added Banelings
overwhelmed it. `jev_squad` still eked out 1/10 win. Across both rounds Jev
almost never stims (0% in 3a, 1% in 3b); random's roughly 20% stim rate is
a large part of why random keeps winning.

## Cost and latency

Both rounds ran at 12 game loops ≈ 0.54 s per decision step (535.7 ms
budget). `jev_marine` median latency stayed low (147.3 ms in 3a, 166.8 ms
in 3b), but round 3b's p90 spiked to 1116.1 ms — well over budget — because
answering more Marines per call (more Zerg pressure, more living Marines
early on) means a bigger per-call payload; `late_decisions` is 0 in the
aggregate results, so the slow calls didn't visibly break execution in
these runs, but the margin is thin. `jev_squad`'s single-question latency
stayed comfortably under budget in both rounds (p90 178.7 ms and 321.8 ms).
10 fights of `jev_marine` cost $0.05 (3a) and $0.04 (3b); `jev_squad` cost
$0.08 (3a) and $0.09 (3b).

## Limits and failures

- Random is a strong baseline throughout: 10/10 wins in round 3a, and still
  the best score in round 3b (13.50 vs `jev_squad`'s 11.70).
- `jev_marine` went from 6/10 wins (3a) to 0/10 (3b) when the scenario got
  harder; per-Marine control does not automatically scale to a tougher
  fight.
- Jev almost never chooses `stim` even though it is one of five available
  actions (0% in 3a, 1% in 3b).
- Round 3b's p90 latency for `jev_marine` (1116.1 ms) is well past the
  535.7 ms real-time decision budget.
- Small sample: 10 fights per policy per round, one map, one seed range.
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
5. Policies in this episode: `attack_move`, `random`, `jev_squad`,
   `jev_marine`.
   - `uv run arena run --policy <name> --runs 10`
   - `uv run arena run --policy jev_marine --record` (captures video +
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
