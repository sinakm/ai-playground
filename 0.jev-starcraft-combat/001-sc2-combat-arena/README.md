# 001 — SC2 combat arena

TypeSafe's Jev (`jev-1.13.0`, System One structured-decision model) picks one
squad-wide action every 0.18 seconds for 12 Terran Marines fighting 6 Zerg
Banelings and 10 Zerglings: can a single structured-decision call beat "do
nothing smart" and beat picking randomly?

## Video

Release: https://github.com/sinakm/ai-playground/releases/tag/ep001

- `ep001-jev-v1-spread-only.mp4` — Jev's fight. The recorded run timed out
  with 10 Marines alive, 1 Zerg killed, 33 of 312 decisions late.
- `ep001-random-baseline-win.mp4` — the random baseline's win: 7 Marines
  alive, 16/16 Zerg killed, 13.75 s.

Download form: `https://github.com/sinakm/ai-playground/releases/download/ep001/<file>`

## Results

Scenario: 12 Marines vs 6 Banelings + 10 Zerglings (16 Zerg), 10 fights per
policy, 60 s fight cap.

| Policy | Runs | Wins | Win rate | Marines alive | Enemies killed | Fight s | Median ms | p90 ms | Cost USD |
|---|---|---|---|---|---|---|---|---|---|
| attack_move | 10 | 0 | 0.00 | 0 | 9 | 4.75 | - | - | 0.00 |
| random | 10 | 5 | 0.50 | 2.30 | 12.60 | 19.23 | - | - | 0.00 |
| jev_squad | 10 | 0 | 0.00 | 9.70 | 2.50 | 60.00 | 143.80 | 178.40 | 0.27 |

(`attack_move` and `random` make no API calls, so they have no latency or cost.)

## What Jev decided and why

Jev chose one action for the whole squad each step: `spread`, `clump`,
`retreat`, `attack`, or `stim`. The criteria text described what each action
*was*, not *when* to use it ("split marines apart so baneling splash hits
fewer of them"), and that description sounded reasonable in every game
state. Jev picked `spread` 100% of the time. Because `spread` in this round
was a pure move order with no attack step, and a new move order overwrote
the old one every 0.18 s, the squad walked apart and never closed to fight.
It survived (9.70 Marines alive on average) but almost never killed
anything (2.50 kills on average) and ran out the 60 s clock in every one of
10 fights.

## Cost and latency

Jev's median decision latency was 143.8 ms, p90 178.4 ms, against a
real-time decision budget of 4 game loops ≈ 0.18 s (178.6 ms) per step —
inside budget on the whole, though p90 sits close to the edge. 10 fights of
`jev_squad` cost $0.27 (6,467,889 input tokens, 188,160 output tokens at
$0.042/M input, $0/M output).

## Limits and failures

- Jev lost every fight (0/10). The random baseline won half its fights
  (5/10) and did far more damage per fight than Jev.
- `spread` was a pure move action with no criteria for *when* to use it, so
  Jev always chose it and the squad never engaged — see "what Jev decided"
  above. This is a scenario/prompt design flaw fixed in episode 002, not a
  model capability finding.
- Small sample: 10 fights per policy, one map, one seed range.
- Audio capture in recorded runs (`soundcard`) is Windows-only.
- The scripted Zerg opponent is not the game's own AI: the bot drives it
  directly with `debug_control_enemy`, so its behavior is fully
  reproducible but is not "real" Zerg play.

## How to run

1. Install [Battle.net](https://battle.net) and free StarCraft II (Windows
   or Mac). Set graphics to Low.
2. `uv run python scripts/setup_maps.py` — downloads Blizzard's map pack and
   installs `Flat128.SC2Map`. Extracting it accepts the [AI and Machine
   Learning License](https://blzdistsc2-a.akamaihd.net/AI_AND_MACHINE_LEARNING_LICENSE.html).
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

## Glossary

- **Marine** — Terran ranged infantry unit; this arena's controlled unit,
  12 per fight.
- **Zergling** — fast Zerg melee unit.
- **Baneling** — Zerg unit that explodes on contact, dealing splash damage
  to every unit nearby.
- **Cell** — one map grid unit of distance; the unit used in Jev's criteria
  text ("within 3 cells").
- **Game loop** — StarCraft II's internal simulation tick, 22.4 per second;
  decision intervals in this series are measured in loops.
- **Kite** — move away from an approaching enemy between attacks, to keep
  dealing damage while taking less.
- **Split** — step away from a clumped teammate so one Baneling can't hit
  both.
- **Stim** — Stimpack, a Marine ability that speeds up attack and movement
  for a duration at an HP cost.
- **Attack-move** — an SC2 order that moves toward a point and automatically
  engages any enemy found along the way, without kiting or splitting.
