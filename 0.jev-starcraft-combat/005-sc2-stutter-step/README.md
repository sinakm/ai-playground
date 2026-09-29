# 005 — SC2 stutter-step

Same Marines-vs-Zerg fight as [004](../004-sc2-squad-commander/) (round 4h's
scenario and commander code), with one new technique: stutter-step. A
Marine's rifle has a cooldown between shots; stutter-stepping means shoot
when it's ready and step away while it reloads, so the Marine keeps its
damage output while taking less. TypeSafe's Jev (`jev-1.13.0`) decides which
Marines stutter; code executes the technique every game tick. Does giving
Jev this real technique — instead of another commander refinement — close
the gap with random that plateaued in episode 004? See [001's
glossary](../001-sc2-combat-arena/README.md#glossary) for term definitions.

## Video

Release: https://github.com/sinakm/ai-playground/releases/tag/ep005

- `ep005-stutter-step.mp4` — the cut: episode 004's commander (before) →
  `stutter_all` → Jev + stutter, with round cards.
- `ep005-stutter-all-win.mp4` — realtime `stutter_all` win: 8 Marines alive,
  13.9 s.
- `ep005-jev-stutter-win.mp4` — realtime `jev_commander_stutter` win, after
  the round 5b fix: 7 Marines alive, 9.9 s.

Download form: `https://github.com/sinakm/ai-playground/releases/download/ep005/<file>`

## Results

Scenario: 12 Marines vs 10 Banelings + 10 Zerglings (20 Zerg), 60 s fight
cap, paused evaluation. 20 fights per policy except `attack_move` (10).
Score = Zerg killed if any Marine survives, else 0. `random` here picks a
random action per Marine over all 9 soldier actions, including `stutter`.
`stutter_all` is scripted (no AI, no API calls): every living Marine
stutter-steps for the whole fight.

| Policy | Runs | Wins | Score | Win rate | Marines alive | Survivor HP | Enemies killed | Fight s | Median ms | p90 ms | Cost USD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| attack_move | 10 | 0 | 0 | 0.00 | 0 | - | 9.20 | 3.83 | - | - | 0.00 |
| random | 20 | 14 | 14 | 0.70 | 3 | 30.60 | 18.90 | 20.44 | - | - | 0.00 |
| stutter_all | 20 | 20 | 20 | 1.00 | 8.85 | 51.30 | 20 | 11.39 | - | - | 0.00 |
| jev_commander | 20 | 8 | 8 | 0.40 | 2 | 18.20 | 16.50 | 15.11 | 326.40 | 407.10 | 0.25 |
| jev_commander_stutter | 20 | 20 | 20 | 1.00 | 7.60 | 32.90 | 20 | 9.39 | 356.30 | 436.70 | 0.22 |

## What Jev decided and why

The `stutter` action runs entirely in code, every game tick between
decisions: if a Marine's `weapon_cooldown` is 0, it attacks the nearest
enemy; otherwise it steps 0.75 cells away from the nearest threat (the
nearest Baneling if one is within 6 cells, else the nearest enemy). Jev's
only job is deciding which Marines get that action, from the criterion
"enemies are within 5 cells and no baneling is about to reach you: shoot
when your rifle is ready, step back while it reloads" — one of nine soldier
actions offered to `jev_commander_stutter` (the other eight are round 4h's
set, minus `stim`, which the commander still calls separately).

Comparing the executed action mix across all 20 `jev_commander_stutter`
fights (after code-side reflexes and confidence gating, from
`decisions.jsonl`): `attack` 36%, `stutter` 21%, `split` 12%, `focus_bane`
9%, `kite` 9%, `retreat` 8%, `cover_ally` 3%, `retreat_to_squad` (a
reflex-only action) 3%. Jev used the new action about a fifth of the time,
concentrated when Marines were already clear of Banelings.

The headline: technique mattered more than the model. `jev_commander_stutter`
won all 20 fights, matching `stutter_all`'s scripted, no-AI 20/20, and beat
plain `jev_commander`'s 8/20 by a wide margin. But `stutter_all` still did
it better on every survival metric: `jev_commander_stutter` finished faster
(9.39 s vs 11.39 s) while keeping about one fewer Marine alive (7.60 vs
8.85) and at less than two-thirds the survivor HP (32.9 vs 51.3). Adding a
planning layer and partial stutter use won every fight, but a scripted
policy that stutters unconditionally was the stronger survivor.

Round 5b was a scenario/infrastructure fix, not a model change. In the
original realtime path the bot blocked on Jev's two calls (commander +
soldiers, roughly 350 ms) before applying any order, which froze the
stutter reflex for that whole window — a Marine mid-cooldown just stood
still instead of stepping away. All three realtime `jev_commander_stutter`
recordings taken before the fix were losses: 0 Marines alive each time
(14, 14, and 15 Zerg killed across the three, 11–18 s fights). The fix ran
Jev's decision calls on a background thread in realtime mode only (the
paused evaluation path above is unchanged); per-tick reflexes and the
stutter loop keep running while a call is in flight, and the returned
decision is applied against whichever units are still there when it
arrives. After the fix, the next three realtime recordings were 3/3 wins:
7 Marines alive in 9.91 s (2 late decisions, mean apply delay 12.4 loops),
2 Marines alive in 15.71 s (3 late, 10.3 loops), and 2 Marines alive in
17.14 s (0 late, 10.3 loops). The lesson: put fast reflexes in front of a
slower planner, not behind it.

## Cost and latency

The decision interval is 12 game loops ≈ 0.54 s (535.7 ms budget), same as
episode 004. `jev_commander_stutter`'s two-call latency (commander +
soldiers) was median 356.3 ms, p90 436.7 ms in paused evaluation — inside
budget; `jev_commander` (no `stutter` option) was similar, median 326.4 ms,
p90 407.1 ms. `late_decisions` is 0 for both in the paused `summary.json`.
20 fights of `jev_commander_stutter` cost $0.22; `jev_commander` cost $0.25
(more Marines alive longer in `jev_commander_stutter`'s wins means shorter
fights and fewer total decisions, despite the extra action option).
`random`, `stutter_all`, and `attack_move` make no API calls, so they have
no latency or cost.

Realtime showcase latency (post-5b-fix recordings) ran higher and more
variable than paused evaluation — individual calls up to 750 ms — because
the background thread now competes with the SC2 client and recording
pipeline for the CPU; the fix tolerates this by applying decisions late
against current unit positions rather than blocking the game.

## Limits and failures

- `stutter_all` — pure code, no AI at all — remains the strongest policy on
  survival: highest Marines alive (8.85) and highest survivor HP (51.3) of
  any policy, `jev_commander_stutter` included. Jev's planning layer wins
  as often but survives worse.
- Round 5b was a real bug, not a tuning issue: before the fix, all 3
  recorded realtime `jev_commander_stutter` fights were total losses (0
  Marines alive) because the bot's per-tick reflexes and stutter loop froze
  for the ~350 ms of each Jev call. This only affected the realtime path;
  the paused evaluation numbers above were never exposed to it.
- Even after the fix, realtime decisions apply roughly 10–12 game loops
  (about 0.45–0.5 s) after they were requested, and some are later than the
  12-loop decision interval itself (late decisions: 2 and 3 of 16–26 in two
  of the three post-fix recordings). Applying a decision against whichever
  units are still there covers for this, but it is not the same as the
  zero-delay paused evaluation.
- The realtime SC2 client steps roughly every 4 game loops rather than
  every 1, so realtime stutter-stepping is coarser than what the paused
  evaluation above measures; the showcase videos are not a frame-accurate
  reproduction of the results table.
- `evaluate.py` excludes every realtime run from the aggregate entirely
  (`results/table.md` and `results/summary.json` are paused-only), so
  realtime-only fields — `skipped_decisions`, `mean_apply_delay_loops`,
  `step_reflex_changes` — never appear in the results tables above; they
  are only in each realtime run's own `summary.json`.
- Small sample: 20 fights per policy (10 for `attack_move`), one map, one
  seed range; `summary.json` gives only means, no variance.
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
5. Policies in this episode: `attack_move`, `random`, `stutter_all`,
   `jev_commander`, `jev_commander_stutter`.
   - `uv run arena run --policy <name> --runs 20` (`--runs 10` for
     `attack_move`, matching the table above)
   - `uv run arena run --policy jev_commander_stutter --record` (captures
     realtime video + audio of one fight; implies `--realtime`)
   - `uv run arena evaluate` (aggregates paused `runs/` into `results/`;
     realtime runs are excluded)
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
