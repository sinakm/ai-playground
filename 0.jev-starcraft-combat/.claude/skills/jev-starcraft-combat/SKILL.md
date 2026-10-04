---
name: jev-starcraft-combat
description: Everything needed to continue the Jev x StarCraft II combat series (episodes 001-005) - guiding principles, environment setup, how the arena and Jev integration work, what was tried, what failed, tech debt and future ideas. Use before starting a new episode or round in this folder, or when debugging the SC2 arena.
---

# Jev x StarCraft II combat (series 0)

Episode 006/007 results and open issues: `notes/2026-10-03-ep006-ep007-findings.md`.

TypeSafe's Jev (a "System One" structured-decision model) controls 12 Terran
Marines against Zerg Banelings and Zerglings in StarCraft II. Each episode
folder is self-contained (own `pyproject.toml`, README, `src/arena/`, tests,
`results/`). Later episodes copy code from earlier ones on purpose.

| Ep | Folder | Question | Best Jev result |
|----|--------|----------|-----------------|
| 001 | `001-sc2-combat-arena` | Squad-level Jev, prompt without a goal | 0/10, spread 100% |
| 002 | `002-sc2-squad-with-goal` | Goal in prompt, 0.54 s steps, "when to use" criteria | 4/10 (random 7/10) |
| 003 | `003-sc2-marine-micro` | One question per Marine | 6/10 (random 10/10) |
| 004 | `004-sc2-squad-commander` | Commander + blackboard + reflexes, rounds 4a-4h | 10/20 at 4f (random 15/20) |
| 005 | `005-sc2-stutter-step` | Stutter-step action; background Jev in realtime | 20/20 (stutter_all also 20/20) |
| 006 | `006-sc2-semantic-net` | Six shared Noul perceptions instead of a commander plan | 16/20 (commander and stutter_all 20/20) |
| 007 | `007-sc2-trainable-semantic-policy` | Jev as frozen perception, NumPy MLP trained with REINFORCE | no learning in 40 fights; see `notes/` |
| 008 | `008-sc2-distilled-semantic-policy` | Behavior-clone the 005 commander into an 11-input MLP over Jev perception | 16/20 (teacher 49/50); see `notes/` |

## Guiding principles

- Measure, don't estimate. 10 fights per policy per round (20 from 004f).
  Report failures and baselines that beat the model.
- Always run baselines in the same folder and code: `attack_move` (floor),
  `random` (same action pool), and a scripted technique baseline when a new
  technique is added (`stutter_all`). A scripted baseline can be the honest
  winner.
- Change one thing per round. Archive the previous round before changing
  code: move `runs/*-2026*` to `runs/round<X>/` and copy `results/` files to
  `results/round<X>/`. (Also copy `src/arena/config.py` into the archive: not
  done so far, see tech debt.)
- Evaluation runs are paused (`realtime=False`, game waits for decisions).
  Showcase videos are realtime and are NOT reproducible from paused seeds.
  Report both honestly.
- No test calls the Jev API or launches SC2 unless marked `@pytest.mark.sc2`.
- Every number in a README, video card or post is checked against
  `results/summary.json`, `results/table.md` or run logs. Distinguish Jev's raw
  choices (`marine_actions`) from executed actions (`executed_actions`, after
  code rules): in 005 Jev itself chose `stutter` 4%, the executed mix was 21%.
- Criteria text tells Jev WHEN to use an option, not what it does. Jev can
  only judge facts present in the state.

## Environment setup (Windows)

1. Battle.net (`winget install Blizzard.BattleNet`), then install StarCraft II
   (free to play: "Play Free" on the SC2 shop page, NOT StarCraft Remastered).
   Launch once, set graphics to Low. Default path
   `C:\Program Files (x86)\StarCraft II`; python-sc2 finds it there even when
   Documents is redirected to OneDrive.
2. `uv` with `UV_LINK_MODE=copy` (hardlinks fail on this machine:
   "os error 396"). Each episode: `cd <episode> && uv sync`.
3. Map: `uv run python scripts/setup_maps.py` installs `Flat128.SC2Map` from
   Blizzard's Melee pack (zip password `iagreetotheeula`, AI and ML License).
4. `.env` with `TYPESAFE_API_KEY=` anywhere up the folder tree
   (`load_dotenv(find_dotenv(usecwd=True))`). Never commit it; gitleaks runs
   as a pre-commit hook in this repo.
5. `ffmpeg` on PATH for recording/rendering; `soundcard` records speaker
   loopback (Windows only). Turn on Do Not Disturb while recording.
6. Memory: SC2 + long batches got killed by low memory twice. Run fights in
   foreground batches of 5 (`--runs 5 --seed-base N`).

Commands (per episode):

```
uv run pytest                                  # unit tests, no SC2, no API
uv run pytest -m sc2                           # SC2 smoke test (attack_move)
uv run arena run --policy <name> --runs 5 --seed-base 0
uv run arena run --policy <name> --record      # realtime showcase + capture
uv run arena evaluate                          # results/table.md, summary.json, chart.png
uv run arena render --run runs/<dir>           # showcase.mp4 with Jev panel
```

Series tools (run from an episode folder with its venv):
`tools/build_cut.py` (cards + clips into one video), `tools/action_demo.py`
(screenshot each forced action).

## How the arena works

- Map Flat128; spawn with `debug_create_unit` at center (64,64): Marines at
  x-7, Banelings at x+7, Zerglings at x+8,y+2. Counts in `config.py`
  (6, 10 or 14 Banelings; 10 Zerglings).
- `debug_control_enemy()` once (it is a toggle): our bot orders the Zerg. The
  built-in AI left spawned units idle.
- Visibility: `debug_show_map()` (a toggle). Do NOT also pass
  `run_game(disable_fog=True)`: in realtime they cancel each other and the
  spawned Zerg become invisible. Without show_map the Zerg were invisible
  depending on the seed (they sit outside Marine sight range).
- 20 s pre-fight wait (448 loops) with Zerg held at spawn anchors, so SC2's
  debug chat text fades before the fight. Fight starts when 12 Marines and all
  Zerg are visible (or spawn + 96 loops); aborts after that with
  `result: "aborted"` (excluded from evaluation).
- Zerg: until 004c attack-move to the Marine center every 8 loops; from 004d
  each Zerg attacks its nearest Marine (realistic; broke kiting and made
  `bait` meaningful).
- Decisions every 12 loops (0.54 s at Faster speed) from 002b (4 loops before:
  move orders were replaced before their queued attack ever ran).
- Camera moves to the fight at spawn and follows the midpoint of both armies.
- Paused mode sets `game_step = 1`; realtime uses python-sc2's step of 4.
- `on_end` fallback writes `summary.json` when python-sc2 ends the game before
  `on_step` sees the final state. `ArenaBot.close()` is idempotent; `play_one`
  closes it in `finally`.

## Jev integration facts

- SDK `typesafe-sdk==0.7.2`: `TypeSafeClient().system_one(state=<dict>,
  questions={name: Choice(instructions=..., criteria={option: text})})`.
  Answers: `.choice`, `.probabilities`, `.confidence`; `Noul` answers expose
  `.noul` (probability of yes). Response `.usage.input_tokens`, `.model`
  (`jev-1.13.0`).
- Latency: median ~150 ms; 12 questions in one call cost about the same time as
  1; two sequential calls ~330 ms median. Concurrent calls are SLOWER than
  sequential (10 at once took 4.1 s): always one call at a time.
- Price $0.042 per million input tokens, output free. A 20-fight commander
  round cost ~$0.25. Series total ~$3.
- Transient `TypeSafeInternalServerError` (HTTP 520) happens: 004+ retries
  once, then logs an `api_error` step and keeps previous orders.
- Realtime: a blocking Jev call freezes every per-tick behavior. 005b runs Jev
  on a background thread (`scheduler.DecisionScheduler`) in realtime only;
  answers land ~10-12 game loops after the request.
- Confidence is a real signal: routing answers under 0.5 to simple code
  defaults (004f) was the biggest single commander gain.

## What worked, what didn't

Worked:
- Telling Jev the goal and the score (002).
- Criteria phrased as "when to use it" (002b): Jev started mixing actions.
- Per-Marine questions in one call (003): better than squad-level.
- Code reflexes under Jev's plan (004d/e): kite when a Baneling is within 2.5
  cells, regroup when swarmed. "Fast reflexes in front, slower planner behind."
- Confidence gating (004f): 2/10 -> 10/20 wins.
- Stutter-step as a code primitive (005): scripted and Jev both 20/20.
- Background Jev thread in realtime (005b): live wins 0/3 -> 3/3.

Didn't work:
- Prompt without a goal: `spread` 100% forever (001).
- `spread` as a pure move order: Marines stop shooting (001) and a queued
  attack never runs if orders are replaced every 0.18 s (002a).
- Options Jev can't evaluate: `bait` "if you are closest" before the state
  had `is_closest_to_banelings` (004a: 0 baits).
- `bait` while Zerg targeted the squad center (could never pull anything).
- Plans overriding survival: commander Marines kited 57% next to Banelings vs
  91% solo (004c).
- Widening `pre_split` to 12 cells: stayed true all fight, 46% split, no
  shooting (4f); fixed by pre-contact-only (4g).
- More rule tuning after 4f: plateau at ~50% (4g, 4h). The gap was HP
  efficiency, i.e. a missing technique, not decisions.
- Jev almost never chooses `stim` or `stutter` on its own (stim 0-1%,
  stutter 4%); defaults and commander-level questions did that work.

## Tech debt

- Archived rounds lack a `config.py` snapshot; unit counts and intervals of old
  rounds are known only from READMEs/specs.
- `evaluate.py` does not aggregate realtime-only fields (`skipped_decisions`,
  `mean_apply_delay_loops`) nor per-fight spread (no 95% CI in summary.json).
- Realtime showcases don't reproduce paused results (timing); showcase picks
  are "best of N takes", say so.
- Reflexes only run at decision steps in paused mode (every 12 loops); 005b
  runs them every tick only in realtime.
- `jev_marine` in 003b had p90 latency 1.1 s with 24 Zerg in the state (over
  the 0.54 s budget).
- SC2 window size varies with Windows DPI (1024x768 or ~2496x1558); the
  overlay letterboxes into 1440x1080.
- Episodes duplicate code by design; bug fixes do not propagate backwards.
- 002 and 004 contain several rounds in one folder; `results/` top level is
  always the latest round.
- `random`'s action pool changes between episodes, so random is not the same
  baseline across episodes.

## Future ideas

- Jev vs Jev: Jev also commands the Zerg (targets per Baneling, surround/wait).
- A harder fight where `stutter_all` alone loses, so judgment matters again.
- Stutter as a commander-level decision (who stutters), since soldiers rarely
  pick it.
- Per-Marine target selection among nearest Banelings (`Choice` with dynamic
  options) and threat scoring with `Score` questions.
- Last step's outcome in the state (feedback loop).
- Realtime-mode evaluation to price latency into results.
- Medivacs ("heal which Marine?"), terrain, full game with scripted economy.

## Starting a new episode

1. Write a short spec (goal, changes, policies, success criteria).
2. Copy the previous episode folder without `runs/`, `results/`, `.venv/`;
   rename the project in `pyproject.toml`; `uv sync`; tests pass before changes.
3. Implement with tests first for pure modules; SC2 smoke test with
   `attack_move`; one free baseline run before any Jev run.
4. Run free policies first, Jev last, batches of 5, then `arena evaluate`.
5. Record 1-3 realtime showcase takes, render, look at frames, build a cut.
6. README (contract in the repo root README), release `ep<NNN>` with videos,
   update the series table above.
