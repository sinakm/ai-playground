"""CLI: run fights, evaluate results, render the showcase video."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from sc2 import maps
from sc2.data import Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer

from arena import config
from arena.bot import ArenaBot
from arena.policies import POLICY_NAMES, make_policy

EP_DIR = Path(__file__).resolve().parents[2]
RUNS_DIR = EP_DIR / "runs"
RESULTS_DIR = EP_DIR / "results"

WARMUP_STATE = {"note": "warmup call before the fight", "rules": config.STATE_RULES}


def play_one(policy_name: str, seed: int, runs_dir: Path, realtime: bool, recorder_factory=None, training: bool = False, weights_path: Path | None = None, sample_eval: bool = False, temperature: float = 1.0) -> Path:
    run_dir = runs_dir / f"{policy_name}-{time.strftime('%Y%m%d-%H%M%S')}-s{seed}"
    run_dir.mkdir(parents=True)
    if (
        policy_name == "jev_trainable_semantic"
        and not training
        and weights_path is not None
        and not weights_path.exists()
    ):
        raise FileNotFoundError(
            f"policy weights not found: {weights_path}. Collect teacher traces and run 'arena clone' first."
        )
    policy = make_policy(policy_name, seed, training=training, weights_path=str(weights_path) if weights_path else None, sample_eval=sample_eval, temperature=temperature)
    if hasattr(policy, "warmup"):
        policy.warmup(WARMUP_STATE)
    recorder = recorder_factory(run_dir) if recorder_factory else None
    bot = ArenaBot(policy, run_dir, seed, realtime, recorder)
    try:
        run_game(
            maps.get(config.MAP_NAME),
            [Bot(Race.Terran, bot), Computer(Race.Zerg, Difficulty.VeryEasy)],
            realtime=realtime,
            save_replay_as=str(run_dir / "fight.SC2Replay"),
            random_seed=seed,
        )
    finally:
        bot.close()
    return run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="arena")
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="play fights")
    r.add_argument("--policy", choices=POLICY_NAMES, required=True)
    r.add_argument("--runs", type=int, default=1)
    r.add_argument("--seed-base", type=int, default=0)
    r.add_argument("--realtime", action="store_true")
    r.add_argument("--record", action="store_true", help="capture SC2 window (+ audio); implies --realtime")
    r.add_argument("--no-audio", action="store_true")
    r.add_argument("--train", action="store_true", help="fine-tune policy weights with episodic REINFORCE after each battle")
    r.add_argument("--weights", type=Path, default=EP_DIR / "models" / "semantic_policy_bc.json")
    r.add_argument("--sample-eval", action="store_true", help="sample actions at evaluation instead of argmax")
    r.add_argument("--temperature", type=float, default=1.0, help="softmax temperature for --sample-eval")
    sub.add_parser("evaluate", help="aggregate runs/ into results/")
    c = sub.add_parser("clone", help="behavior-clone the MLP from jev_teacher_collect runs")
    c.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    c.add_argument("--weights", type=Path, default=EP_DIR / "models" / "semantic_policy_bc.json")
    c.add_argument("--metrics", type=Path, default=EP_DIR / "models" / "clone_metrics.json")
    c.add_argument("--epochs", type=int, default=150)
    c.add_argument("--learning-rate", type=float, default=0.03)
    c.add_argument("--batch-size", type=int, default=512)
    c.add_argument("--balance-power", type=float, default=0.5)
    c.add_argument("--val-fraction", type=float, default=0.2)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--init-weights", type=Path, default=None)
    d = sub.add_parser("render", help="render showcase video for one run dir")
    d.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.cmd == "run":
        factory = None
        if args.record:
            from arena.record import make_recorder_factory

            factory = make_recorder_factory(audio=not args.no_audio)
        for i in range(args.runs):
            run_dir = play_one(args.policy, args.seed_base + i, RUNS_DIR, args.realtime or args.record, factory, args.train, args.weights, args.sample_eval, args.temperature)
            print(f"run {i + 1}/{args.runs}: {run_dir}")
    elif args.cmd == "evaluate":
        from arena.evaluate import evaluate, format_table

        print(format_table(evaluate(RUNS_DIR, RESULTS_DIR)))
    elif args.cmd == "clone":
        import json
        from arena.clone import train_behavior_clone

        metrics = train_behavior_clone(
            args.runs_dir,
            args.weights,
            args.metrics,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            balance_power=args.balance_power,
            val_fraction=args.val_fraction,
            seed=args.seed,
            init_weights=args.init_weights,
        )
        print(json.dumps(metrics, indent=2))
    else:
        from arena.overlay import render

        print(render(args.run, RESULTS_DIR / "summary.json"))


if __name__ == "__main__":
    main()
