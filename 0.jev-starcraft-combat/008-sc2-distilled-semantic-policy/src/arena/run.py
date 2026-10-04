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


def play_one(policy_name: str, seed: int, runs_dir: Path, realtime: bool, recorder_factory=None, training: bool = False, weights_path: Path | None = None, dataset_path: Path | None = None, temperature: float = 0.7) -> Path:
    run_dir = runs_dir / f"{policy_name}-{time.strftime('%Y%m%d-%H%M%S')}-s{seed}"
    run_dir.mkdir(parents=True)
    policy = make_policy(policy_name, seed, training=training, weights_path=str(weights_path) if weights_path else None, dataset_path=str(dataset_path) if dataset_path else None, temperature=temperature)
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
    r.add_argument("--train", action="store_true", help="update episode-007 policy weights after each battle")
    r.add_argument("--weights", type=Path, default=EP_DIR / "models" / "distilled_policy.json")
    r.add_argument("--dataset", type=Path, default=EP_DIR / "data" / "teacher.jsonl")
    r.add_argument("--temperature", type=float, default=0.7, help="student sampling temperature; 0 = argmax")
    bc = sub.add_parser("train-bc", help="train the episode-008 student from teacher JSONL")
    bc.add_argument("--dataset", type=Path, default=EP_DIR / "data" / "teacher.jsonl")
    bc.add_argument("--weights", type=Path, default=EP_DIR / "models" / "distilled_policy.json")
    bc.add_argument("--epochs", type=int, default=120)
    bc.add_argument("--batch-size", type=int, default=256)
    bc.add_argument("--learning-rate", type=float, default=0.03)
    bc.add_argument("--seed", type=int, default=0)
    sub.add_parser("evaluate", help="aggregate runs/ into results/")
    d = sub.add_parser("render", help="render showcase video for one run dir")
    d.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.cmd == "run":
        factory = None
        if args.record:
            from arena.record import make_recorder_factory

            factory = make_recorder_factory(audio=not args.no_audio)
        for i in range(args.runs):
            run_dir = play_one(args.policy, args.seed_base + i, RUNS_DIR, args.realtime or args.record, factory, args.train, args.weights, args.dataset, args.temperature)
            print(f"run {i + 1}/{args.runs}: {run_dir}")
    elif args.cmd == "train-bc":
        import json
        from arena.distill import train_behavior_clone
        metrics = train_behavior_clone(
            args.dataset, args.weights, epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.learning_rate, seed=args.seed,
        )
        print(json.dumps(metrics, indent=2))
    elif args.cmd == "evaluate":
        from arena.evaluate import evaluate, format_table

        print(format_table(evaluate(RUNS_DIR, RESULTS_DIR)))
    else:
        from arena.overlay import render

        print(render(args.run, RESULTS_DIR / "summary.json"))


if __name__ == "__main__":
    main()
