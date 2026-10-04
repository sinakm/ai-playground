"""Behavior-cloning utilities for episode 008."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from arena.semantic_policy import ACTIONS, GLOBAL_KEYS, SemanticMLP


def _records(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def load_teacher_run(run_dir: Path):
    """Return action samples and one global stim sample per teacher decision."""
    xs, ys, stim_xs, stim_ys = [], [], [], []
    net = SemanticMLP(seed=0)
    decisions = run_dir / "decisions.jsonl"
    if not decisions.exists():
        return xs, ys, stim_xs, stim_ys

    for rec in _records(decisions):
        if rec.get("policy") != "jev_teacher_collect" or rec.get("action") == "api_error":
            continue
        global_a = rec.get("semantic_activations") or {}
        local_a = rec.get("local_activations") or {}
        executed = rec.get("executed_actions") or {}

        # Commander stim is an independent simultaneous action, so learn it once per
        # decision rather than duplicating it for every Marine.
        stim_xs.append(np.asarray([global_a.get(k, 0.5) for k in GLOBAL_KEYS], dtype=float))
        stim_ys.append(int(bool(rec.get("stim_now"))))

        for tag, action in executed.items():
            if action not in ACTIONS:
                # retreat_to_squad is an executed-only reflex and remains code-side.
                continue
            local = local_a.get(str(tag), local_a.get(tag, {}))
            xs.append(net.vector(global_a, local))
            ys.append(ACTIONS.index(action))
    return xs, ys, stim_xs, stim_ys


def teacher_diagnostics(runs):
    wins = 0
    target_total = 0
    target_matches = 0
    for run in runs:
        summary = run / "summary.json"
        if summary.exists():
            wins += int(json.loads(summary.read_text(encoding="utf-8")).get("result") == "win")
        decisions = run / "decisions.jsonl"
        if not decisions.exists():
            continue
        for rec in _records(decisions):
            if rec.get("policy") != "jev_teacher_collect" or rec.get("action") == "api_error":
                continue
            teacher = rec.get("priority_target")
            perceived = rec.get("perception_target")
            if teacher is not None and perceived is not None:
                target_total += 1
                target_matches += int(teacher == perceived)
    return {
        "teacher_wins": wins,
        "teacher_win_rate": wins / len(runs) if runs else None,
        "target_comparisons": target_total,
        "perception_target_agreement": target_matches / target_total if target_total else None,
    }


def _merge(runs):
    xs, ys, sx, sy = [], [], [], []
    for run in runs:
        a, b, c, d = load_teacher_run(run)
        xs.extend(a); ys.extend(b); sx.extend(c); sy.extend(d)
    return (
        np.asarray(xs, dtype=float),
        np.asarray(ys, dtype=int),
        np.asarray(sx, dtype=float),
        np.asarray(sy, dtype=int),
    )


def train_behavior_clone(
    runs_dir: Path,
    weights_path: Path,
    metrics_path: Path | None = None,
    epochs: int = 150,
    learning_rate: float = 0.03,
    batch_size: int = 512,
    balance_power: float = 0.5,
    val_fraction: float = 0.2,
    seed: int = 0,
    init_weights: Path | None = None,
):
    runs = sorted(
        p for p in Path(runs_dir).iterdir()
        if p.is_dir() and p.name.startswith("jev_teacher_collect-") and (p / "decisions.jsonl").exists()
    )
    if not runs:
        raise ValueError("no jev_teacher_collect runs found")

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(runs))
    runs = [runs[i] for i in order]
    n_val = 0 if len(runs) == 1 else max(1, int(round(len(runs) * val_fraction)))
    val_runs = runs[:n_val]
    train_runs = runs[n_val:]
    if not train_runs:
        train_runs, val_runs = runs, []

    x, y, sx, sy = _merge(train_runs)
    vx, vy, vsx, vsy = _merge(val_runs) if val_runs else (
        np.empty((0, 10)), np.empty((0,), dtype=int),
        np.empty((0, len(GLOBAL_KEYS))), np.empty((0,), dtype=int),
    )

    model = SemanticMLP(seed=seed)
    if init_weights is not None:
        model.load(init_weights)
    train_metrics = model.fit_supervised(
        x, y, sx, sy,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        balance_power=balance_power,
        seed=seed,
    )
    val_metrics = model.supervised_metrics(vx, vy, vsx, vsy)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(weights_path)

    metrics = {
        "teacher_runs": len(runs),
        **teacher_diagnostics(runs),
        "train_runs": len(train_runs),
        "validation_runs": len(val_runs),
        "train_action_samples": int(len(x)),
        "validation_action_samples": int(len(vx)),
        "train_stim_samples": int(len(sx)),
        "validation_stim_samples": int(len(vsx)),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "balance_power": float(balance_power),
        "train": train_metrics,
        "validation": val_metrics,
        "weights_path": str(weights_path),
    }
    if metrics_path is not None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
