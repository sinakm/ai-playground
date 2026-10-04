import json
from pathlib import Path

import numpy as np

from arena.clone import load_teacher_run, train_behavior_clone
from arena.semantic_policy import ACTIONS, SemanticMLP


def _write_run(root: Path, name: str, rows: list[dict]):
    d = root / name
    d.mkdir(parents=True)
    (d / "decisions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )
    return d


def _row(loop: int, action1="attack", action2="split", stim=False):
    global_a = {
        "baneling_pressure": 0.8 if action2 == "split" else 0.2,
        "clumping_danger": 0.9 if action2 == "split" else 0.1,
        "encirclement_risk": 0.3,
        "focus_fire_opportunity": 0.4,
        "retreat_pressure": 0.2,
        "formation_instability": 0.7,
    }
    return {
        "fight_loop": loop,
        "policy": "jev_teacher_collect",
        "action": action1,
        "semantic_activations": global_a,
        "local_activations": {
            "1": {"personal_danger": 0.1, "isolation": 0.1, "escape_pressure": 0.1, "firing_opportunity": 0.9},
            "2": {"personal_danger": 0.8, "isolation": 0.2, "escape_pressure": 0.7, "firing_opportunity": 0.3},
        },
        "executed_actions": {"1": action1, "2": action2, "3": "retreat_to_squad"},
        "stim_now": stim,
    }


def test_load_teacher_run_uses_executed_actions_and_skips_reflex_only(tmp_path):
    run = _write_run(tmp_path, "jev_teacher_collect-a", [_row(0)])
    x, y, sx, sy = load_teacher_run(run)
    assert len(x) == 2
    assert [ACTIONS[i] for i in y] == ["attack", "split"]
    assert len(sx) == 1 and sy == [0]


def test_supervised_fit_learns_simple_mapping_and_stim():
    m = SemanticMLP(seed=1)
    x = []
    y = []
    sx = []
    sy = []
    for i in range(300):
        g = {"baneling_pressure": 0.9 if i % 2 else 0.1}
        local = {"personal_danger": 0.9 if i % 2 else 0.1}
        x.append(m.vector(g, local))
        y.append(ACTIONS.index("kite" if i % 2 else "attack"))
        sx.append(m.global_vector(g))
        sy.append(i % 2)
    m.fit_supervised(x, y, sx, sy, epochs=80, learning_rate=0.08, balance_power=0.0, seed=2)
    metrics = m.supervised_metrics(x, y, sx, sy)
    assert metrics["action_accuracy"] > 0.95
    assert metrics["stim_accuracy"] > 0.95


def test_clone_splits_by_run_and_saves_weights(tmp_path):
    for i in range(5):
        _write_run(
            tmp_path,
            f"jev_teacher_collect-{i}",
            [_row(j, action1="attack", action2="split", stim=(j % 2 == 0)) for j in range(20)],
        )
    weights = tmp_path / "model.json"
    metrics_file = tmp_path / "metrics.json"
    metrics = train_behavior_clone(
        tmp_path,
        weights,
        metrics_file,
        epochs=5,
        val_fraction=0.2,
        seed=1,
    )
    assert metrics["teacher_runs"] == 5
    assert metrics["validation_runs"] == 1
    assert weights.exists() and metrics_file.exists()
