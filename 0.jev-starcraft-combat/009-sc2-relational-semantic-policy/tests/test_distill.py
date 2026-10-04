import json
import numpy as np
from pathlib import Path
from types import SimpleNamespace

from arena.distill import ACTIONS, GLOBAL_KEYS, LOCAL_KEYS, DistilledSemanticMLP, train_behavior_clone
from arena.policies import JevTeacherCollector


def _g(v=0.2):
    return {k: v for k in GLOBAL_KEYS}


def _l(v=0.2):
    return {k: v for k in LOCAL_KEYS}


def test_distilled_shapes_and_save_load(tmp_path):
    m = DistilledSemanticMLP(seed=1)
    x = m.vector(_g(), _l())
    assert x.shape == (len(GLOBAL_KEYS) + len(LOCAL_KEYS),)
    p = m.action_probs(x)
    assert p.shape == (len(ACTIONS),) and np.isclose(p.sum(), 1.0)
    out = tmp_path / "m.json"
    m.save(out)
    n = DistilledSemanticMLP(seed=9); n.load(out)
    assert np.allclose(m.w1, n.w1) and np.allclose(m.ws, n.ws)


def test_behavior_clone_learns_simple_teacher(tmp_path):
    data = tmp_path / "teacher.jsonl"
    rows = []
    # Four trajectory groups. High personal danger -> kite, low -> attack.
    for seed in range(4):
        for step in range(25):
            danger = 0.9 if step % 2 else 0.1
            action = "kite" if danger > 0.5 else "attack"
            rows.append({
                "episode_seed": seed,
                "global_activations": _g(0.3),
                "local_activations": {"1": {**_l(0.2), "personal_danger": danger}},
                "stim_now": step % 5 == 0,
                "executed_actions": {"1": action},
                "perception_priority_target": None,
                "teacher_priority_target": None,
            })
    data.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    out = tmp_path / "weights.json"
    metrics = train_behavior_clone(data, out, epochs=180, learning_rate=0.08, batch_size=32, seed=3)
    assert out.exists()
    assert metrics["validation"]["action_accuracy"] > 0.9


class FakePerception:
    def ask(self, state, tags):
        return SimpleNamespace(
            global_activations=_g(0.4),
            local_activations={tag: _l(0.6) for tag in tags},
            priority_target=20,
            latency_ms=1.0, input_tokens=10, output_tokens=1, model="jev-test",
        )


class FakeCommander:
    def ask(self, state, tags, previous_plan=None, after_contact=False, soldier_actions=None):
        return SimpleNamespace(
            plan="focus_banes", plan_confidence=0.9, plan_raw="focus_banes",
            plan_kept_low_confidence=False, stim_now=True, target_tag=20,
            actions={tag: "focus_bane" for tag in tags},
            confidences={tag: 0.9 for tag in tags},
            commander_latency_ms=2.0, soldier_latency_ms=3.0,
            input_tokens=20, output_tokens=2, model="jev-test",
        )


def test_teacher_collector_writes_executed_label(tmp_path):
    path = tmp_path / "teacher.jsonl"
    p = JevTeacherCollector(
        seed=77, dataset_path=str(path),
        commander_client=FakeCommander(), perception_client=FakePerception(),
    )
    d = p.decide({"marines": [{"id": 1}], "summary": {}, "priority_candidates": []})
    assert d.marine_actions == {1: "focus_bane"} and d.stim_now is True
    p.observe_executed({1: "kite"})
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["episode_seed"] == 77
    assert row["executed_actions"]["1"] == "kite"
    assert row["perception_priority_target"] == 20


def test_class_balancing_recovers_rare_focus_signal(tmp_path):
    data = tmp_path / "teacher_rel.jsonl"
    rows = []
    for seed in range(5):
        for step in range(40):
            focus = step % 5 == 0
            local = _l(0.2)
            local["should_focus_priority_target"] = 0.95 if focus else 0.05
            local["priority_target_shootable"] = 0.9 if focus else 0.2
            rows.append({
                "episode_seed": seed,
                "global_activations": _g(0.3),
                "local_activations": {"1": local},
                "stim_now": False,
                "executed_actions": {"1": "focus_bane" if focus else "attack"},
                "perception_priority_target": 20,
                "teacher_priority_target": 20,
            })
    data.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    out = tmp_path / "rel.json"
    metrics = train_behavior_clone(
        data, out, epochs=180, learning_rate=0.08, batch_size=32, seed=4
    )
    assert metrics["action_class_weights"]["focus_bane"] > metrics["action_class_weights"]["attack"]
    assert metrics["validation"]["action_recall"]["focus_bane"] > 0.5
