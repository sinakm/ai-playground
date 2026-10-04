import json
import numpy as np
from types import SimpleNamespace

from arena.distill import (
    ACTIONS, GLOBAL_KEYS, LOCAL_KEYS, RELATIONAL_KEYS,
    RelationalSemanticMLP, train_behavior_clone,
)
from arena.policies import JevTeacherCollector


def _g(v=0.2):
    return {k: v for k in GLOBAL_KEYS}


def _l(v=0.2):
    return {k: v for k in LOCAL_KEYS}


def _r(v=0.2):
    return {k: v for k in RELATIONAL_KEYS}


def test_relational_shapes_and_save_load(tmp_path):
    m = RelationalSemanticMLP(seed=1)
    x = m.vector(_g(), _l(), _r())
    assert x.shape == (len(GLOBAL_KEYS) + len(LOCAL_KEYS) + len(RELATIONAL_KEYS),)
    p = m.action_probs(x)
    assert p.shape == (len(ACTIONS),) and np.isclose(p.sum(), 1.0)
    out = tmp_path / "m.json"
    m.save(out)
    n = RelationalSemanticMLP(seed=9)
    n.load(out)
    assert np.allclose(m.w1, n.w1) and np.allclose(m.ws, n.ws)


def test_behavior_clone_can_learn_relational_decision_boundary(tmp_path):
    data = tmp_path / "teacher.jsonl"
    rows = []
    # Relation, not local danger, determines whether to focus the priority Baneling.
    for seed in range(6):
        for step in range(30):
            focus = step % 2 == 1
            rel = {**_r(0.1), "target_engagement_value": 0.95 if focus else 0.05}
            rows.append({
                "episode_seed": seed,
                "global_activations": _g(0.3),
                "local_activations": {"1": _l(0.4)},
                "relational_activations": {"1": rel},
                "stim_now": step % 10 == 0,
                "executed_actions": {"1": "focus_bane" if focus else "attack"},
                "perception_priority_target": 20,
                "teacher_priority_target": 20,
            })
    data.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    out = tmp_path / "weights.json"
    metrics = train_behavior_clone(
        data, out, epochs=180, learning_rate=0.08, batch_size=32, seed=3
    )
    assert out.exists()
    assert metrics["validation"]["action"]["accuracy"] > 0.9
    assert metrics["validation"]["action"]["per_class"]["focus_bane"]["recall"] > 0.9
    assert metrics["priority_target_agreement"] == 1.0


class FakePerception:
    def ask(self, state, tags):
        return SimpleNamespace(
            global_activations=_g(0.4),
            local_activations={tag: _l(0.6) for tag in tags},
            relational_activations={tag: _r(0.7) for tag in tags},
            priority_target=20,
            node_latency_ms=1.0,
            edge_latency_ms=2.0,
            latency_ms=3.0,
            input_tokens=10,
            output_tokens=1,
            model="jev-test",
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


def test_teacher_collector_writes_relational_executed_label(tmp_path):
    path = tmp_path / "teacher.jsonl"
    p = JevTeacherCollector(
        seed=77, dataset_path=str(path),
        commander_client=FakeCommander(), perception_client=FakePerception(),
    )
    d = p.decide({"marines": [{"id": 1}], "summary": {}, "priority_candidates": []})
    assert d.marine_actions == {1: "focus_bane"} and d.stim_now is True
    assert d.relational_activations[1]["target_engagement_value"] == 0.7
    p.observe_executed({1: "kite"})
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["episode_seed"] == 77
    assert row["executed_actions"]["1"] == "kite"
    assert row["relational_activations"]["1"]["cover_effectiveness"] == 0.7
    assert row["perception_priority_target"] == 20
