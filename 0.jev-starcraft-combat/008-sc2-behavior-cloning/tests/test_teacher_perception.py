from types import SimpleNamespace

from arena.jev import JevPerceptionClient
from arena.policies import JevTeacherCollect


class FakeRawPerception:
    def __init__(self):
        self.questions = None

    def system_one(self, state, questions):
        self.questions = questions
        answers = {}
        for key in questions:
            if key == "semantic_priority_target":
                choice = next(iter(questions[key].criteria))
                answers[key] = SimpleNamespace(choice=choice)
            else:
                answers[key] = SimpleNamespace(noul=0.7)
        return SimpleNamespace(
            answers=answers,
            usage=SimpleNamespace(input_tokens=10, output_tokens=1),
            model="jev-test",
        )


def test_perception_call_returns_global_local_and_referential_target():
    raw = FakeRawPerception()
    state = {
        "priority_candidates": [
            {"id": 20, "distance_to_squad_center": 2.0, "nearest_marine_distance": 1.0}
        ]
    }
    a = JevPerceptionClient(client=raw).ask(state, [1, 2])
    assert a.priority_target == 20
    assert a.global_activations["baneling_pressure"] == 0.7
    assert a.local_activations[1]["personal_danger"] == 0.7
    assert "semantic_priority_target" in raw.questions


class FakePerception:
    def ask(self, state, tags):
        return SimpleNamespace(
            global_activations={"baneling_pressure": 0.8},
            local_activations={tag: {"personal_danger": 0.2} for tag in tags},
            priority_target=21,
            latency_ms=10.0,
            input_tokens=100,
            output_tokens=1,
            model="jev-perception",
        )


class FakeCommander:
    def ask(self, state, tags, previous_plan=None, after_contact=False, soldier_actions=None):
        return SimpleNamespace(
            plan="focus_banes",
            plan_confidence=0.9,
            plan_raw="focus_banes",
            plan_kept_low_confidence=False,
            stim_now=True,
            target_tag=20,
            actions={tag: "focus_bane" for tag in tags},
            confidences={tag: 0.9 for tag in tags},
            commander_latency_ms=20.0,
            soldier_latency_ms=30.0,
            input_tokens=200,
            output_tokens=2,
            model="jev-commander",
        )


def test_teacher_acts_with_commander_but_logs_perception_target():
    p = JevTeacherCollect(commander=FakeCommander(), perception=FakePerception())
    d = p.decide({
        "marines": [{"id": 1}, {"id": 2}],
        "summary": {"nearest_baneling_distance": 3.0},
    })
    assert d.priority_target == 20
    assert d.perception_target == 21
    assert d.marine_actions == {1: "focus_bane", 2: "focus_bane"}
    assert d.stim_now is True
    assert d.latency_ms == 60.0
