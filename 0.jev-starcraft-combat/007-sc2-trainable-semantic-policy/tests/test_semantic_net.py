"""Episode 006 semantic hidden layer. No SC2, no network."""

from types import SimpleNamespace

from arena import config
from arena.jev import JevSemanticClient
from arena.policies import JevSemanticNet, make_policy


class FakeSemantic:
    def __init__(self):
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        if "baneling_pressure" in questions:
            answers = {
                key: SimpleNamespace(
                    noul=0.8 if key in ("baneling_pressure", "clumping_danger") else 0.2
                )
                for key in questions
            }
            usage = SimpleNamespace(input_tokens=1000, output_tokens=0)
        else:
            answers = {
                key: SimpleNamespace(
                    choice="stutter", probabilities={"stutter": 0.9}, confidence=0.9
                )
                for key in questions
            }
            usage = SimpleNamespace(input_tokens=2000, output_tokens=2)
        return SimpleNamespace(answers=answers, usage=usage, model="jev-test")


def state():
    return {
        "marines": [{"id": 1}, {"id": 2}],
        "squad": {"marines_alive": 2},
        "priority_candidates": [],
    }


def test_semantic_client_two_layers():
    fake = FakeSemantic()
    a = JevSemanticClient(client=fake).ask(state(), [1, 2])
    assert len(fake.calls) == 2
    assert set(fake.calls[0][1]) == set(config.SEMANTIC_PERCEPTIONS)
    assert "semantic_activations" not in fake.calls[0][0]
    assert fake.calls[1][0]["semantic_activations"] == a.activations
    assert a.activations["baneling_pressure"] == 0.8
    assert a.activations["encirclement_risk"] == 0.2
    assert a.actions == {1: "stutter", 2: "stutter"}
    assert a.input_tokens == 3000


def test_semantic_policy_has_no_discrete_plan():
    fake = FakeSemantic()
    p = make_policy("jev_semantic_net", seed=0, jev_client=JevSemanticClient(client=fake))
    d = p.decide(state())
    assert isinstance(p, JevSemanticNet)
    assert d.squad_plan is None
    assert d.priority_target is None
    assert d.semantic_activations["clumping_danger"] == 0.8
    assert d.marine_actions == {1: "stutter", 2: "stutter"}
