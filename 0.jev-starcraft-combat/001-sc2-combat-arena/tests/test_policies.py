from types import SimpleNamespace

import pytest

from arena import config
from arena.jev import QUESTION_KEY, JevSquadClient
from arena.policies import POLICY_NAMES, AttackMove, JevSquad, RandomPolicy, make_policy


class FakeTypeSafe:
    def __init__(self):
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        answer = SimpleNamespace(choice="spread", probabilities={"spread": 0.9, "attack": 0.1}, confidence=0.88)
        return SimpleNamespace(
            answers={QUESTION_KEY: answer},
            usage=SimpleNamespace(input_tokens=1469, output_tokens=56),
            model="jev-test",
        )


def test_attack_move_always_attacks():
    assert AttackMove().decide({}).action == "attack"


def test_random_is_seeded_and_valid():
    a = [RandomPolicy(seed=3).decide({}).action for _ in range(1)]
    b = [RandomPolicy(seed=3).decide({}).action for _ in range(1)]
    assert a == b
    p = RandomPolicy(seed=5)
    assert all(p.decide({}).action in config.ACTIONS for _ in range(50))


def test_jev_squad_passes_state_and_question():
    fake = FakeTypeSafe()
    d = JevSquad(JevSquadClient(client=fake)).decide({"k": 1})
    state, questions = fake.calls[0]
    assert state == {"k": 1}
    assert set(questions[QUESTION_KEY].criteria) == set(config.ACTIONS)
    assert (d.action, d.confidence, d.input_tokens, d.output_tokens, d.model) == ("spread", 0.88, 1469, 56, "jev-test")
    assert d.probabilities == {"spread": 0.9, "attack": 0.1}
    assert d.latency_ms >= 0.0


def test_make_policy():
    assert make_policy("attack_move", seed=0).name == "attack_move"
    assert make_policy("random", seed=0).name == "random"
    assert make_policy("jev_squad", seed=0, jev_client=JevSquadClient(client=FakeTypeSafe())).name == "jev_squad"
    assert POLICY_NAMES == ("attack_move", "random", "jev_squad")
    with pytest.raises(ValueError):
        make_policy("nope", seed=0)
