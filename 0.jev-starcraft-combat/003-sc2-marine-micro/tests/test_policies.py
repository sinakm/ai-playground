from types import SimpleNamespace

import pytest

from arena import config
from arena.jev import QUESTION_KEY, JevMarineClient, JevSquadClient
from arena.policies import POLICY_NAMES, AttackMove, JevMarine, JevSquad, RandomMarine, make_policy


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


class FakePerMarine:
    """Answers each `marine_<tag>` question from a {tag: (choice, confidence)} script."""

    def __init__(self, script: dict[int, tuple[str, float]]):
        self.script = script
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        answers = {}
        for key in questions:
            choice, conf = self.script[int(key.removeprefix("marine_"))]
            answers[key] = SimpleNamespace(choice=choice, probabilities={choice: conf}, confidence=conf)
        return SimpleNamespace(
            answers=answers, usage=SimpleNamespace(input_tokens=3400, output_tokens=0), model="jev-test"
        )


def marine_state(tags):
    return {"marines": [{"id": t, "x": 0.0, "y": 0.0, "hp": 45, "stimmed": False} for t in tags]}


def test_attack_move_always_attacks():
    d = AttackMove().decide({})
    assert d.action == "attack"
    assert d.marine_actions is None


def test_random_marine_is_seeded_per_marine_and_valid():
    s = marine_state([5, 6, 7])
    a = RandomMarine(seed=3).decide(s)
    b = RandomMarine(seed=3).decide(s)
    assert a.marine_actions == b.marine_actions
    assert set(a.marine_actions) == {5, 6, 7}
    p = RandomMarine(seed=5)
    for _ in range(30):
        d = p.decide(s)
        assert all(v in config.MARINE_ACTIONS for v in d.marine_actions.values())
        assert d.action in config.MARINE_ACTIONS
        assert sum(d.probabilities.values()) == pytest.approx(1.0)


def test_jev_squad_passes_state_and_question():
    fake = FakeTypeSafe()
    d = JevSquad(JevSquadClient(client=fake)).decide({"k": 1})
    state, questions = fake.calls[0]
    assert state == {"k": 1}
    assert set(questions[QUESTION_KEY].criteria) == set(config.ACTIONS)
    assert (d.action, d.confidence, d.input_tokens, d.output_tokens, d.model) == ("spread", 0.88, 1469, 56, "jev-test")
    assert d.probabilities == {"spread": 0.9, "attack": 0.1}
    assert d.latency_ms >= 0.0
    assert d.marine_actions is None


def test_jev_marine_client_one_call_one_question_per_tag():
    fake = FakePerMarine({11: ("kite", 0.9), 12: ("attack", 0.7)})
    a = JevMarineClient(client=fake).ask({"s": 1}, [11, 12])
    assert len(fake.calls) == 1
    state, questions = fake.calls[0]
    assert state == {"s": 1}
    assert list(questions) == ["marine_11", "marine_12"]
    q = questions["marine_11"]
    assert dict(q.criteria) == config.MARINE_ACTIONS
    assert q.instructions == config.MARINE_INSTRUCTIONS_TEMPLATE.format(tag=11)
    assert a.actions == {11: "kite", 12: "attack"}
    assert a.confidences == {11: 0.9, 12: 0.7}
    assert (a.input_tokens, a.output_tokens, a.model) == (3400, 0, "jev-test")
    assert a.latency_ms >= 0.0


def test_jev_marine_aggregates_most_common_mean_and_shares():
    fake = FakePerMarine({1: ("kite", 0.9), 2: ("kite", 0.5), 3: ("attack", 0.7), 4: ("stim", 0.5)})
    d = JevMarine(JevMarineClient(client=fake)).decide(marine_state([1, 2, 3, 4]))
    assert d.action == "kite"
    assert d.confidence == pytest.approx(0.65)
    assert d.probabilities == {"kite": 0.5, "split": 0.0, "attack": 0.25, "stim": 0.25, "retreat": 0.0}
    assert d.marine_actions == {1: "kite", 2: "kite", 3: "attack", 4: "stim"}
    assert d.marine_confidences == {1: 0.9, 2: 0.5, 3: 0.7, 4: 0.5}
    assert (d.input_tokens, d.model) == (3400, "jev-test")
    assert list(fake.calls[0][1]) == ["marine_1", "marine_2", "marine_3", "marine_4"]


def test_jev_marine_tie_uses_action_order():
    fake = FakePerMarine({1: ("retreat", 0.6), 2: ("attack", 0.6), 3: ("split", 0.6), 4: ("retreat", 0.6), 5: ("split", 0.6)})
    d = JevMarine(JevMarineClient(client=fake)).decide(marine_state([1, 2, 3, 4, 5]))
    assert d.action == "split"  # split and retreat tie at 2; split comes first in MARINE_ACTIONS


def test_jev_marine_warmup_uses_one_marine():
    fake = FakePerMarine({1: ("attack", 0.5)})
    JevMarine(JevMarineClient(client=fake)).warmup({"note": "warmup"})
    state, questions = fake.calls[0]
    assert list(questions) == ["marine_1"]
    assert [m["id"] for m in state["marines"]] == [1]


def test_make_policy():
    assert make_policy("attack_move", seed=0).name == "attack_move"
    assert isinstance(make_policy("random", seed=0), RandomMarine)
    assert make_policy("jev_squad", seed=0, jev_client=JevSquadClient(client=FakeTypeSafe())).name == "jev_squad"
    p = make_policy("jev_marine", seed=0, jev_client=JevMarineClient(client=FakePerMarine({})))
    assert isinstance(p, JevMarine) and p.name == "jev_marine"
    assert POLICY_NAMES == ("attack_move", "random", "jev_squad", "jev_marine")
    with pytest.raises(ValueError):
        make_policy("nope", seed=0)
