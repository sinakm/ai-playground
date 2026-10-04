import pytest

from arena import config
from arena.policies import POLICY_NAMES, AttackMove, RandomMarine, make_policy


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


def test_make_policy():
    assert make_policy("attack_move", seed=0).name == "attack_move"
    assert isinstance(make_policy("random", seed=0), RandomMarine)
    assert make_policy("stutter_all", seed=0).name == "stutter_all"
    assert POLICY_NAMES == ("attack_move", "random", "stutter_all", "jev_commander", "jev_commander_stutter", "jev_semantic_net", "jev_trainable_semantic")
    with pytest.raises(ValueError):
        make_policy("nope", seed=0)
