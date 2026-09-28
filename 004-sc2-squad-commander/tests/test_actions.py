import math

import pytest

from arena import config
from arena.actions import Order, plan_orders
from arena.views import UnitView

M = [UnitView(1, "marine", 0.0, 0.0, 45.0), UnitView(2, "marine", 0.0, 2.0, 15.0)]
E = [UnitView(10, "baneling", 6.0, 0.0, 30.0), UnitView(11, "zergling", 6.0, 2.0, 35.0)]


def test_attack_targets_enemy_centroid():
    assert plan_orders("attack", M, E) == [Order(1, "attack", 6.0, 1.0), Order(2, "attack", 6.0, 1.0)]


def test_clump_moves_to_squad_centroid():
    assert plan_orders("clump", M, E) == [Order(1, "move", 0.0, 1.0), Order(2, "move", 0.0, 1.0)]


def test_retreat_moves_away_from_enemy_centroid():
    o = plan_orders("retreat", M, E)[0]
    assert o.kind == "move"
    assert o.x < 0.0
    assert math.isclose(math.hypot(o.x - 0.0, o.y - 0.0), config.RETREAT_DISTANCE)


def test_stim_skips_low_hp_and_already_stimmed():
    marines = M + [UnitView(3, "marine", 1.0, 1.0, 45.0, stimmed=True)]
    assert plan_orders("stim", marines, E) == [Order(1, "stim")]


def test_spread_moves_away_from_baneling_and_neighbor():
    orders = plan_orders("spread", M, E)
    first = orders[0]
    assert first.kind == "kite"
    assert first.x < 0.0  # away from baneling at x=6
    assert first.y < 0.0  # away from marine 2 at y=2
    assert math.isclose(math.hypot(first.x, first.y), config.SPREAD_STEP)
    assert (first.tx, first.ty) == (6.0, 1.0)  # enemy centroid


def test_empty_sides_and_unknown_action():
    assert plan_orders("attack", [], E) == []
    assert plan_orders("attack", M, []) == []
    with pytest.raises(ValueError):
        plan_orders("dance", M, E)


# --- per-marine orders ---

from arena.actions import plan_marine_orders  # noqa: E402

PM = [
    UnitView(1, "marine", 0.0, 0.0, 45.0),
    UnitView(2, "marine", 0.0, 0.5, 15.0),
    UnitView(3, "marine", 0.0, -5.0, 45.0, stimmed=True),
]
PE = [UnitView(10, "baneling", 2.0, 0.0, 30.0), UnitView(11, "zergling", 6.0, 2.0, 35.0)]
PE_CENTROID = (4.0, 1.0)


def test_marine_actions_config():
    assert list(config.JEV_MARINE_ACTIONS) == ["kite", "split", "attack", "stim", "retreat"]
    assert config.MARINE_INSTRUCTIONS_TEMPLATE.format(tag=7).startswith("Best action for the marine with id 7,")


def test_marine_kite_steps_away_from_nearest_baneling_then_attacks():
    [o] = plan_marine_orders({1: "kite"}, PM, PE)
    assert o.kind == "kite" and o.unit_id == 1
    assert math.isclose(o.x, -config.KITE_STEP) and math.isclose(o.y, 0.0, abs_tol=1e-9)
    assert (o.tx, o.ty) == PE_CENTROID


def test_marine_split_steps_away_from_nearest_marine_then_attacks():
    [o] = plan_marine_orders({1: "split"}, PM, PE)
    assert o.kind == "kite"
    assert math.isclose(o.x, 0.0, abs_tol=1e-9) and math.isclose(o.y, -config.SPLIT_STEP)
    assert (o.tx, o.ty) == PE_CENTROID


def test_marine_attack_targets_nearest_enemy():
    assert plan_marine_orders({3: "attack"}, PM, PE) == [Order(3, "attack", 2.0, 0.0)]


def test_marine_stim_only_if_unstimmed_and_healthy():
    assert plan_marine_orders({1: "stim", 2: "stim", 3: "stim"}, PM, PE) == [Order(1, "stim")]


def test_marine_retreat_moves_away_from_enemy_centroid():
    [o] = plan_marine_orders({1: "retreat"}, PM, PE)
    assert o.kind == "move"
    assert o.x < 0.0
    assert math.isclose(math.hypot(o.x, o.y), config.MARINE_RETREAT_DISTANCE)


def test_marine_orders_mixed_and_unknown():
    orders = plan_marine_orders({1: "attack", 2: "retreat", 3: "kite"}, PM, PE)
    assert [o.unit_id for o in orders] == [1, 2, 3]
    assert plan_marine_orders({1: "attack"}, PM, []) == []
    assert plan_marine_orders({99: "attack"}, PM, PE) == []
    with pytest.raises(ValueError):
        plan_marine_orders({1: "dance"}, PM, PE)


def test_marine_split_alone_falls_back_to_attack():
    [o] = plan_marine_orders({1: "split"}, [PM[0]], PE)
    assert o.kind == "attack"
