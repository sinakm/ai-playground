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
    assert first.kind == "move"
    assert first.x < 0.0  # away from baneling at x=6
    assert first.y < 0.0  # away from marine 2 at y=2
    assert math.isclose(math.hypot(first.x, first.y), config.SPREAD_STEP)


def test_empty_sides_and_unknown_action():
    assert plan_orders("attack", [], E) == []
    assert plan_orders("attack", M, []) == []
    with pytest.raises(ValueError):
        plan_orders("dance", M, E)
