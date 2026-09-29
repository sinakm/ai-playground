import pytest

from arena import config
from arena.state import build_state, centroid, distance
from arena.views import UnitView

M = [UnitView(1, "marine", 0.0, 0.0, 45.0), UnitView(2, "marine", 2.0, 0.0, 30.0, stimmed=True)]
E = [UnitView(10, "baneling", 5.0, 0.0, 30.0), UnitView(11, "zergling", 8.0, 0.0, 35.0)]


def test_centroid_and_distance():
    assert centroid(M) == (1.0, 0.0)
    assert distance(0, 0, 3, 4) == 5.0
    with pytest.raises(ValueError):
        centroid([])


def test_build_state_summary():
    s = build_state(M, E, fight_loop=224)
    assert s["game_time_s"] == 10.0
    assert s["summary"] == {
        "marines_alive": 2,
        "banelings_alive": 1,
        "zerglings_alive": 1,
        "squad_spread": 1.0,
        "nearest_baneling_distance": 3.0,
        "nearest_baneling_to_squad_center": 4.0,
        "stimmed_marines": 1,
        "marines_in_splash_danger": 1,
        "seconds_left": 50.0,
    }
    assert s["rules"] == config.STATE_RULES


def test_build_state_marines_in_splash_danger():
    close = [UnitView(1, "marine", 4.0, 0.0, 45.0), UnitView(2, "marine", 20.0, 0.0, 45.0)]
    bane = [UnitView(10, "baneling", 5.0, 0.0, 30.0)]
    s = build_state(close, bane, fight_loop=0)
    assert s["summary"]["marines_in_splash_danger"] == 1
    assert s["summary"]["seconds_left"] == 60.0


def test_build_state_no_marines_in_splash_danger():
    s = build_state(M, [], fight_loop=0)
    assert s["summary"]["marines_in_splash_danger"] == 0


def test_build_state_units():
    s = build_state(M, E, fight_loop=0)
    assert s["marines"][1] == {
        "id": 2, "x": 2.0, "y": 0.0, "hp": 30, "stimmed": True,
        "nearest_baneling_distance": 3.0, "nearest_marine_distance": 2.0, "nearest_enemy_distance": 3.0,
    }
    assert s["enemies"][0] == {"id": 10, "kind": "baneling", "x": 5.0, "y": 0.0, "hp": 30, "distance_to_squad": 4.0}


def test_build_state_no_banelings():
    s = build_state(M, [E[1]], fight_loop=0)
    assert s["summary"]["nearest_baneling_distance"] is None
    assert s["summary"]["nearest_baneling_to_squad_center"] is None
    assert s["summary"]["banelings_alive"] == 0


def test_build_state_no_marines_nearest_baneling_to_squad_center():
    s = build_state([], E, fight_loop=0)
    assert s["summary"]["nearest_baneling_to_squad_center"] is None


def test_build_state_per_marine_distances_none_when_absent():
    s = build_state([M[0]], [E[1]], fight_loop=0)
    m = s["marines"][0]
    assert m["nearest_baneling_distance"] is None
    assert m["nearest_marine_distance"] is None
    assert m["nearest_enemy_distance"] == 8.0


def test_build_state_per_marine_distances_rounded():
    ms = [UnitView(1, "marine", 0.0, 0.0, 45.0), UnitView(2, "marine", 1.0, 1.0, 45.0)]
    s = build_state(ms, E, fight_loop=0)
    assert s["marines"][0]["nearest_marine_distance"] == 1.4
