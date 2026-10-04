"""Episode 004: commander plan, shared blackboard, support actions. No SC2, no network."""

import math
from types import SimpleNamespace

import pytest

from arena import config
from arena.actions import Order, plan_marine_orders, resolve_bait
from arena.jev import JevCommanderClient, bane_key
from arena.overlay import POLICY_LABELS, PANEL_SIZE, commander_header, cumulative, panel_frame
from arena.policies import POLICY_NAMES, JevCommander, RandomMarine, make_policy
from arena.state import Blackboard, build_commander_state
from arena.views import UnitView


def marine(tag, x, y, hp=45.0, stimmed=False):
    return UnitView(tag, "marine", x, y, hp, stimmed)


def bane(tag, x, y):
    return UnitView(tag, "baneling", x, y, 30.0)


def ling(tag, x, y):
    return UnitView(tag, "zergling", x, y, 35.0)


# ---------- config ----------


def test_commander_config():
    assert list(config.MARINE_ACTIONS) == [
        "kite", "split", "attack", "stim", "retreat", "focus_bane", "cover_ally", "bait", "stutter",
    ]
    assert config.SQUAD_PLANS == {
        "focus_banes": "banelings are within 6 cells of the squad: everyone shoots the priority baneling",
        "bait_and_split": "banelings are grouped and heading at a clumped squad: one marine baits, the rest spread",
        "pre_split": "banelings are within 12 cells of the squad and marines are clumped: spread out before they arrive",
        "hold_and_shoot": "no baneling within 6 cells: hold ground and shoot the nearest enemy",
        "fall_back": "more than half the squad is under 20 HP and banelings are close: pull back together",
    }
    assert config.MARINE_ACTIONS["bait"].startswith("the plan is bait_and_split")
    assert (config.BAIT_DISTANCE, config.COVER_RADIUS, config.LOW_HP, config.PRIORITY_TARGET_CANDIDATES) == (
        5.0, 4.0, 15, 5,
    )


# ---------- state / blackboard ----------

M4 = [marine(1, 0.0, 0.0), marine(2, 1.0, 0.0, hp=10.0), marine(3, 0.0, 2.0, stimmed=True), marine(4, 10.0, 0.0)]
E3 = [bane(20, 3.0, 0.0), bane(21, 20.0, 0.0), ling(30, 5.0, 5.0)]


def test_blackboard_first_step_defaults():
    s = build_commander_state(M4, E3, fight_loop=0, memory=Blackboard())
    m1 = s["marines"][0]
    assert m1["id"] == 1 and m1["last_action"] is None and m1["hp_lost_last_step"] == 0
    assert s["squad"] == {"marines_alive": 4, "stimmed_count": 1, "low_hp_count": 1, "zerg_killed_last_step": 0}


def test_blackboard_last_action_and_hp_lost():
    mem = Blackboard()
    mem.record(M4, E3, {1: "kite", 2: "attack", 3: "stim"})
    now = [marine(1, 0.0, 0.0, hp=38.0), marine(2, 1.0, 0.0, hp=10.0), marine(3, 0.0, 2.0, stimmed=True)]
    s = build_commander_state(now, E3[:2], fight_loop=12, memory=mem)
    by_id = {m["id"]: m for m in s["marines"]}
    assert by_id[1]["last_action"] == "kite" and by_id[1]["hp_lost_last_step"] == 7
    assert by_id[2]["hp_lost_last_step"] == 0
    assert by_id[3]["last_action"] == "stim"
    assert s["squad"]["zerg_killed_last_step"] == 1
    mate1 = next(t for t in by_id[2]["teammates"] if t["id"] == 1)
    assert mate1["hp_lost_last_step"] == 7 and mate1["last_action"] == "kite"
    assert s["squad"]["marines_alive"] == 3


def test_blackboard_marine_fields_and_teammates():
    s = build_commander_state(M4, E3, fight_loop=0, memory=Blackboard())
    m1 = s["marines"][0]
    assert set(m1) == {
        "id", "hp", "stimmed", "last_action", "hp_lost_last_step",
        "nearest_baneling_distance", "nearest_marine_distance", "nearest_enemy_distance", "teammates",
        "is_closest_to_banelings", "banelings_within_3", "zerglings_within_2",
    }
    assert m1["nearest_baneling_distance"] == 3.0
    assert m1["nearest_marine_distance"] == 1.0
    # 3 nearest teammates, nearest first
    assert [t["id"] for t in m1["teammates"]] == [2, 3, 4]
    t2 = m1["teammates"][0]
    assert set(t2) == {"id", "hp", "last_action", "baneling_within_3", "distance", "hp_lost_last_step",
                      "zerglings_within_2"}
    assert t2["distance"] == 1.0 and t2["hp_lost_last_step"] == 0
    assert m1["teammates"][2]["distance"] == 10.0
    assert t2["hp"] == 10 and t2["baneling_within_3"] is True  # bane 20 at 2.0 cells from marine 2
    assert m1["teammates"][2]["baneling_within_3"] is False


def test_priority_candidates_nearest_to_center_capped():
    ms = [marine(1, 0.0, 0.0), marine(2, 2.0, 0.0)]
    banes = [bane(100 + i, 1.0 + i * 2.0, 0.0) for i in range(7)]
    s = build_commander_state(ms, banes, fight_loop=0, memory=Blackboard())
    c = s["priority_candidates"]
    assert [b["id"] for b in c] == [100, 101, 102, 103, 104]
    assert c[1] == {"id": 101, "distance_to_squad_center": 2.0, "nearest_marine_distance": 1.0}


def test_no_banelings_no_candidates():
    s = build_commander_state([marine(1, 0.0, 0.0)], [ling(30, 5.0, 0.0)], fight_loop=0, memory=Blackboard())
    assert s["priority_candidates"] == []


# ---------- actions ----------

SQ = [marine(1, 0.0, 0.0), marine(2, 2.0, 0.0, hp=10.0), marine(3, 4.0, 0.0), marine(4, 20.0, 20.0)]
EN = [bane(20, 8.0, 0.0), bane(21, -6.0, 0.0), ling(30, 2.0, 3.0), ling(31, 30.0, 30.0)]


def test_one_bait_per_step_closest_to_banelings_wins():
    resolved = resolve_bait({1: "bait", 3: "bait", 2: "attack"}, SQ, EN)
    # marine 3 at x=4 is 4 cells from bane 20; marine 1 is 6 from bane 21
    assert resolved == {1: "kite", 3: "bait", 2: "attack"}
    orders = {o.unit_id: o for o in plan_marine_orders({1: "bait", 3: "bait"}, SQ, EN)}
    assert orders[1].kind == "kite"
    assert orders[3].kind == "move"


def test_bait_moves_away_from_squad_center():
    ms = [marine(1, 0.0, 0.0), marine(2, 2.0, 0.0), marine(3, 4.0, 0.0)]  # center (2, 0)
    [o] = plan_marine_orders({3: "bait"}, ms, EN)
    assert o.kind == "move"
    assert math.isclose(o.x, 4.0 + config.BAIT_DISTANCE) and math.isclose(o.y, 0.0, abs_tol=1e-9)


def test_cover_ally_shoots_enemy_nearest_weakest_teammate_in_radius():
    [o] = plan_marine_orders({1: "cover_ally"}, SQ, EN)
    # weakest teammate within 4 cells of marine 1 is marine 2 (hp 10); nearest enemy to it is zergling 30
    assert o == Order(1, "attack_unit", 2.0, 3.0, target_id=30)


def test_cover_ally_without_teammate_in_radius_attacks_nearest():
    [o] = plan_marine_orders({4: "cover_ally"}, SQ, EN)
    assert o == Order(4, "attack", 30.0, 30.0)


def test_focus_bane_uses_priority_target():
    # priority target within FOCUS_RANGE of the marine (the range guard is tested in test_round4d)
    ms = [marine(1, 0.0, 0.0), marine(2, 2.0, 0.0)]
    es = [bane(20, 4.0, 0.0), bane(21, -3.0, 0.0)]
    [o] = plan_marine_orders({1: "focus_bane"}, ms, es, priority_target=21)
    assert o == Order(1, "attack_unit", -3.0, 0.0, target_id=21)


def test_focus_bane_falls_back_to_bane_nearest_squad_center():
    # squad center (1, 0): bane 20 at (4,0) is nearer than bane 21 at (-3.5,0)
    ms = [marine(1, 0.0, 0.0), marine(2, 2.0, 0.0)]
    es = [bane(20, 4.0, 0.0), bane(21, -3.5, 0.0)]
    [o] = plan_marine_orders({1: "focus_bane"}, ms, es, priority_target=999)
    assert o.target_id == 20
    [o] = plan_marine_orders({1: "focus_bane"}, ms, es)
    assert o.target_id == 20


def test_focus_bane_without_banelings_attacks_nearest():
    [o] = plan_marine_orders({1: "focus_bane"}, SQ, [ling(30, 2.0, 3.0)])
    assert o == Order(1, "attack", 2.0, 3.0)


# ---------- jev commander client ----------


class FakeCommander:
    """Call 1 answers squad_plan (+ priority_target); call 2 answers every marine_<tag>."""

    def __init__(self, plan="focus_banes", target=None, marine_choice="focus_bane"):
        self.plan, self.target, self.marine_choice = plan, target, marine_choice
        self.calls = []
        self.in_flight = 0
        self.max_in_flight = 0

    def system_one(self, state, questions):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.calls.append((state, questions))
        answers = {}
        if "squad_plan" in questions:
            answers["squad_plan"] = SimpleNamespace(
                choice=self.plan, probabilities={self.plan: 0.8}, confidence=0.8
            )
            if "priority_target" in questions:
                t = self.target or next(iter(questions["priority_target"].criteria))
                answers["priority_target"] = SimpleNamespace(choice=t, probabilities={t: 0.6}, confidence=0.6)
            usage = SimpleNamespace(input_tokens=1000, output_tokens=2)
        else:
            for key in questions:
                answers[key] = SimpleNamespace(
                    choice=self.marine_choice, probabilities={self.marine_choice: 0.7}, confidence=0.7
                )
            usage = SimpleNamespace(input_tokens=3000, output_tokens=5)
        self.in_flight -= 1
        return SimpleNamespace(answers=answers, usage=usage, model="jev-test")


def commander_state():
    return build_commander_state(M4, E3, fight_loop=0, memory=Blackboard())


def test_commander_client_two_sequential_calls_plan_in_second_state():
    fake = FakeCommander(target=bane_key(21))
    a = JevCommanderClient(client=fake).ask(commander_state(), [1, 2, 3, 4])
    assert len(fake.calls) == 2 and fake.max_in_flight == 1
    s1, q1 = fake.calls[0]
    assert set(q1) == {"squad_plan", "priority_target", "stim_now"}
    assert dict(q1["squad_plan"].criteria) == config.SQUAD_PLANS
    assert "squad_plan" not in s1
    s2, q2 = fake.calls[1]
    assert s2["squad_plan"] == "focus_banes"
    assert s2["priority_target"] == "bane_21"
    assert list(q2) == ["marine_1", "marine_2", "marine_3", "marine_4"]
    assert dict(q2["marine_1"].criteria) == config.COMMANDER_MARINE_ACTIONS
    assert q2["marine_1"].instructions == config.COMMANDER_MARINE_INSTRUCTIONS_TEMPLATE.format(tag=1)
    assert a.plan == "focus_banes" and a.target_tag == 21
    assert a.actions == {1: "focus_bane", 2: "focus_bane", 3: "focus_bane", 4: "focus_bane"}
    assert a.confidences[1] == 0.7
    assert (a.input_tokens, a.output_tokens, a.model) == (4000, 7, "jev-test")
    assert a.commander_latency_ms >= 0.0 and a.soldier_latency_ms >= 0.0


def test_commander_priority_target_options_are_dynamic():
    fake = FakeCommander()
    JevCommanderClient(client=fake).ask(commander_state(), [1])
    opts = dict(fake.calls[0][1]["priority_target"].criteria)
    # bane 20 at (3,0), bane 21 at (20,0); squad center (2.75, 0.5)
    assert list(opts) == ["bane_20", "bane_21"]
    assert opts["bane_20"] == "baneling 20: 0.6 cells from squad center, nearest marine 2.0 cells"


def test_commander_without_banelings_omits_priority_target():
    fake = FakeCommander(plan="hold_and_shoot", marine_choice="attack")
    s = build_commander_state(M4, [ling(30, 5.0, 5.0)], fight_loop=0, memory=Blackboard())
    a = JevCommanderClient(client=fake).ask(s, [1, 2])
    assert set(fake.calls[0][1]) == {"squad_plan", "stim_now"}
    assert fake.calls[1][0]["priority_target"] is None
    assert a.target_tag is None and a.plan == "hold_and_shoot"


# ---------- policies ----------


def test_jev_commander_policy_decision():
    fake = FakeCommander(target=bane_key(20))
    p = make_policy("jev_commander", seed=0, jev_client=JevCommanderClient(client=fake))
    assert isinstance(p, JevCommander) and p.name == "jev_commander" and p.uses_blackboard
    d = p.decide(commander_state())
    assert d.squad_plan == "focus_banes" and d.priority_target == 20
    assert d.action == "focus_bane"
    assert d.latency_ms == pytest.approx(d.commander_latency_ms + d.soldier_latency_ms)
    assert set(d.probabilities) == set(config.MARINE_ACTIONS)
    assert d.input_tokens == 4000


def test_jev_commander_warmup_two_calls():
    fake = FakeCommander()
    JevCommander(JevCommanderClient(client=fake)).warmup({"note": "warmup"})
    assert len(fake.calls) == 2


def test_random_covers_all_nine_actions():
    p = RandomMarine(seed=1)
    seen = set()
    for _ in range(50):
        seen |= set(p.decide({"marines": [{"id": i} for i in range(12)]}).marine_actions.values())
    assert seen == set(config.MARINE_ACTIONS)


def test_policy_names():
    assert POLICY_NAMES == ("attack_move", "random", "stutter_all", "jev_commander", "jev_commander_stutter", "jev_semantic_net", "jev_trainable_semantic", "jev_relational_teacher", "jev_relational_student")


# ---------- overlay ----------

COMMANDER_REC = {
    "policy": "jev_commander", "action": "focus_bane",
    "probabilities": {a: 0.0 for a in config.MARINE_ACTIONS} | {"focus_bane": 0.75, "bait": 0.25},
    "confidence": 0.7, "latency_ms": 310.0, "commander_latency_ms": 150.0, "soldier_latency_ms": 160.0,
    "input_tokens": 4000, "output_tokens": 7, "marines_alive": 4, "enemies_alive": 12, "late": False,
    "marine_actions": {"1": "focus_bane", "2": "focus_bane", "3": "focus_bane", "4": "bait"},
    "marine_confidences": {"1": 0.7, "2": 0.7, "3": 0.7, "4": 0.7},
    "squad_plan": "bait_and_split", "priority_target": 20,
}


def test_commander_header_and_panel():
    assert commander_header(COMMANDER_REC) == "COMMANDER: BAIT_AND_SPLIT"
    assert commander_header({"squad_plan": None}) is None
    img = panel_frame(COMMANDER_REC, cumulative([COMMANDER_REC])[0], header="JEV COMMANDER")
    assert img.size == PANEL_SIZE


def test_commander_labels():
    header, title, subtitle = POLICY_LABELS["jev_commander"]
    assert title == "Jev vs Banelings, round 4: a squad with a commander"
    assert "9 actions" in POLICY_LABELS["random"][2]
