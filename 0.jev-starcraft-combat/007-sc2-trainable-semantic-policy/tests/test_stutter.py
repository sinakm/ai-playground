"""Episode 005: stutter-step action, stutter_all baseline, commander + stutter. No SC2, no network."""

import json
import math
from types import SimpleNamespace

import pytest

from arena import config
from arena.actions import Order, execute_actions, plan_marine_orders, stutter_order, stutter_step_orders
from arena.bot import ArenaBot, stutter_tags
from arena.evaluate import format_table, summarize
from arena.jev import JevCommanderClient
from arena.overlay import POLICY_LABELS, PANEL_SIZE, _action_list, cumulative, panel_frame, stats_lines
from arena.policies import (
    POLICY_NAMES,
    Decision,
    JevCommander,
    JevCommanderStutter,
    RandomMarine,
    StutterAll,
    make_policy,
)
from arena.views import UnitView


def marine(tag, x, y, hp=45.0, stimmed=False):
    return UnitView(tag, "marine", x, y, hp, stimmed)


def bane(tag, x, y):
    return UnitView(tag, "baneling", x, y, 30.0)


def ling(tag, x, y):
    return UnitView(tag, "zergling", x, y, 35.0)


# ---------- config ----------


def test_stutter_config():
    assert config.STUTTER_STEP == 0.75
    assert config.STUTTER_THREAT_BANELING_RADIUS == 6.0
    assert config.STUTTER_ENEMY_RADIUS == 5.0
    assert config.STUTTER_CRITERION == (
        "enemies are within 5 cells and no baneling is about to reach you: "
        "shoot when your rifle is ready, step back while it reloads"
    )
    assert config.MARINE_ACTIONS["stutter"] == config.STUTTER_CRITERION
    assert "stutter" not in config.COMMANDER_MARINE_ACTIONS
    assert list(config.COMMANDER_STUTTER_MARINE_ACTIONS) == list(config.COMMANDER_MARINE_ACTIONS) + ["stutter"]


# ---------- stutter_order ----------


def test_stutter_ready_attacks_nearest_enemy_unit():
    m = marine(1, 0.0, 0.0)
    es = [ling(30, 4.0, 0.0), bane(20, 2.0, 0.0)]
    assert stutter_order(m, True, es) == Order(1, "attack_unit", 2.0, 0.0, target_id=20)


def test_stutter_cooldown_steps_away_from_baneling_within_6():
    m = marine(1, 0.0, 0.0)
    es = [ling(30, 0.0, 2.0), bane(20, 5.0, 0.0)]  # zergling nearer, but a baneling is within 6
    o = stutter_order(m, False, es)
    assert o.kind == "move" and o.unit_id == 1
    assert (o.x, o.y) == pytest.approx((-0.75, 0.0))


def test_stutter_cooldown_steps_away_from_nearest_enemy_when_no_baneling_close():
    m = marine(1, 0.0, 0.0)
    es = [ling(30, 0.0, 2.0), bane(20, 7.0, 0.0)]  # baneling beyond 6 cells
    o = stutter_order(m, False, es)
    assert o.kind == "move"
    assert (o.x, o.y) == pytest.approx((0.0, -0.75))
    assert math.hypot(o.x, o.y) == pytest.approx(config.STUTTER_STEP)


def test_stutter_cooldown_on_top_of_threat_attacks():
    o = stutter_order(marine(1, 1.0, 1.0), False, [ling(30, 1.0, 1.0)])
    assert o == Order(1, "attack_unit", 1.0, 1.0, target_id=30)


def test_stutter_no_enemies_no_order():
    assert stutter_order(marine(1, 0.0, 0.0), True, []) is None


def test_stutter_step_orders_only_living_stutterers():
    ms = [marine(1, 0.0, 0.0), marine(2, 0.0, 10.0), marine(3, 0.0, 20.0)]
    es = [ling(30, 3.0, 0.0)]
    orders = stutter_step_orders({1, 3, 99}, ms, {1: True, 2: True, 3: False}, es)
    assert [o.unit_id for o in orders] == [1, 3]
    assert orders[0].kind == "attack_unit" and orders[1].kind == "move"


def test_plan_marine_orders_leaves_stutter_to_per_step_code():
    ms = [marine(1, 0.0, 0.0), marine(2, 0.0, 5.0)]
    orders = plan_marine_orders({1: "stutter", 2: "attack"}, ms, [ling(30, 4.0, 0.0)])
    assert [o.unit_id for o in orders] == [2]


# ---------- low-confidence default and reflexes ----------


def test_low_confidence_stutter_default():
    ms = [marine(1, 0.0, 0.0), marine(2, 0.0, 20.0), marine(3, 0.0, 40.0), marine(4, 0.0, 60.0, hp=10.0)]
    es = [ling(30, 4.0, 0.0), bane(20, 2.8, 20.0), ling(31, 0.0, 49.0), ling(32, 4.9, 60.0)]
    conf = {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2}
    acts = {1: "bait", 2: "bait", 3: "bait", 4: "bait"}
    stutter = execute_actions(acts, ms, es, confidences=conf, min_confidence=0.5, stutter_default=True)
    assert stutter == {1: "stutter", 2: "kite", 3: "attack", 4: "stutter"}
    plain = execute_actions(acts, ms, es, confidences=conf, min_confidence=0.5)
    assert plain == {1: "attack", 2: "kite", 3: "attack", 4: "retreat"}


def test_reflexes_override_stutter():
    ms = [marine(1, 0.0, 0.0), marine(2, 20.0, 0.0), marine(3, 40.0, 0.0), marine(4, 60.0, 0.0, hp=10.0)]
    es = [bane(20, 2.0, 0.0), ling(30, 21.0, 0.0), ling(31, 21.0, 0.5), ling(32, 21.0, -0.5),
          ling(33, 44.0, 0.0), ling(34, 61.0, 0.0)]
    stats = {}
    ex = execute_actions({t: "stutter" for t in (1, 2, 3, 4)}, ms, es, reflex=True, stats=stats)
    assert ex == {1: "kite", 2: "retreat_to_squad", 3: "stutter", 4: "retreat_to_squad"}
    assert stats["reflex_count"] == 3


# ---------- policies ----------


def state_of(tags):
    return {"marines": [{"id": t} for t in tags]}


def test_policy_names():
    assert POLICY_NAMES == ("attack_move", "random", "stutter_all", "jev_commander", "jev_commander_stutter", "jev_semantic_net")
    with pytest.raises(ValueError):
        make_policy("jev_marine", seed=0)


def test_stutter_all_policy():
    p = make_policy("stutter_all", seed=0)
    assert isinstance(p, StutterAll) and p.name == "stutter_all"
    d = p.decide(state_of([4, 5, 6]))
    assert d.marine_actions == {4: "stutter", 5: "stutter", 6: "stutter"}
    assert d.action == "stutter" and d.probabilities["stutter"] == 1.0
    assert d.confidence is None and d.model is None


def test_random_pool_includes_stutter():
    p = RandomMarine(seed=1)
    seen = set()
    for _ in range(60):
        seen |= set(p.decide(state_of(range(12))).marine_actions.values())
    assert "stutter" in seen and seen == set(config.MARINE_ACTIONS)


class FakeCommander:
    def __init__(self, marine_choice="stutter"):
        self.marine_choice = marine_choice
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        answers = {}
        if "squad_plan" in questions:
            answers["squad_plan"] = SimpleNamespace(choice="hold_and_shoot", confidence=0.8)
            answers["stim_now"] = SimpleNamespace(noul=0.0)
        else:
            for k in questions:
                answers[k] = SimpleNamespace(choice=self.marine_choice, confidence=0.9)
        return SimpleNamespace(answers=answers, usage=SimpleNamespace(input_tokens=1, output_tokens=0), model="m")


CSTATE = {"marines": [{"id": 1}, {"id": 2}], "priority_candidates": [], "summary": {}}


def test_commander_stutter_offers_stutter():
    fake = FakeCommander()
    p = make_policy("jev_commander_stutter", seed=0, jev_client=JevCommanderClient(client=fake))
    assert isinstance(p, JevCommanderStutter) and p.name == "jev_commander_stutter"
    assert p.uses_blackboard and p.stutter_default
    d = p.decide(CSTATE)
    assert dict(fake.calls[1][1]["marine_1"].criteria) == config.COMMANDER_STUTTER_MARINE_ACTIONS
    assert d.marine_actions == {1: "stutter", 2: "stutter"} and d.action == "stutter"


def test_commander_does_not_offer_stutter():
    fake = FakeCommander(marine_choice="attack")
    p = make_policy("jev_commander", seed=0, jev_client=JevCommanderClient(client=fake))
    assert type(p) is JevCommander and not getattr(p, "stutter_default", False)
    p.decide(CSTATE)
    assert "stutter" not in dict(fake.calls[1][1]["marine_1"].criteria)


# ---------- bot: stutter set lifecycle, logging ----------


def test_stutter_tags_helper():
    assert stutter_tags({1: "stutter", 2: "attack", 3: "stutter"}) == {1, 3}
    assert stutter_tags(None) == set()


class Scripted:
    uses_blackboard = False

    def __init__(self, name, script):
        self.name = name
        self.script = list(script)

    def decide(self, state):
        acts = self.script.pop(0)
        return Decision(action="attack", marine_actions=acts)


def test_stutter_set_lifecycle_and_record(tmp_path):
    bot = ArenaBot(Scripted("random", [{1: "stutter", 2: "stutter"}, {1: "attack", 2: "stutter"}]),
                   tmp_path, seed=0, realtime=False)
    ms = [marine(1, 0.0, 0.0), marine(2, 0.0, 10.0)]
    es = [ling(30, 30.0, 0.0)]
    state, d, ex, orders = bot._plan_step(ms, es, elapsed=12)
    assert bot.stuttering == {1, 2} and orders == []
    bot._log_decision(d, ex, state, 12, 2, 1)
    state, d, ex, orders = bot._plan_step(ms, es, elapsed=24)
    assert bot.stuttering == {2} and [o.unit_id for o in orders] == [1]
    bot._log_decision(d, ex, state, 24, 2, 1)
    # Per-step stutter: dead Marines drop out, living stutterers get one order each.
    step = bot._stutter_orders([marine(2, 0.0, 10.0)], {2: False}, es)
    assert [o.unit_id for o in step] == [2] and bot.stutter_steps == 1
    bot.close()
    recs = [json.loads(line) for line in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert [r["stuttering"] for r in recs] == [2, 1]


def test_squad_decision_clears_stutter_set(tmp_path):
    class Squad:
        name = "attack_move"

        def decide(self, state):
            return Decision(action="attack")

    bot = ArenaBot(Squad(), tmp_path, seed=0, realtime=False)
    bot.stuttering = {1}
    bot._plan_step([marine(1, 0.0, 0.0)], [ling(30, 30.0, 0.0)], elapsed=12)
    bot.close()
    assert bot.stuttering == set()


def test_commander_reflex_removes_marine_from_stutter_set(tmp_path):
    class Cmd:
        name = "jev_commander_stutter"
        uses_blackboard = True
        stutter_default = True

        def decide(self, state):
            return Decision(action="stutter", marine_actions={1: "stutter", 2: "stutter"},
                            squad_plan="hold_and_shoot", model="m")

    bot = ArenaBot(Cmd(), tmp_path, seed=0, realtime=False)
    ms = [marine(1, 0.0, 0.0), marine(2, 20.0, 0.0)]
    _, _, ex, _ = bot._plan_step(ms, [bane(20, 2.0, 0.0), ling(30, 24.0, 0.0)], elapsed=12)
    bot.close()
    assert ex == {1: "kite", 2: "stutter"} and bot.stuttering == {2}


def test_summary_has_stutter_steps_and_hp(tmp_path):
    import asyncio

    bot = ArenaBot(Scripted("stutter_all", []), tmp_path, seed=0, realtime=False)
    bot.stutter_steps = 7
    asyncio.run(bot._finish(2, 0, 100, hp_alive_sum=51.0, leave=False))
    bot.close()
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["stutter_steps"] == 7 and s["hp_alive_sum"] == 51.0 and s["result"] == "win"


# ---------- overlay ----------


REC = {
    "policy": "stutter_all", "action": "stutter", "probabilities": None, "confidence": None,
    "latency_ms": 0.0, "input_tokens": 0, "output_tokens": 0, "marines_alive": 10, "enemies_alive": 5,
    "late": False, "marine_actions": {str(i): "stutter" for i in range(10)}, "stuttering": 9,
}


def test_overlay_stuttering_line():
    lines = stats_lines(REC, cumulative([REC])[0])
    assert "stuttering 9/10" in lines
    no = {k: v for k, v in REC.items() if k != "stuttering"}
    assert not any(line.startswith("stuttering") for line in stats_lines(no, cumulative([no])[0]))
    assert panel_frame(REC, cumulative([REC])[0]).size == PANEL_SIZE


def test_overlay_labels_and_action_lists():
    assert POLICY_LABELS["stutter_all"] == (
        "STUTTER ALL", "Stutter-all vs Banelings", "scripted baseline: every marine stutter-steps"
    )
    assert POLICY_LABELS["jev_commander_stutter"][0] == "COMMANDER + STUTTER"
    assert POLICY_LABELS["jev_commander_stutter"][1] == "Jev vs Banelings, round 5: stutter-step"
    assert "jev_marine" not in POLICY_LABELS
    assert "stutter" in _action_list({"policy": "jev_commander_stutter", "marine_actions": {}})
    assert "stutter" not in _action_list({"policy": "jev_commander", "marine_actions": {}})
    rec = {**REC, "policy": "jev_commander_stutter", "squad_plan": "hold_and_shoot"}
    assert panel_frame(rec, cumulative([rec])[0]).size == PANEL_SIZE


# ---------- evaluate ----------


def run(policy, marines, hp=None):
    r = {
        "policy": policy, "seed": 0, "realtime": False, "result": "win" if marines else "loss",
        "marines_alive": marines, "enemies_alive": 0, "enemies_killed": 20, "fight_seconds": 10.0,
        "decisions": 10, "late_decisions": 0, "latencies_ms": [], "input_tokens": 0, "output_tokens": 0,
    }
    if hp is not None:
        r["hp_alive_sum"] = hp
    return r


def test_evaluate_survivor_hp():
    m = summarize([run("stutter_all", 4, 120.0), run("stutter_all", 2, 30.0), run("stutter_all", 0, 0.0),
                   run("random", 3)])
    assert m["stutter_all"]["mean_survivor_hp"] == 25.0  # 150 HP over 6 survivors
    assert m["random"]["mean_survivor_hp"] is None
    t = format_table(m)
    assert "Survivor HP" in t.splitlines()[0]
    assert list(summarize([run(p, 1) for p in reversed(POLICY_NAMES)])) == list(POLICY_NAMES)
