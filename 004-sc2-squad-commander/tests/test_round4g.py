"""Round 4g: pre-contact split, crowding check, group escape reflex. No SC2, no network."""

import json
from types import SimpleNamespace

from arena import config
from arena.actions import execute_actions, plan_marine_orders
from arena.bot import ArenaBot
from arena.jev import JevCommanderClient
from arena.policies import Decision, JevCommander
from arena.state import in_contact
from arena.views import UnitView


def marine(tag, x, y, hp=45.0):
    return UnitView(tag, "marine", x, y, hp)


def bane(tag, x, y):
    return UnitView(tag, "baneling", x, y, 30.0)


def ling(tag, x, y):
    return UnitView(tag, "zergling", x, y, 35.0)


def test_config():
    assert config.PRE_CONTACT_DISTANCE == 4.0
    assert config.CROWDED_DISTANCE == 1.5
    assert (config.GROUP_ESCAPE_NEIGHBORS, config.GROUP_ESCAPE_RADIUS) == (2, 2.0)


# Clumped pair (1, 2) and a lone marine 3.
MS = [marine(1, 0.0, 0.0), marine(2, 1.0, 0.0), marine(3, 0.0, 6.0)]


def test_in_contact():
    assert in_contact(MS, [bane(20, 5.0, 0.0)]) is True  # 4.0 from marine 2 (inclusive)
    assert in_contact(MS, [bane(20, 5.1, 0.0), ling(30, 1.0, 1.0)]) is False  # zerglings do not count
    assert in_contact(MS, []) is False


# ---------- 1. pre_split only before contact ----------


def test_pre_split_converts_before_contact_only_crowded():
    stats = {}
    raw = {1: "attack", 2: "focus_bane", 3: "attack"}
    ex = execute_actions(raw, MS, [bane(20, 10.0, 0.0)], squad_plan="pre_split", stats=stats)
    assert ex == {1: "split", 2: "split", 3: "attack"}  # marine 3 is not crowded
    assert stats["contact"] is False and stats["pre_split_active"] is True


def test_pre_split_has_no_effect_after_contact():
    stats = {}
    raw = {1: "attack", 2: "focus_bane", 3: "attack"}
    ex = execute_actions(raw, MS, [bane(20, 4.5, 0.0)], squad_plan="pre_split", stats=stats)
    assert ex == raw
    assert stats["contact"] is True and stats["pre_split_active"] is False


def test_stats_without_pre_split():
    stats = {}
    execute_actions({1: "attack"}, MS, [bane(20, 10.0, 0.0)], squad_plan="hold_and_shoot", stats=stats)
    assert stats["contact"] is False and stats["pre_split_active"] is False


# ---------- 2. bait_and_split split conversion needs crowding ----------


def test_bait_and_split_split_only_when_crowded():
    ms = [marine(1, 0.0, 0.0), marine(2, 1.0, 0.0), marine(3, 0.0, 4.0), marine(4, 3.0, 3.0)]
    es = [bane(20, 2.7, 0.0), bane(21, 1.0, 6.5)]
    # marine 2 is 1.7 from bane 20 (closest -> bait; reflex off here); marine 1 is 2.7 from it;
    # marine 3 is 2.69 from bane 21; marine 4 is 3.01 from bane 20.
    ex = execute_actions({1: "attack", 2: "attack", 3: "attack", 4: "attack"}, ms, es, squad_plan="bait_and_split")
    assert ex[2] == "bait"
    assert ex[1] == "split"  # bane 20 within 3, marine 2 within 1.5
    assert ex[3] == "attack"  # bane 21 within 3 (2.69) but no marine within 1.5


# ---------- 3. group escape ----------


def test_group_escape_retreats_instead_of_kite():
    ms = [marine(1, 0.0, 0.0), marine(2, 1.0, 0.0), marine(3, 0.0, 1.5), marine(4, 20.0, 0.0)]
    es = [bane(20, -2.0, 0.0), bane(21, 22.0, 0.0)]
    stats = {}
    ex = execute_actions({1: "attack", 2: "attack", 4: "attack"}, ms, es, reflex=True, stats=stats)
    assert ex[1] == "retreat"  # bane at 2.0; marines 2 and 3 within 2.0
    assert ex[4] == "kite"  # bane at 2.0, alone
    assert ex[2] == "attack"  # bane 3.0 away: no reflex
    assert stats["reflex_count"] == 2
    orders = {o.unit_id: o for o in plan_marine_orders(ex, ms, es)}
    assert orders[1].kind == "move"  # existing per-Marine retreat


def test_group_escape_not_counted_when_already_retreating():
    ms = [marine(1, 0.0, 0.0), marine(2, 1.0, 0.0), marine(3, 0.0, 1.5)]
    stats = {}
    execute_actions({1: "retreat"}, ms, [bane(20, -2.0, 0.0)], reflex=True, stats=stats)
    assert stats["reflex_count"] == 0


# ---------- kept plan after contact ----------


class FakePlan:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        if "squad_plan" in questions:
            plan, conf = self.script.pop(0)
            answers = {"squad_plan": SimpleNamespace(choice=plan, probabilities={}, confidence=conf)}
        else:
            answers = {k: SimpleNamespace(choice="attack", probabilities={}, confidence=0.9) for k in questions}
        return SimpleNamespace(answers=answers, usage=SimpleNamespace(input_tokens=1, output_tokens=0), model="m")


def state(nearest_bane):
    return {"marines": [{"id": 1}], "priority_candidates": [], "summary": {"nearest_baneling_distance": nearest_bane}}


def test_kept_pre_split_overridden_after_contact():
    fake = FakePlan([("pre_split", 0.9), ("hold_and_shoot", 0.3), ("focus_banes", 0.3), ("hold_and_shoot", 0.3)])
    p = JevCommander(JevCommanderClient(client=fake))
    assert p.decide(state(10.0)).squad_plan == "pre_split"
    d = p.decide(state(9.0))  # no contact yet: keep pre_split
    assert d.squad_plan == "pre_split" and d.plan_kept_low_confidence is True
    d = p.decide(state(3.0))  # contact: a kept pre_split falls back to Jev's raw plan
    assert d.squad_plan == "focus_banes" and d.plan_kept_low_confidence is False
    assert fake.calls[5][0]["squad_plan"] == "focus_banes"
    d = p.decide(state(6.0))  # contact already happened: sticky, kept plan is focus_banes (not pre_split)
    assert d.squad_plan == "focus_banes" and d.plan_kept_low_confidence is True


def test_kept_pre_split_overridden_when_contact_happened_earlier():
    fake = FakePlan([("focus_banes", 0.9), ("pre_split", 0.9), ("hold_and_shoot", 0.2)])
    p = JevCommander(JevCommanderClient(client=fake))
    p.decide(state(3.0))  # contact
    p.decide(state(7.0))  # confident pre_split after contact is Jev's call; it has no code effect
    d = p.decide(state(7.0))
    assert d.squad_plan == "hold_and_shoot" and d.plan_kept_low_confidence is False


# ---------- 4. record ----------


class Scripted:
    name = "jev_commander"
    uses_blackboard = True

    def decide(self, state):
        return Decision(action="attack", marine_actions={1: "attack", 2: "attack"}, squad_plan="pre_split",
                        commander_latency_ms=1.0, soldier_latency_ms=1.0, model="m")


def test_record_logs_contact_and_pre_split_active(tmp_path):
    bot = ArenaBot(Scripted(), tmp_path, seed=0, realtime=False)
    for es in ([bane(20, 10.0, 0.0)], [bane(20, 4.0, 0.0)]):
        state_, d, executed, _ = bot._plan_step(MS[:2], es, elapsed=12)
        bot._log_decision(d, executed, state_, 12, 2, 1)
    bot.close()
    recs = [json.loads(line) for line in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert (recs[0]["contact"], recs[0]["pre_split_active"]) == (False, True)
    assert recs[0]["executed_actions"] == {"1": "split", "2": "split"}
    assert (recs[1]["contact"], recs[1]["pre_split_active"]) == (True, False)
