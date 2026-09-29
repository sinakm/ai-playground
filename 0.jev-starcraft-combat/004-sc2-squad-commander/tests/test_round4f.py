"""Round 4f: early pre_split and confidence gating (jev_commander only). No SC2, no network."""

import json
from types import SimpleNamespace

from arena import config
from arena.actions import execute_actions
from arena.bot import ArenaBot
from arena.jev import JevCommanderClient
from arena.policies import Decision, JevCommander
from arena.views import UnitView


def marine(tag, x, y, hp=45.0):
    return UnitView(tag, "marine", x, y, hp)


def bane(tag, x, y):
    return UnitView(tag, "baneling", x, y, 30.0)


def ling(tag, x, y):
    return UnitView(tag, "zergling", x, y, 35.0)


def test_config():
    assert config.SQUAD_PLANS["pre_split"] == (
        "banelings are within 12 cells of the squad and marines are clumped: spread out before they arrive"
    )
    assert (config.SOLDIER_MIN_CONFIDENCE, config.COMMANDER_MIN_CONFIDENCE) == (0.5, 0.5)


# ---------- soldier gating ----------

# marine 1: baneling 2.8 cells away (inside 3, outside the 2.5 reflex); marine 2 and 3: nothing close.
MS = [marine(1, 0.0, 0.0), marine(2, -10.0, 0.0), marine(3, -10.0, 8.0)]
ES = [bane(20, 2.8, 0.0), ling(30, -30.0, 0.0)]


def test_soldier_gating_defaults_kite_near_baneling_else_attack():
    stats = {}
    ex = execute_actions(
        {1: "cover_ally", 2: "bait", 3: "retreat"}, MS, ES,
        confidences={1: 0.4, 2: 0.3, 3: 0.9}, min_confidence=0.5, stats=stats,
    )
    assert ex == {1: "kite", 2: "attack", 3: "retreat"}
    assert stats["low_confidence_marines"] == 2


def test_soldier_gating_happens_before_roles():
    # bait_and_split makes marine 1 (closest) the bait even though its low-confidence choice became kite
    stats = {}
    ex = execute_actions({1: "attack", 2: "attack"}, MS, ES, squad_plan="bait_and_split",
                         confidences={1: 0.1, 2: 0.2}, min_confidence=0.5, stats=stats)
    assert ex[1] == "bait" and ex[2] == "attack" and stats["low_confidence_marines"] == 2


def test_no_gating_without_min_confidence():
    stats = {}
    ex = execute_actions({2: "bait"}, MS, ES, confidences={2: 0.1}, stats=stats)
    assert ex == {2: "bait"} and stats["low_confidence_marines"] == 0


# ---------- commander plan keep ----------


class FakePlan:
    """Call 1 answers squad_plan from a script of (plan, confidence); call 2 answers attack."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        if "squad_plan" in questions:
            plan, conf = self.script.pop(0)
            answers = {"squad_plan": SimpleNamespace(choice=plan, probabilities={plan: conf}, confidence=conf)}
        else:
            answers = {k: SimpleNamespace(choice="attack", probabilities={}, confidence=0.9) for k in questions}
        return SimpleNamespace(answers=answers, usage=SimpleNamespace(input_tokens=1, output_tokens=0), model="m")


STATE = {"marines": [{"id": 1}], "priority_candidates": []}


def test_commander_first_step_uses_low_confidence_answer():
    fake = FakePlan([("pre_split", 0.3)])
    d = JevCommander(JevCommanderClient(client=fake)).decide(STATE)
    assert d.squad_plan == "pre_split" and d.plan_kept_low_confidence is False


def test_commander_keeps_previous_plan_on_low_confidence():
    fake = FakePlan([("pre_split", 0.9), ("hold_and_shoot", 0.4), ("focus_banes", 0.6)])
    p = JevCommander(JevCommanderClient(client=fake))
    assert p.decide(STATE).squad_plan == "pre_split"
    d = p.decide(STATE)
    assert d.squad_plan == "pre_split" and d.plan_kept_low_confidence is True
    assert fake.calls[3][0]["squad_plan"] == "pre_split"  # soldiers see the kept plan
    d = p.decide(STATE)
    assert d.squad_plan == "focus_banes" and d.plan_kept_low_confidence is False


def test_warmup_does_not_set_previous_plan():
    fake = FakePlan([("pre_split", 0.9), ("hold_and_shoot", 0.2)])
    p = JevCommander(JevCommanderClient(client=fake))
    p.warmup({"note": "warmup"})
    d = p.decide(STATE)
    assert d.squad_plan == "hold_and_shoot" and d.plan_kept_low_confidence is False


# ---------- bot record ----------


class Scripted:
    name = "jev_commander"
    uses_blackboard = True

    def decide(self, state):
        return Decision(action="attack", marine_actions={1: "cover_ally", 2: "bait", 3: "retreat"},
                        marine_confidences={1: 0.4, 2: 0.3, 3: 0.9}, squad_plan="hold_and_shoot",
                        plan_kept_low_confidence=True, commander_latency_ms=1.0, soldier_latency_ms=1.0, model="m")


def test_bot_record_counters_and_raw_choices(tmp_path):
    bot = ArenaBot(Scripted(), tmp_path, seed=0, realtime=False)
    state, d, executed, _ = bot._plan_step(MS, ES, elapsed=12)
    bot._log_decision(d, executed, state, 12, 3, 2)
    bot.close()
    rec = json.loads((tmp_path / "decisions.jsonl").read_text().splitlines()[0])
    assert rec["marine_actions"] == {"1": "cover_ally", "2": "bait", "3": "retreat"}
    assert rec["executed_actions"] == {"1": "kite", "2": "attack", "3": "retreat"}
    assert rec["low_confidence_marines"] == 2
    assert rec["plan_kept_low_confidence"] is True


def test_no_gating_for_random(tmp_path):
    policy = SimpleNamespace(
        name="random",
        decide=lambda s: Decision(action="bait", marine_actions={2: "bait"}, marine_confidences={2: 0.1}),
    )
    bot = ArenaBot(policy, tmp_path, seed=0, realtime=False)
    state, d, executed, _ = bot._plan_step(MS, ES, elapsed=12)
    bot._log_decision(d, executed, state, 12, 3, 2)
    bot.close()
    rec = json.loads((tmp_path / "decisions.jsonl").read_text().splitlines()[0])
    assert executed == {2: "bait"} and rec["low_confidence_marines"] == 0
    assert rec["plan_kept_low_confidence"] is None
