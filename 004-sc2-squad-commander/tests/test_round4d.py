"""Round 4d: reflex kite, pre_split, nearest-target Zerg, commander stim, focus range guard, stim wording."""

import json
from types import SimpleNamespace

from arena import config
from arena.actions import Order, execute_actions, plan_marine_orders, squad_stim_orders
from arena.bot import ArenaBot, zerg_targets
from arena.jev import JevCommanderClient
from arena.overlay import PANEL_SIZE, commander_header, cumulative, panel_frame
from arena.policies import Decision, JevCommander, RandomMarine
from arena.views import UnitView


def marine(tag, x, y, hp=45.0, stimmed=False):
    return UnitView(tag, "marine", x, y, hp, stimmed)


def bane(tag, x, y):
    return UnitView(tag, "baneling", x, y, 30.0)


def ling(tag, x, y):
    return UnitView(tag, "zergling", x, y, 35.0)


# ---------- 1. reflex ----------

MS = [marine(1, 0.0, 0.0), marine(2, 3.0, 0.0), marine(3, -10.0, 0.0)]
ES = [bane(20, 5.0, 0.0), ling(30, 0.0, 10.0)]  # marine 2 is 2.0 from bane 20


def test_reflex_kites_marines_with_baneling_within_2_5():
    assert config.REFLEX_KITE_DISTANCE == 2.5
    stats = {}
    ex = execute_actions({1: "attack", 2: "focus_bane", 3: "attack"}, MS, ES, reflex=True, stats=stats)
    assert ex == {1: "attack", 2: "kite", 3: "attack"}
    assert stats["reflex_count"] == 1


def test_reflex_overrides_role_assignment_and_is_off_by_default():
    stats = {}
    # bait_and_split makes marine 2 (closest) the bait; the reflex then turns it into kite
    ex = execute_actions({1: "attack", 2: "attack"}, MS, ES, squad_plan="bait_and_split", reflex=True, stats=stats)
    assert ex[2] == "kite" and stats["reflex_count"] == 1
    assert execute_actions({2: "attack"}, MS, ES) == {2: "attack"}


def test_reflex_does_not_count_marines_already_kiting():
    stats = {}
    execute_actions({2: "kite"}, MS, ES, reflex=True, stats=stats)
    assert stats["reflex_count"] == 0


# ---------- 2. pre_split ----------


def test_pre_split_plan_text():
    assert config.SQUAD_PLANS["pre_split"] == (
        "banelings are within 12 cells of the squad and marines are clumped: spread out before they arrive"
    )


def test_pre_split_converts_attack_and_focus_bane_to_split():
    ms = [marine(1, 0.0, 0.0), marine(2, 1.0, 0.0), marine(3, 0.0, 1.0), marine(4, 1.0, 1.0)]
    es = [bane(20, 7.0, 0.0)]
    ex = execute_actions({1: "attack", 2: "focus_bane", 3: "kite", 4: "retreat"}, ms, es, squad_plan="pre_split")
    assert ex == {1: "split", 2: "split", 3: "kite", 4: "retreat"}


# ---------- 3. Zerg target the nearest Marine ----------


def test_zerg_targets_nearest_marine():
    ms = [marine(1, 0.0, 0.0), marine(2, 10.0, 0.0)]
    es = [bane(20, 2.0, 0.0), ling(30, 9.0, 1.0), bane(21, 5.0, 0.0)]
    assert zerg_targets(ms, es) == {20: 1, 30: 2, 21: 1}  # tie at 5.0 -> lowest tag
    assert zerg_targets([], es) == {}


# ---------- 4. commander stim ----------


class FakeStim:
    def __init__(self, stim_p):
        self.stim_p = stim_p
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        answers = {}
        if "squad_plan" in questions:
            answers["squad_plan"] = SimpleNamespace(choice="hold_and_shoot", probabilities={}, confidence=0.9)
            answers["stim_now"] = SimpleNamespace(noul=self.stim_p)
        else:
            for k in questions:
                answers[k] = SimpleNamespace(choice="attack", probabilities={}, confidence=0.8)
        return SimpleNamespace(answers=answers, usage=SimpleNamespace(input_tokens=1, output_tokens=0), model="m")


def test_commander_call_1_asks_stim_now_and_soldiers_cannot_stim():
    fake = FakeStim(0.8)
    a = JevCommanderClient(client=fake).ask({"priority_candidates": []}, [1, 2])
    q1 = fake.calls[0][1]
    assert type(q1["stim_now"]).__name__ == "Noul"
    assert q1["stim_now"].instructions == (
        "Should the whole squad stim now? Yes when enemies are within 8 cells, "
        "most marines are unstimmed and above 20 HP."
    )
    assert a.stim_now is True and a.stim_now_p == 0.8
    soldier_criteria = dict(fake.calls[1][1]["marine_1"].criteria)
    assert list(soldier_criteria) == ["kite", "split", "attack", "retreat", "focus_bane", "cover_ally", "bait"]
    assert JevCommanderClient(client=FakeStim(0.3)).ask({"priority_candidates": []}, [1]).stim_now is False


def test_stim_kept_for_random_and_jev_marine():
    assert "stim" in config.MARINE_ACTIONS and "stim" in config.JEV_MARINE_ACTIONS
    seen = set()
    p = RandomMarine(seed=2)
    for _ in range(40):
        seen |= set(p.decide({"marines": [{"id": i} for i in range(12)]}).marine_actions.values())
    assert "stim" in seen


def test_jev_commander_decision_carries_stim_now():
    d = JevCommander(JevCommanderClient(client=FakeStim(0.9))).decide(
        {"marines": [{"id": 1}], "priority_candidates": []}
    )
    assert d.stim_now is True


def test_squad_stim_orders_unstimmed_above_20_hp():
    ms = [marine(1, 0, 0), marine(2, 0, 0, hp=20.0), marine(3, 0, 0, stimmed=True), marine(4, 0, 0, hp=21.0)]
    assert squad_stim_orders(ms) == [Order(1, "stim"), Order(4, "stim")]


class ScriptedStim:
    name = "jev_commander"
    uses_blackboard = True

    def __init__(self, stim_now):
        self.stim_now = stim_now

    def decide(self, state):
        return Decision(action="attack", marine_actions={1: "attack", 2: "attack"}, squad_plan="hold_and_shoot",
                        stim_now=self.stim_now, commander_latency_ms=1.0, soldier_latency_ms=1.0, model="m")


def test_bot_stim_now_yes_adds_stim_orders_first_and_logs(tmp_path):
    ms = [marine(1, 0.0, 0.0), marine(2, 0.0, 3.0, hp=15.0)]
    es = [ling(30, 10.0, 0.0)]
    bot = ArenaBot(ScriptedStim(True), tmp_path, seed=0, realtime=False)
    state, d, executed, orders = bot._plan_step(ms, es, elapsed=12)
    assert orders[0] == Order(1, "stim")
    assert [o.kind for o in orders].count("stim") == 1  # marine 2 has 15 HP
    assert [o.kind for o in orders[1:]] == ["attack", "attack"]
    bot._log_decision(d, executed, state, 12, 2, 1)
    bot.close()
    rec = json.loads((tmp_path / "decisions.jsonl").read_text().splitlines()[0])
    assert rec["stim_now"] is True and rec["reflex_count"] == 0


def test_bot_stim_now_no_issues_no_stim(tmp_path):
    bot = ArenaBot(ScriptedStim(False), tmp_path, seed=0, realtime=False)
    _, _, _, orders = bot._plan_step([marine(1, 0.0, 0.0)], [ling(30, 10.0, 0.0)], elapsed=12)
    bot.close()
    assert [o.kind for o in orders] == ["attack"]


def test_bot_logs_reflex_count(tmp_path):
    bot = ArenaBot(ScriptedStim(False), tmp_path, seed=0, realtime=False)
    ms = [marine(1, 0.0, 0.0), marine(2, 3.0, 0.0)]
    state, d, executed, _ = bot._plan_step(ms, [bane(20, 5.0, 0.0)], elapsed=12)
    bot._log_decision(d, executed, state, 12, 2, 1)
    bot.close()
    rec = json.loads((tmp_path / "decisions.jsonl").read_text().splitlines()[0])
    assert rec["executed_actions"] == {"1": "attack", "2": "kite"} and rec["reflex_count"] == 1


def test_no_reflex_for_random(tmp_path):
    policy = SimpleNamespace(name="random", decide=lambda s: Decision(action="attack", marine_actions={2: "attack"}))
    bot = ArenaBot(policy, tmp_path, seed=0, realtime=False)
    _, _, executed, _ = bot._plan_step([marine(2, 3.0, 0.0)], [bane(20, 5.0, 0.0)], elapsed=12)
    bot.close()
    assert executed == {2: "attack"}


# ---------- 5. focus range guard ----------


def test_focus_bane_only_within_5_cells_of_target():
    assert config.FOCUS_RANGE == 5.0
    ms = [marine(1, 0.0, 0.0), marine(2, 6.0, 0.0)]
    es = [bane(20, 11.0, 0.0), ling(30, -1.0, 0.0)]
    orders = {o.unit_id: o for o in plan_marine_orders({1: "focus_bane", 2: "focus_bane"}, ms, es, priority_target=20)}
    assert orders[2] == Order(2, "attack_unit", 11.0, 0.0, target_id=20)  # exactly 5.0 cells
    assert orders[1] == Order(1, "attack", -1.0, 0.0)  # 11 cells away: shoot nearest enemy


# ---------- 6. stim wording ----------


def test_stim_wording():
    for text in (config.ACTIONS["stim"], config.MARINE_ACTIONS["stim"]):
        assert "nearly doubles damage" not in text
        assert "adds about 50% damage output" in text


# ---------- overlay ----------


def test_commander_header_appends_stim():
    assert commander_header({"squad_plan": "pre_split", "stim_now": True}) == "COMMANDER: PRE_SPLIT · STIM"
    assert commander_header({"squad_plan": "pre_split", "stim_now": False}) == "COMMANDER: PRE_SPLIT"
    rec = {
        "policy": "jev_commander", "action": "split", "probabilities": None, "confidence": 0.7,
        "latency_ms": 300.0, "commander_latency_ms": 150.0, "soldier_latency_ms": 150.0,
        "input_tokens": 10, "output_tokens": 0, "marines_alive": 2, "enemies_alive": 10, "late": False,
        "marine_actions": {"1": "attack", "2": "attack"}, "executed_actions": {"1": "split", "2": "kite"},
        "squad_plan": "pre_split", "priority_target": None, "stim_now": True, "reflex_count": 1,
    }
    assert panel_frame(rec, cumulative([rec])[0]).size == PANEL_SIZE
