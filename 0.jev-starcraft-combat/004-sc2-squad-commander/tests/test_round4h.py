"""Round 4h: low-HP retreat default, healthy-only commander stim, low-HP reflex. No SC2, no network."""

import inspect
import json

from arena import bot as bot_module
from arena import config
from arena.actions import Order, execute_actions, squad_stim_orders
from arena.bot import ArenaBot
from arena.policies import Decision
from arena.views import UnitView


def marine(tag, x, y, hp=45.0, stimmed=False):
    return UnitView(tag, "marine", x, y, hp, stimmed)


def bane(tag, x, y):
    return UnitView(tag, "baneling", x, y, 30.0)


def ling(tag, x, y):
    return UnitView(tag, "zergling", x, y, 35.0)


# ---------- 1. low-HP default in confidence gating ----------


def test_low_confidence_low_hp_defaults_to_retreat_bane_kite_first():
    ms = [marine(1, 0.0, 0.0, hp=15.0), marine(2, 0.0, 10.0, hp=12.0), marine(3, 0.0, 20.0, hp=40.0)]
    es = [bane(20, 2.8, 10.0), ling(30, 30.0, 30.0)]  # bane 2.8 from marine 2 (no reflex: > 2.5)
    stats = {}
    ex = execute_actions({1: "bait", 2: "attack", 3: "bait"}, ms, es,
                         confidences={1: 0.2, 2: 0.2, 3: 0.2}, min_confidence=0.5, stats=stats)
    assert ex == {1: "retreat", 2: "kite", 3: "attack"}
    assert stats["low_confidence_marines"] == 3


# ---------- 2. healthy-only stim ----------


def test_commander_stim_only_above_30_hp():
    assert config.COMMANDER_STIM_MIN_HP == 30
    ms = [marine(1, 0, 0, hp=31.0), marine(2, 0, 0, hp=30.0), marine(3, 0, 0, hp=45.0, stimmed=True),
          marine(4, 0, 0, hp=25.0)]
    assert squad_stim_orders(ms) == [Order(1, "stim")]


def test_stim_now_instructions_mention_healthy_marines():
    assert "most healthy marines (over 30 HP) are unstimmed" in config.STIM_NOW_INSTRUCTIONS


def test_stimmed_view_reflects_buff():
    assert "has_buff(BuffId.STIMPACK)" in inspect.getsource(bot_module.to_view)


class Scripted:
    name = "jev_commander"
    uses_blackboard = True

    def __init__(self, stim_now=True):
        self.stim_now = stim_now

    def decide(self, state):
        return Decision(action="attack", marine_actions={m["id"]: "attack" for m in state["marines"]},
                        squad_plan="hold_and_shoot", stim_now=self.stim_now,
                        commander_latency_ms=1.0, soldier_latency_ms=1.0, model="m")


def test_stim_now_can_fire_again_after_buff_wears_off(tmp_path):
    bot = ArenaBot(Scripted(), tmp_path, seed=0, realtime=False)
    es = [ling(30, 20.0, 0.0)]
    _, _, _, o1 = bot._plan_step([marine(1, 0.0, 0.0)], es, elapsed=12)
    _, _, _, o2 = bot._plan_step([marine(1, 0.0, 0.0, stimmed=True)], es, elapsed=24)
    _, _, _, o3 = bot._plan_step([marine(1, 0.0, 0.0, hp=35.0)], es, elapsed=240)  # buff gone
    bot.close()
    count = [sum(o.kind == "stim" for o in os_) for os_ in (o1, o2, o3)]
    assert count == [1, 0, 1]


# ---------- 3. low-HP reflex ----------


def test_low_hp_reflex_with_zergling_close():
    ms = [marine(1, 0.0, 0.0, hp=15.0), marine(2, 10.0, 0.0, hp=15.0), marine(3, 20.0, 0.0, hp=40.0),
          marine(4, 30.0, 0.0, hp=10.0)]
    es = [ling(30, 1.5, 0.0), ling(31, 12.5, 0.0), ling(32, 21.0, 0.0), bane(20, 32.0, 0.0), ling(33, 31.0, 0.0)]
    stats = {}
    ex = execute_actions({1: "attack", 2: "attack", 3: "attack", 4: "attack"}, ms, es, reflex=True, stats=stats)
    assert ex[1] == "retreat_to_squad"  # 15 HP, zergling at 1.5
    assert ex[2] == "attack"  # zergling at 2.5: too far
    assert ex[3] == "attack"  # healthy
    assert ex[4] == "kite"  # baneling within 2.5 wins
    assert stats["reflex_count"] == 2


def test_low_hp_reflex_off_without_reflex_flag():
    ms = [marine(1, 0.0, 0.0, hp=15.0)]
    assert execute_actions({1: "attack"}, ms, [ling(30, 1.5, 0.0)]) == {1: "attack"}


# ---------- 4. record ----------


def test_record_logs_low_hp_and_stimmed_this_step(tmp_path):
    bot = ArenaBot(Scripted(), tmp_path, seed=0, realtime=False)
    ms = [marine(1, 0.0, 0.0, hp=45.0), marine(2, 0.0, 5.0, hp=45.0), marine(3, 0.0, 10.0, hp=14.0),
          marine(4, 0.0, 15.0, hp=45.0, stimmed=True)]
    state, d, executed, _ = bot._plan_step(ms, [ling(30, 30.0, 0.0)], elapsed=12)
    bot._log_decision(d, executed, state, 12, 4, 1)
    bot.close()
    rec = json.loads((tmp_path / "decisions.jsonl").read_text().splitlines()[0])
    assert rec["low_hp_marines"] == 1
    assert rec["stimmed_this_step"] == 2


def test_record_stimmed_this_step_zero_without_stim_now(tmp_path):
    bot = ArenaBot(Scripted(stim_now=False), tmp_path, seed=0, realtime=False)
    state, d, executed, _ = bot._plan_step([marine(1, 0.0, 0.0)], [ling(30, 30.0, 0.0)], elapsed=12)
    bot._log_decision(d, executed, state, 12, 1, 1)
    bot.close()
    rec = json.loads((tmp_path / "decisions.jsonl").read_text().splitlines()[0])
    assert rec["stimmed_this_step"] == 0 and rec["low_hp_marines"] == 0
