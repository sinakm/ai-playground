"""Round 4b: role facts in the state, plan-driven roles in code, focus_bane cap. No SC2, no network."""

import json

from arena import config
from arena.actions import execute_actions, plan_marine_orders
from arena.bot import ArenaBot
from arena.overlay import PANEL_SIZE, cumulative, panel_frame, panel_actions
from arena.policies import Decision
from arena.state import Blackboard, build_commander_state, closest_to_banelings
from arena.views import UnitView


def marine(tag, x, y, hp=45.0):
    return UnitView(tag, "marine", x, y, hp)


def bane(tag, x, y):
    return UnitView(tag, "baneling", x, y, 30.0)


def ling(tag, x, y):
    return UnitView(tag, "zergling", x, y, 35.0)


# Marines on a line; banelings to the right.
MS = [marine(1, 0.0, 0.0), marine(2, 1.0, 0.0), marine(3, 2.0, 0.0), marine(4, 3.0, 0.0), marine(5, -8.0, 0.0)]
ES = [bane(20, 5.0, 0.0), bane(21, 5.5, 1.0), ling(30, 0.0, 10.0)]


# ---------- role facts ----------


def test_is_closest_to_banelings_and_banelings_within_3():
    s = build_commander_state(MS, ES, fight_loop=0, memory=Blackboard())
    by_id = {m["id"]: m for m in s["marines"]}
    assert [t for t, m in by_id.items() if m["is_closest_to_banelings"]] == [4]
    assert by_id[4]["banelings_within_3"] == 2  # bane 20 at 2.0, bane 21 at ~2.7
    assert by_id[3]["banelings_within_3"] == 1  # bane 20 at 3.0 (inclusive), bane 21 at ~3.64
    assert by_id[1]["banelings_within_3"] == 0
    assert closest_to_banelings(MS, ES) == 4


def test_role_facts_without_banelings():
    s = build_commander_state(MS, [ling(30, 0.0, 10.0)], fight_loop=0, memory=Blackboard())
    assert not any(m["is_closest_to_banelings"] for m in s["marines"])
    assert all(m["banelings_within_3"] == 0 for m in s["marines"])
    assert closest_to_banelings(MS, [ling(30, 0.0, 10.0)]) is None


# ---------- plan-driven roles ----------


def test_bait_and_split_assigns_bait_to_closest_and_converts_threatened_to_split():
    raw = {1: "attack", 2: "focus_bane", 3: "attack", 4: "attack", 5: "bait"}
    ex = execute_actions(raw, MS, ES, squad_plan="bait_and_split")
    assert ex[4] == "bait"  # closest to banelings, overrides its Jev choice
    assert ex[3] == "split"  # attack with a baneling within 3
    assert ex[1] == "attack" and ex[2] == "focus_bane"  # no baneling within 3: unchanged
    assert ex[5] == "kite"  # extra bait choices still become kite (one bait per step)


def test_roles_only_apply_under_bait_and_split():
    raw = {1: "attack", 3: "attack", 4: "attack"}
    assert execute_actions(raw, MS, ES, squad_plan="focus_banes") == raw
    assert execute_actions(raw, MS, ES) == raw


def test_bait_and_split_without_banelings_changes_nothing():
    raw = {1: "attack", 4: "attack"}
    assert execute_actions(raw, MS, [ling(30, 0.0, 10.0)], squad_plan="bait_and_split") == raw


def test_bait_override_only_for_marines_jev_answered():
    raw = {1: "attack"}  # marine 4 has no Jev answer: no order invented for it
    assert execute_actions(raw, MS, ES, squad_plan="bait_and_split") == {1: "attack"}


# ---------- focus_bane cap ----------


def test_focus_bane_cap_keeps_four_closest_to_priority_target():
    ms = [marine(i, float(i), 0.0) for i in range(1, 8)]
    es = [bane(20, 10.0, 0.0), bane(21, -10.0, 0.0)]
    raw = {i: "focus_bane" for i in range(1, 8)}
    ex = execute_actions(raw, ms, es, priority_target=20)
    assert config.FOCUS_BANE_CAP == 4
    assert sorted(t for t, a in ex.items() if a == "focus_bane") == [4, 5, 6, 7]
    assert sorted(t for t, a in ex.items() if a == "attack") == [1, 2, 3]
    ex = execute_actions(raw, ms, es, priority_target=21)
    assert sorted(t for t, a in ex.items() if a == "focus_bane") == [1, 2, 3, 4]


def test_focus_bane_cap_orders():
    ms = [marine(i, float(i), 0.0) for i in range(1, 8)]
    es = [bane(20, 8.0, 0.0)]  # marines 4..7 within FOCUS_RANGE of it
    ex = execute_actions({i: "focus_bane" for i in range(1, 8)}, ms, es, priority_target=20)
    kinds = [o.kind for o in plan_marine_orders(ex, ms, es, priority_target=20)]
    assert kinds.count("attack_unit") == 4 and kinds.count("attack") == 3


# ---------- instructions ----------


def test_soldier_instructions_mention_role_facts():
    assert (
        "Your squad_plan and role facts are in the state; follow the plan unless your own situation "
        "clearly needs another action." in config.COMMANDER_MARINE_INSTRUCTIONS_TEMPLATE.format(tag=1)
    )


# ---------- bot logging and blackboard ----------


class ScriptedCommander:
    name = "jev_commander"
    uses_blackboard = True

    def decide(self, state):
        raw = {1: "attack", 2: "attack", 3: "attack", 4: "attack", 5: "attack"}
        return Decision(action="attack", marine_actions=raw, squad_plan="bait_and_split", priority_target=20,
                        commander_latency_ms=1.0, soldier_latency_ms=1.0, latency_ms=2.0, model="m")


def test_bot_logs_executed_actions_and_blackboard_uses_them(tmp_path):
    # Distances kept above the 2.5-cell reflex: marine 4 is 2.6 cells from the Baneling
    # (closest -> bait), marine 3 is 2.8 cells away (within 3 -> split).
    ms = [marine(1, -10.0, 0.0), marine(3, 0.0, 2.8), marine(4, 2.6, 0.0)]
    es = [bane(20, 0.0, 0.0)]
    bot = ArenaBot(ScriptedCommander(), tmp_path, seed=0, realtime=False)
    state, d, executed, orders = bot._plan_step(ms, es, elapsed=12)
    assert executed[4] == "bait" and executed[3] == "split"
    bot._log_decision(d, executed, state, elapsed=12, marines_alive=len(ms), enemies_alive=len(es))
    bot.close()
    [rec] = [json.loads(line) for line in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert rec["marine_actions"]["4"] == "attack"
    assert rec["executed_actions"]["4"] == "bait" and rec["executed_actions"]["3"] == "split"
    assert bot.blackboard.last_actions[4] == "bait"
    nxt = build_commander_state(ms, es, fight_loop=24, memory=bot.blackboard)
    assert {m["id"]: m["last_action"] for m in nxt["marines"]}[4] == "bait"


# ---------- overlay ----------


def test_overlay_bars_use_executed_actions():
    rec = {
        "policy": "jev_commander", "action": "attack", "probabilities": None, "confidence": 0.7,
        "latency_ms": 300.0, "commander_latency_ms": 150.0, "soldier_latency_ms": 150.0,
        "input_tokens": 10, "output_tokens": 0, "marines_alive": 4, "enemies_alive": 10, "late": False,
        "marine_actions": {"1": "attack", "2": "attack", "3": "attack", "4": "attack"},
        "executed_actions": {"1": "attack", "2": "split", "3": "split", "4": "bait"},
        "squad_plan": "bait_and_split", "priority_target": 20,
    }
    shares, top = panel_actions(rec)
    assert shares["split"] == 0.5 and shares["bait"] == 0.25 and shares["attack"] == 0.25
    assert top == "split"
    assert panel_frame(rec, cumulative([rec])[0]).size == PANEL_SIZE
