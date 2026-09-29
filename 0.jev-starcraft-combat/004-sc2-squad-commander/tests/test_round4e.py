"""Round 4e: zergling swarm reflex (retreat_to_squad) and the cover_ally guard. No SC2, no network."""

import json
import math

from arena import config
from arena.actions import Order, execute_actions, plan_marine_orders
from arena.bot import ArenaBot
from arena.overlay import PANEL_SIZE, cumulative, panel_actions, panel_frame
from arena.policies import Decision
from arena.state import Blackboard, build_commander_state
from arena.views import UnitView


def marine(tag, x, y, hp=45.0):
    return UnitView(tag, "marine", x, y, hp)


def bane(tag, x, y):
    return UnitView(tag, "baneling", x, y, 30.0)


def ling(tag, x, y):
    return UnitView(tag, "zergling", x, y, 35.0)


def test_config():
    assert (config.ZERGLING_SWARM_COUNT, config.ZERGLING_SWARM_RADIUS, config.REGROUP_STEP) == (3, 2.0, 2.0)
    assert config.MARINE_ACTIONS["cover_ally"].endswith(
        "shoot the enemy closest to that teammate and no baneling is within 3 cells of you"
    )
    assert config.COMMANDER_MARINE_ACTIONS["cover_ally"] == config.MARINE_ACTIONS["cover_ally"]


# ---------- 1. zerglings_within_2 ----------

# Marine 1 is swarmed by 3 zerglings; marine 2 has 1 zergling close; marine 3 is far behind.
MS = [marine(1, 10.0, 0.0), marine(2, 0.0, 0.0), marine(3, -4.0, 0.0)]
LINGS = [ling(30, 11.5, 0.0), ling(31, 10.0, 1.5), ling(32, 10.0, -2.0), ling(33, 1.0, 1.0)]


def test_zerglings_within_2_own_and_teammates():
    s = build_commander_state(MS, LINGS, fight_loop=0, memory=Blackboard())
    by_id = {m["id"]: m for m in s["marines"]}
    assert by_id[1]["zerglings_within_2"] == 3  # 1.5, 1.5, 2.0 (inclusive)
    assert by_id[2]["zerglings_within_2"] == 1
    assert by_id[3]["zerglings_within_2"] == 0
    mate1 = next(t for t in by_id[2]["teammates"] if t["id"] == 1)
    assert mate1["zerglings_within_2"] == 3


# ---------- 2. swarm reflex ----------


def test_swarm_reflex_retreats_to_squad_and_counts():
    stats = {}
    ex = execute_actions({1: "attack", 2: "attack", 3: "attack"}, MS, LINGS, reflex=True, stats=stats)
    assert ex == {1: "retreat_to_squad", 2: "attack", 3: "attack"}
    assert stats["reflex_count"] == 1


def test_swarm_reflex_off_without_reflex_flag():
    assert execute_actions({1: "attack"}, MS, LINGS) == {1: "attack"}


def test_baneling_reflex_wins_over_swarm_reflex():
    es = LINGS + [bane(20, 8.0, 0.0)]  # 2.0 cells from marine 1
    stats = {}
    ex = execute_actions({1: "attack"}, MS, es, reflex=True, stats=stats)
    assert ex[1] == "kite" and stats["reflex_count"] == 1


def test_both_reflexes_count():
    ms = MS + [marine(4, 30.0, 0.0)]
    es = LINGS + [bane(20, 31.0, 0.0)]
    stats = {}
    ex = execute_actions({1: "attack", 4: "attack"}, ms, es, reflex=True, stats=stats)
    assert ex == {1: "retreat_to_squad", 4: "kite"} and stats["reflex_count"] == 2


def test_retreat_to_squad_order_moves_2_cells_toward_others_then_attacks():
    [o] = plan_marine_orders({1: "retreat_to_squad"}, MS, LINGS)
    # others' centroid (-2, 0): move from (10, 0) to (8, 0), then attack-move to the enemy centroid
    ex, ey = (sum(e.x for e in LINGS) / 4, sum(e.y for e in LINGS) / 4)
    assert o.kind == "kite" and o.unit_id == 1
    assert math.isclose(o.x, 10.0 - config.REGROUP_STEP) and math.isclose(o.y, 0.0, abs_tol=1e-9)
    assert math.isclose(o.tx, ex) and math.isclose(o.ty, ey)


def test_retreat_to_squad_does_not_overshoot_and_alone_attacks():
    ms = [marine(1, 0.0, 0.0), marine(2, 1.0, 0.0)]
    [o] = plan_marine_orders({1: "retreat_to_squad"}, ms, LINGS)
    assert math.isclose(o.x, 1.0) and math.isclose(o.y, 0.0, abs_tol=1e-9)
    [o] = plan_marine_orders({1: "retreat_to_squad"}, [marine(1, 10.0, 0.0)], LINGS)
    assert o == Order(1, "attack", 11.5, 0.0)


# ---------- 3. cover_ally guard ----------


def test_cover_ally_guard_converts_to_kite_for_any_policy():
    ms = [marine(1, 0.0, 0.0), marine(2, 1.0, 0.0, hp=10.0), marine(3, -20.0, 0.0), marine(4, -21.0, 0.0, hp=5.0)]
    es = [bane(20, 3.0, 0.0), ling(30, -22.0, 0.0)]
    ex = execute_actions({1: "cover_ally", 3: "cover_ally"}, ms, es)  # no reflex flag: random, too
    assert ex == {1: "kite", 3: "cover_ally"}  # bane at 3.0 (inclusive) blocks marine 1


# ---------- bot + overlay ----------


class Scripted:
    name = "jev_commander"
    uses_blackboard = True

    def decide(self, state):
        return Decision(action="attack", marine_actions={1: "attack", 2: "attack", 3: "attack"},
                        squad_plan="hold_and_shoot", commander_latency_ms=1.0, soldier_latency_ms=1.0, model="m")


def test_bot_logs_retreat_to_squad(tmp_path):
    bot = ArenaBot(Scripted(), tmp_path, seed=0, realtime=False)
    state, d, executed, orders = bot._plan_step(MS, LINGS, elapsed=12)
    bot._log_decision(d, executed, state, 12, 3, 4)
    bot.close()
    rec = json.loads((tmp_path / "decisions.jsonl").read_text().splitlines()[0])
    assert rec["executed_actions"]["1"] == "retreat_to_squad" and rec["reflex_count"] == 1
    assert bot.blackboard.last_actions[1] == "retreat_to_squad"
    shares, _ = panel_actions(rec)
    assert shares["retreat_to_squad"] == 1 / 3
    assert panel_frame(rec, cumulative([rec])[0]).size == PANEL_SIZE
