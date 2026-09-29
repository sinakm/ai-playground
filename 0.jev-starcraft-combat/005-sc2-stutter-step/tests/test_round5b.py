"""Round 5b: background decisions in realtime. No SC2, no network."""

import asyncio
import json
import threading

import pytest
from typesafe_sdk import TypeSafeInternalServerError

from arena import config
from arena.bot import ArenaBot, decide_or_error, step_reflex
from arena.policies import Decision
from arena.scheduler import DecisionScheduler
from arena.views import UnitView


def marine(tag, x, y, hp=45.0):
    return UnitView(tag, "marine", x, y, hp)


def bane(tag, x, y):
    return UnitView(tag, "baneling", x, y, 30.0)


def ling(tag, x, y):
    return UnitView(tag, "zergling", x, y, 35.0)


def wait_done(sched, timeout=2.0):
    sched._future.result(timeout=timeout)


# ---------- DecisionScheduler ----------


def test_one_in_flight_and_skip_counting():
    gate = threading.Event()
    calls = []

    def fn(snap):
        calls.append(snap)
        gate.wait(2.0)
        return snap * 10

    s = DecisionScheduler(fn)
    assert s.poll() is None and not s.pending
    assert s.maybe_start(1) is True
    assert s.maybe_start(2) is False and s.maybe_start(3) is False
    assert s.skipped == 2 and s.started == 1 and s.poll() is None
    gate.set()
    wait_done(s)
    done = s.poll()
    assert (done.snapshot, done.result) == (1, 10)
    assert s.poll() is None and not s.pending  # delivered once
    assert s.maybe_start(4) is True
    wait_done(s)
    assert s.poll().result == 40 and calls == [1, 4]
    s.close()


def test_exception_reraised_on_poll():
    s = DecisionScheduler(lambda snap: 1 / 0)
    s.maybe_start("x")
    s._future.exception(timeout=2.0)
    with pytest.raises(ZeroDivisionError):
        s.poll()
    assert not s.pending
    s.close()


def test_close_with_call_in_flight_does_not_hang():
    gate = threading.Event()
    s = DecisionScheduler(lambda snap: gate.wait(2.0))
    s.maybe_start(1)
    s.close()
    s.close()
    assert s.poll() is None and s.maybe_start(2) is False
    gate.set()


def test_close_unused_scheduler():
    DecisionScheduler(lambda s: s).close()


# ---------- bot: apply against current units ----------


class Cmd:
    name = "jev_commander_stutter"
    uses_blackboard = True
    stutter_default = True

    def __init__(self, actions):
        self.actions = actions

    def decide(self, state):
        return Decision(action="stutter", marine_actions=dict(self.actions), squad_plan="hold_and_shoot",
                        model="m", latency_ms=350.0)


class Raising:
    name = "jev_commander_stutter"
    uses_blackboard = True

    def decide(self, state):
        raise TypeSafeInternalServerError(520, "", {}, message="520")


def records(tmp_path):
    return [json.loads(line) for line in (tmp_path / "decisions.jsonl").read_text().splitlines()]


def test_apply_on_completion_uses_current_units(tmp_path):
    bot = ArenaBot(Cmd({1: "attack", 2: "stutter", 3: "attack"}), tmp_path, seed=0, realtime=True)
    snap_ms = [marine(1, 0.0, 0.0), marine(2, 0.0, 10.0), marine(3, 0.0, 20.0)]
    es_then = [ling(30, 30.0, 0.0)]
    assert bot._start_background(snap_ms, es_then, elapsed=12) is True
    assert bot._start_background(snap_ms, es_then, elapsed=24) is False  # still in flight
    wait_done(bot.scheduler)
    done = bot.scheduler.poll()
    # Marine 3 died meanwhile, Marine 1 moved, a Baneling is now 2 cells from Marine 1.
    ms_now = [marine(1, 5.0, 0.0), marine(2, 0.0, 10.0)]
    es_now = [bane(20, 7.0, 0.0), ling(30, 30.0, 0.0)]
    d, orders = bot._apply_result(done.snapshot, done.result, ms_now, es_now, elapsed=31)
    assert d is not None
    assert {o.unit_id for o in orders} == {1}  # 2 stutters (per-step), 3 is dead
    assert orders[0].kind == "kite"  # reflex against the CURRENT Baneling position
    assert bot.stuttering == {2}
    assert bot.blackboard.last_hp == {1: 45.0, 2: 45.0}
    assert bot.apply_delays == [19]
    asyncio.run(bot._finish(2, 1, 40, leave=False))
    bot.close()
    [rec] = records(tmp_path)
    assert (rec["requested_loop"], rec["applied_loop"], rec["apply_delay_loops"]) == (12, 31, 19)
    assert rec["fight_loop"] == 31 and rec["late"] is True and rec["marines_alive"] == 2
    assert len(rec["state"]["marines"]) == 3  # the snapshot the policy saw
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["skipped_decisions"] == 1 and s["mean_apply_delay_loops"] == 19.0


def test_apply_on_time_is_not_late(tmp_path):
    bot = ArenaBot(Cmd({1: "attack"}), tmp_path, seed=0, realtime=True)
    bot._start_background([marine(1, 0.0, 0.0)], [ling(30, 30.0, 0.0)], elapsed=12)
    wait_done(bot.scheduler)
    done = bot.scheduler.poll()
    bot._apply_result(done.snapshot, done.result, [marine(1, 0.0, 0.0)], [ling(30, 30.0, 0.0)],
                      elapsed=12 + config.DECISION_INTERVAL_LOOPS)
    bot.close()
    [rec] = records(tmp_path)
    assert rec["late"] is False and rec["apply_delay_loops"] == config.DECISION_INTERVAL_LOOPS


def test_background_api_error_keeps_previous_orders(tmp_path):
    bot = ArenaBot(Raising(), tmp_path, seed=0, realtime=True)
    bot.stuttering = {1}
    bot._base_actions = {1: "stutter"}
    bot._start_background([marine(1, 0.0, 0.0)], [ling(30, 30.0, 0.0)], elapsed=12)
    wait_done(bot.scheduler)
    done = bot.scheduler.poll()
    d, orders = bot._apply_result(done.snapshot, done.result, [marine(1, 0.0, 0.0)], [ling(30, 30.0, 0.0)], 20)
    bot.close()
    assert d is None and orders == [] and bot.stuttering == {1} and bot._base_actions == {1: "stutter"}
    assert bot.api_errors == 1 and bot.decisions == 0 and bot.apply_delays == []
    [rec] = records(tmp_path)
    assert rec["action"] == "api_error" and rec["error"] == "TypeSafeInternalServerError"
    assert (rec["requested_loop"], rec["applied_loop"], rec["apply_delay_loops"]) == (12, 20, 8)


def test_paused_path_has_null_delay_fields(tmp_path):
    bot = ArenaBot(Cmd({1: "attack"}), tmp_path, seed=0, realtime=False)
    assert bot.scheduler is None
    ms, es = [marine(1, 0.0, 0.0)], [ling(30, 30.0, 0.0)]
    state, d, ex, _ = bot._plan_step(ms, es, elapsed=12)
    bot._log_decision(d, ex, state, 12, 1, 1)
    asyncio.run(bot._finish(1, 1, 40, leave=False))
    bot.close()
    [rec] = records(tmp_path)
    assert rec["late"] is False and rec["apply_delay_loops"] is None and rec["requested_loop"] is None
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["skipped_decisions"] == 0 and s["mean_apply_delay_loops"] is None


def test_realtime_close_with_in_flight_decision(tmp_path):
    gate = threading.Event()

    class Slow(Cmd):
        def decide(self, state):
            gate.wait(2.0)
            return super().decide(state)

    bot = ArenaBot(Slow({1: "attack"}), tmp_path, seed=0, realtime=True)
    bot._start_background([marine(1, 0.0, 0.0)], [ling(30, 30.0, 0.0)], elapsed=12)
    asyncio.run(bot._finish(1, 1, 20, leave=False))
    bot.close()
    gate.set()
    assert not bot.scheduler.pending and bot.decisions == 0
    log = tmp_path / "decisions.jsonl"
    assert not log.exists() or records(tmp_path) == []


# ---------- per-step reflex pass ----------


def test_step_reflex_overrides_and_reverts():
    base = {1: "stutter", 2: "attack"}
    ms = [marine(1, 0.0, 0.0), marine(2, 20.0, 0.0)]
    desired, changed = step_reflex(base, dict(base), ms, [bane(20, 2.0, 0.0), ling(30, 30.0, 0.0)])
    assert changed == {1: "kite"} and desired == {1: "kite", 2: "attack"}
    # Baneling gone: Marine 1 goes back to its base action.
    desired, changed = step_reflex(base, desired, ms, [ling(30, 30.0, 0.0)])
    assert changed == {1: "stutter"} and desired == base
    # Nothing to do without a base or enemies.
    assert step_reflex({}, {}, ms, [ling(30, 30.0, 0.0)]) == ({}, {})
    assert step_reflex(base, base, ms, []) == (base, {})


def test_bot_reflex_orders_update_stutter_set(tmp_path):
    bot = ArenaBot(Cmd({}), tmp_path, seed=0, realtime=True)
    bot._base_actions = {1: "stutter", 2: "stutter"}
    bot._current_actions = dict(bot._base_actions)
    bot.stuttering = {1, 2}
    ms = [marine(1, 0.0, 0.0), marine(2, 20.0, 0.0)]
    orders = bot._reflex_orders(ms, [bane(20, 2.0, 0.0), ling(30, 24.0, 0.0)])
    assert [(o.unit_id, o.kind) for o in orders] == [(1, "kite")]
    assert bot.stuttering == {2} and bot.step_reflex_changes == 1
    assert bot._reflex_orders(ms, [bane(20, 2.0, 0.0), ling(30, 24.0, 0.0)]) == []  # no re-issue
    assert bot._reflex_orders(ms, [ling(30, 24.0, 0.0)]) == []  # back to stutter: per-step orders
    assert bot.stuttering == {1, 2} and bot.step_reflex_changes == 2
    bot.close()


def test_scheduler_callable_uses_decide_or_error(tmp_path):
    bot = ArenaBot(Raising(), tmp_path, seed=0, realtime=True)
    bot._start_background([marine(1, 0.0, 0.0)], [ling(30, 30.0, 0.0)], elapsed=12)
    wait_done(bot.scheduler)
    d, err = bot.scheduler.poll().result
    bot.close()
    assert d is None and err == decide_or_error(Raising(), {})[1]
