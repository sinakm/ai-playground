"""python-sc2 glue: spawn the fight, ask the policy, issue orders, log everything."""

from __future__ import annotations

import json
import time
from pathlib import Path

from sc2.bot_ai import BotAI
from sc2.data import Result
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from arena import config
from arena.actions import (
    Order,
    apply_reflex,
    execute_actions,
    plan_marine_orders,
    plan_orders,
    squad_stim_orders,
    stutter_step_orders,
)
from arena.jev import TRANSIENT_ERRORS
from arena.log import DecisionLog
from arena.scheduler import DecisionScheduler
from arena.state import Blackboard, build_commander_state, build_state, in_contact
from arena.views import UnitView

KINDS = {UnitTypeId.MARINE: "marine", UnitTypeId.BANELING: "baneling", UnitTypeId.ZERGLING: "zergling"}
ENEMY_TYPES = {UnitTypeId.BANELING, UnitTypeId.ZERGLING}
TOTAL_ENEMIES = config.BANELING_COUNT + config.ZERGLING_COUNT
# Fallback for the case handled in on_end: the SC2 engine can declare a Result
# for our player (e.g. all Marines dead) in the same observation it stops
# calling on_step for, so on_step's own "not marines" / "not enemies" branch
# never runs. See ArenaBot.on_end.
RESULT_MAP = {Result.Victory: "win", Result.Defeat: "loss", Result.Tie: "timeout", Result.Undecided: "timeout"}


def decide_or_error(policy, state: dict):
    """(decision, None), or (None, error class name) when the Jev call failed transiently
    even after the client's retry. Any other exception propagates."""
    try:
        return policy.decide(state), None
    except TRANSIENT_ERRORS as e:
        return None, type(e).__name__


def stutter_tags(executed: dict[int, str] | None) -> set[int]:
    """Marines that stutter until the next decision: those whose executed action is `stutter`.
    A squad-level decision (no per-Marine actions) clears the set."""
    return {tag for tag, a in (executed or {}).items() if a == "stutter"}


def step_reflex(
    base: dict[int, str], current: dict[int, str], marines: list[UnitView], enemies: list[UnitView]
) -> tuple[dict[int, str], dict[int, str]]:
    """Per-step reflex pass (realtime only): the reflexes applied to the actions of the last
    applied decision (`base`), against the units seen now. Returns the new executed-action map
    and the living Marines whose action differs from what they execute now (`current`).
    A Marine whose reflex no longer fires goes back to its base action."""
    if not base or not marines or not enemies:
        return dict(current), {}
    alive = {m.id for m in marines}
    desired, _ = apply_reflex({t: a for t, a in base.items() if t in alive}, marines, enemies)
    changed = {t: a for t, a in desired.items() if current.get(t) != a}
    return {**current, **desired}, changed


def zerg_targets(marines: list[UnitView], enemies: list[UnitView]) -> dict[int, int]:
    """Each Zerg's target: the nearest living Marine (ties: lowest tag)."""
    if not marines:
        return {}
    return {
        e.id: min(marines, key=lambda m: ((m.x - e.x) ** 2 + (m.y - e.y) ** 2, m.id)).id
        for e in enemies
    }


def to_view(unit) -> UnitView:
    return UnitView(
        id=unit.tag,
        kind=KINDS[unit.type_id],
        x=unit.position.x,
        y=unit.position.y,
        hp=unit.health,
        stimmed=unit.has_buff(BuffId.STIMPACK),
    )


def _hp_sum(marines) -> float:
    return float(sum(m.health for m in marines))


class ArenaBot(BotAI):
    def __init__(self, policy, run_dir: Path, seed: int, realtime: bool, recorder=None):
        super().__init__()
        self.policy = policy
        self.run_dir = run_dir
        self.seed = seed
        self.realtime = realtime
        self.recorder = recorder
        self.log = DecisionLog(run_dir / "decisions.jsonl")
        self.start_loop: int | None = None
        self.spawn_loop: int | None = None
        self.initial_enemies: int | None = None
        self.baneling_anchor: Point2 | None = None
        self.zergling_anchor: Point2 | None = None
        self.last_decision_loop: int | None = None
        self.last_enemy_order_loop: int | None = None
        self.last_camera_loop: int | None = None
        self.latencies: list[float] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self.decisions = 0
        self.late = 0
        self.api_errors = 0
        self._reflex_count = 0
        self._low_confidence = 0
        self._contact = False
        self._pre_split_active = False
        self._low_hp_marines = 0
        self._stimmed_this_step = 0
        self.stuttering: set[int] = set()
        self.stutter_steps = 0
        self.blackboard = Blackboard()
        # Realtime only: background decisions and the per-step reflex pass.
        self.scheduler: DecisionScheduler | None = (
            DecisionScheduler(lambda snap: decide_or_error(self.policy, snap["state"])) if realtime else None
        )
        self.apply_delays: list[int] = []
        self.step_reflex_changes = 0
        self._base_actions: dict[int, str] = {}
        self._current_actions: dict[int, str] = {}
        self._priority_target: int | None = None
        self.finished = False
        self._closed = False

    async def on_start(self):
        if not self.realtime:
            self.client.game_step = 1
        if self.recorder is not None:
            self.recorder.start()

    async def on_step(self, iteration: int):
        if self.finished:
            return
        if iteration == 0:
            await self._spawn()
            return
        marines = self.units(UnitTypeId.MARINE)
        enemies = self.enemy_units.of_type(ENEMY_TYPES)
        loop = self.state.game_loop
        if self.start_loop is None:
            if loop >= self.spawn_loop + config.PRE_FIGHT_WAIT_LOOPS + 3 * config.SPAWN_WAIT_LOOPS:
                # Spawn never settled into a startable state at all (e.g. marine
                # count never landed on exactly MARINE_COUNT, or no enemies ever
                # appeared) even after the pre-fight wait plus 3x the normal
                # spawn-wait timeout. Give up on this run rather than looping
                # forever.
                await self._finish(marines.amount, enemies.amount, 0, result_override="aborted")
                return
            if self.last_enemy_order_loop is None or loop - self.last_enemy_order_loop >= config.ENEMY_REORDER_INTERVAL_LOOPS:
                self.last_enemy_order_loop = loop
                self._hold_enemies(enemies)
            spawn_timed_out = loop >= self.spawn_loop + config.SPAWN_WAIT_LOOPS
            if not (
                loop >= self.spawn_loop + config.PRE_FIGHT_WAIT_LOOPS
                and marines.amount == config.MARINE_COUNT
                and (enemies.amount >= TOTAL_ENEMIES or spawn_timed_out)
                and enemies.amount > 0
            ):
                return
            self.start_loop = loop
            self.initial_enemies = enemies.amount
        elapsed = loop - self.start_loop
        if not marines or not enemies or elapsed >= config.MAX_FIGHT_LOOPS:
            await self._finish(marines.amount, enemies.amount, elapsed, hp_alive_sum=_hp_sum(marines))
            return
        if self.last_camera_loop is None or loop - self.last_camera_loop >= config.CAMERA_FOLLOW_INTERVAL_LOOPS:
            self.last_camera_loop = loop
            await self.client.move_camera(
                Point2(((marines.center.x + enemies.center.x) / 2, (marines.center.y + enemies.center.y) / 2))
            )
        if self.last_enemy_order_loop is None or loop - self.last_enemy_order_loop >= config.ENEMY_REORDER_INTERVAL_LOOPS:
            self.last_enemy_order_loop = loop
            targets = zerg_targets([to_view(m) for m in marines], [to_view(e) for e in enemies])
            for e in enemies:
                target = marines.find_by_tag(targets[e.tag])
                e.attack(target if target is not None else marines.center)
        # Two decision paths on purpose. Paused (evaluation): the game waits for the policy, so
        # the synchronous call costs no game time and results stay comparable across rounds.
        # Realtime (recordings): the game does not wait; a synchronous Jev call (~350 ms) would
        # block every on_step, stopping stutter orders and reflexes for most of each decision
        # window. So realtime runs the policy on a worker thread and applies the answer to the
        # units present when it arrives, while stutter and reflexes keep running every step.
        if self.realtime:
            self._realtime_decisions(marines, enemies, loop, elapsed)
        elif self.last_decision_loop is None or loop - self.last_decision_loop >= config.DECISION_INTERVAL_LOOPS:
            self.last_decision_loop = loop
            self._decide(marines, enemies, elapsed)
        if self.stuttering:
            self._stutter_step(marines, enemies)

    def _realtime_decisions(self, marines, enemies, loop: int, elapsed: int) -> None:
        """Realtime: apply a finished background decision, start the next one at a decision
        step (skipped while one is in flight), then run the per-step reflex pass."""
        done = self.scheduler.poll()
        if done is not None:
            self._apply_background(done.snapshot, done.result, marines, enemies, elapsed)
        if self.last_decision_loop is None or loop - self.last_decision_loop >= config.DECISION_INTERVAL_LOOPS:
            self.last_decision_loop = loop
            self._start_background([to_view(u) for u in marines], [to_view(u) for u in enemies], elapsed)
        if getattr(self.policy, "uses_blackboard", False) and self._base_actions:
            self._reflex_step(marines, enemies)

    def _start_background(self, mv: list[UnitView], ev: list[UnitView], elapsed: int) -> bool:
        """SC2-free: snapshot the state now and hand it to the scheduler."""
        state = self._build_decision_state(mv, ev, elapsed)
        return self.scheduler.maybe_start(
            {"state": state, "elapsed": elapsed, "marines_alive": len(mv), "enemies_alive": len(ev)}
        )

    def _apply_background(self, snapshot: dict, result, marines, enemies, elapsed: int) -> None:
        mv = [to_view(u) for u in marines]
        ev = [to_view(u) for u in enemies]
        _, orders = self._apply_result(snapshot, result, mv, ev, elapsed)
        self._issue_orders(orders, marines, enemies)

    def _apply_result(self, snapshot: dict, result, mv: list[UnitView], ev: list[UnitView], elapsed: int):
        """SC2-free part of applying a background decision: plan against the CURRENT units
        (`mv`, `ev`), log the record with requested/applied loops. On an API error, log it and
        keep the previous orders. Returns (decision or None, orders)."""
        d, error = result
        requested = snapshot["elapsed"]
        if d is None:
            self._record_api_error(
                error, requested, snapshot["state"], snapshot["marines_alive"], snapshot["enemies_alive"],
                requested_loop=requested, applied_loop=elapsed,
            )
            return None, []
        executed, orders = self._apply_plan(d, mv, ev)
        self.apply_delays.append(elapsed - requested)
        self._log_decision(
            d, executed, snapshot["state"], elapsed, len(mv), len(ev), requested_loop=requested, applied_loop=elapsed
        )
        return d, orders

    def _reflex_step(self, marines, enemies) -> None:
        mv = [to_view(u) for u in marines]
        ev = [to_view(u) for u in enemies]
        self._issue_orders(self._reflex_orders(mv, ev), marines, enemies)

    def _reflex_orders(self, mv: list[UnitView], ev: list[UnitView]) -> list[Order]:
        """SC2-free part of the per-step reflex pass: update the executed-action map and the
        stutter set; return orders only for Marines whose action changed."""
        desired, changed = step_reflex(self._base_actions, self._current_actions, mv, ev)
        if not changed:
            return []
        self._current_actions = desired
        self.step_reflex_changes += len(changed)
        for tag, a in changed.items():
            if a == "stutter":
                self.stuttering.add(tag)
            else:
                self.stuttering.discard(tag)
        return plan_marine_orders(changed, mv, ev, priority_target=self._priority_target)

    def _stutter_step(self, marines, enemies) -> None:
        """Every bot step: one stutter order per living stuttering Marine (weapon_cooldown == 0
        means the rifle is ready). An attack on the unit it is already attacking is not
        re-issued, so the order does not restart the shot."""
        mv = [to_view(u) for u in marines if u.tag in self.stuttering]
        ready = {u.tag: u.weapon_cooldown == 0 for u in marines if u.tag in self.stuttering}
        orders = self._stutter_orders(mv, ready, [to_view(e) for e in enemies])
        enemy_by_tag = {e.tag: e for e in enemies}
        for o in orders:
            unit = marines.find_by_tag(o.unit_id)
            if unit is None:
                continue
            if o.kind == "attack_unit":
                target = enemy_by_tag.get(o.target_id)
                if target is None:
                    unit.attack(Point2((o.x, o.y)))
                elif not (unit.is_attacking and unit.order_target == target.tag):
                    unit.attack(target)
            else:
                unit.move(Point2((o.x, o.y)))

    def _stutter_orders(self, mv: list[UnitView], ready: dict[int, bool], ev: list[UnitView]) -> list[Order]:
        """SC2-free part of a stutter step: drop dead Marines from the set, count Marine-steps,
        return the orders."""
        self.stuttering &= {m.id for m in mv}
        self.stutter_steps += len(self.stuttering)
        return stutter_step_orders(self.stuttering, mv, ready, ev)

    def _hold_enemies(self, enemies) -> None:
        """Before the fight starts, the built-in AI otherwise drifts idle
        Banelings/Zerglings away from their spawn point. Order each visible
        one back to its own kind's anchor so they stay put and in frame."""
        for e in enemies:
            if e.type_id == UnitTypeId.BANELING:
                e.move(self.baneling_anchor)
            elif e.type_id == UnitTypeId.ZERGLING:
                e.move(self.zergling_anchor)

    async def _spawn(self):
        self.spawn_loop = self.state.game_loop
        # toggle: reveals the map; do not combine with run_game(disable_fog=True)
        await self.client.debug_show_map()
        await self.client.debug_upgrade()
        await self.client.debug_control_enemy()
        c = Point2(config.CENTER)
        self.baneling_anchor = c + Point2(config.BANELING_OFFSET)
        self.zergling_anchor = c + Point2(config.ZERGLING_OFFSET)
        await self.client.debug_create_unit([
            [UnitTypeId.MARINE, config.MARINE_COUNT, c + Point2(config.MARINE_OFFSET), 1],
            [UnitTypeId.BANELING, config.BANELING_COUNT, self.baneling_anchor, 2],
            [UnitTypeId.ZERGLING, config.ZERGLING_COUNT, self.zergling_anchor, 2],
        ])
        await self.client.move_camera(Point2(config.CENTER))

    def _decide(self, marines, enemies, elapsed: int) -> None:
        mv = [to_view(u) for u in marines]
        ev = [to_view(u) for u in enemies]
        state, d, executed, orders = self._plan_step(mv, ev, elapsed)
        if d is None:
            return  # API error already logged; units keep their previous orders.
        self._issue_orders(orders, marines, enemies)
        self._log_decision(d, executed, state, elapsed, len(mv), len(ev))

    def _issue_orders(self, orders: list[Order], marines, enemies) -> None:
        if not orders:
            return
        enemy_by_tag = {e.tag: e for e in enemies}
        for o in orders:
            unit = marines.find_by_tag(o.unit_id)
            if unit is None:
                continue
            if o.kind == "move":
                unit.move(Point2((o.x, o.y)))
            elif o.kind == "attack":
                unit.attack(Point2((o.x, o.y)))
            elif o.kind == "attack_unit":
                target = enemy_by_tag.get(o.target_id)
                unit.attack(target if target is not None else Point2((o.x, o.y)))
            elif o.kind == "kite":
                unit.move(Point2((o.x, o.y)))
                unit.attack(Point2((o.tx, o.ty)), queue=True)
            else:
                unit(AbilityId.EFFECT_STIM_MARINE)

    def _build_decision_state(self, mv: list[UnitView], ev: list[UnitView], elapsed: int) -> dict:
        if getattr(self.policy, "uses_blackboard", False):
            return build_commander_state(mv, ev, elapsed, self.blackboard)
        return build_state(mv, ev, elapsed)

    def _plan_step(self, mv: list[UnitView], ev: list[UnitView], elapsed: int):
        """SC2-free part of a decision step: state, policy call, executed actions, orders.
        Updates the blackboard with the executed actions. On a transient API error, logs an
        api_error record and returns (state, None, None, [])."""
        state = self._build_decision_state(mv, ev, elapsed)
        d, error = decide_or_error(self.policy, state)
        if d is None:
            self._record_api_error(error, elapsed, state, len(mv), len(ev))
            return state, None, None, []
        executed, orders = self._apply_plan(d, mv, ev)
        return state, d, executed, orders

    def _apply_plan(self, d, mv: list[UnitView], ev: list[UnitView]):
        """Executed actions and orders for decision `d` against the units `mv`, `ev`. Updates
        the blackboard, the stutter set and the executed-action map. Returns (executed, orders)."""
        reflex = bool(getattr(self.policy, "uses_blackboard", False))
        stats: dict = {}
        if d.marine_actions is not None:
            executed = execute_actions(
                d.marine_actions, mv, ev, d.squad_plan, d.priority_target, reflex=reflex, stats=stats,
                confidences=d.marine_confidences if reflex else None,
                min_confidence=config.SOLDIER_MIN_CONFIDENCE if reflex else None,
                stutter_default=bool(getattr(self.policy, "stutter_default", False)),
            )
            orders = plan_marine_orders(executed, mv, ev, priority_target=d.priority_target)
            if d.stim_now:
                # Stim first: it is instant and does not replace the Marine's action order.
                stim = squad_stim_orders(mv)
                stats["stimmed_this_step"] = len(stim)
                orders = stim + orders
        else:
            executed = None
            orders = plan_orders(d.action, mv, ev)
        if hasattr(self.policy, "observe_executed"):
            self.policy.observe_executed(executed)
        self.blackboard.record(mv, ev, executed if executed is not None else {m.id: d.action for m in mv})
        self._reflex_count = stats.get("reflex_count", 0)
        self._low_confidence = stats.get("low_confidence_marines", 0)
        self._contact = stats.get("contact", in_contact(mv, ev))
        self._pre_split_active = stats.get("pre_split_active", False)
        self._stimmed_this_step = stats.get("stimmed_this_step", 0)
        self._low_hp_marines = sum(1 for m in mv if m.hp <= config.LOW_HP)
        self.stuttering = stutter_tags(executed)
        self._base_actions = dict(executed or {})
        self._current_actions = dict(executed or {})
        self._priority_target = d.priority_target
        return executed, orders

    def _log_decision(
        self,
        d,
        executed: dict[int, str] | None,
        state: dict,
        elapsed: int,
        marines_alive: int,
        enemies_alive: int,
        requested_loop: int | None = None,
        applied_loop: int | None = None,
    ) -> None:
        delay = applied_loop - requested_loop if requested_loop is not None and applied_loop is not None else None
        if delay is not None:
            late = delay > config.DECISION_INTERVAL_LOOPS
        else:
            late = self.realtime and d.latency_ms > config.DECISION_BUDGET_MS
        self.decisions += 1
        self.late += int(late)
        self.input_tokens += d.input_tokens
        self.output_tokens += d.output_tokens
        if d.model is not None:
            self.latencies.append(d.latency_ms)
        self.log.write({
            "fight_loop": elapsed,
            "wall_time": time.time(),
            "policy": self.policy.name,
            "action": d.action,
            "probabilities": d.probabilities,
            "confidence": d.confidence,
            "latency_ms": round(d.latency_ms, 1),
            "input_tokens": d.input_tokens,
            "output_tokens": d.output_tokens,
            "late": late,
            "marine_actions": d.marine_actions,
            "executed_actions": executed,
            "reflex_count": self._reflex_count,
            "low_confidence_marines": self._low_confidence,
            "contact": self._contact,
            "pre_split_active": self._pre_split_active,
            "low_hp_marines": self._low_hp_marines,
            "stimmed_this_step": self._stimmed_this_step,
            "stuttering": len(self.stuttering),
            "plan_kept_low_confidence": d.plan_kept_low_confidence,
            "stim_now": d.stim_now,
            "marine_confidences": d.marine_confidences,
            "squad_plan": d.squad_plan,
            "semantic_activations": d.semantic_activations,
            "priority_target": d.priority_target,
            "commander_latency_ms": round(d.commander_latency_ms, 1) if d.commander_latency_ms is not None else None,
            "soldier_latency_ms": round(d.soldier_latency_ms, 1) if d.soldier_latency_ms is not None else None,
            "marines_alive": marines_alive,
            "enemies_alive": enemies_alive,
            "requested_loop": requested_loop,
            "applied_loop": applied_loop,
            "apply_delay_loops": delay,
            "state": state,
        })

    def _record_api_error(
        self,
        error: str,
        elapsed: int,
        state: dict,
        marines_alive: int,
        enemies_alive: int,
        requested_loop: int | None = None,
        applied_loop: int | None = None,
    ) -> None:
        delay = applied_loop - requested_loop if requested_loop is not None and applied_loop is not None else None
        self.api_errors += 1
        self.log.write({
            "fight_loop": elapsed,
            "wall_time": time.time(),
            "policy": self.policy.name,
            "action": "api_error",
            "error": error,
            "probabilities": None,
            "confidence": None,
            "latency_ms": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "late": False,
            "marine_actions": None,
            "executed_actions": None,
            "reflex_count": 0,
            "low_confidence_marines": 0,
            "contact": None,
            "pre_split_active": None,
            "low_hp_marines": None,
            "stimmed_this_step": 0,
            "stuttering": len(self.stuttering),
            "plan_kept_low_confidence": None,
            "stim_now": None,
            "marine_confidences": None,
            "squad_plan": None,
            "semantic_activations": None,
            "priority_target": None,
            "commander_latency_ms": None,
            "soldier_latency_ms": None,
            "marines_alive": marines_alive,
            "enemies_alive": enemies_alive,
            "requested_loop": requested_loop,
            "applied_loop": applied_loop,
            "apply_delay_loops": delay,
            "state": state,
        })

    async def _finish(
        self,
        marines_alive: int,
        enemies_alive: int,
        elapsed: int,
        result_override: str | None = None,
        leave: bool = True,
        hp_alive_sum: float = 0.0,
    ) -> None:
        self.finished = True
        self._close_scheduler()
        if result_override is not None:
            result = result_override
        elif enemies_alive == 0 and marines_alive > 0:
            result = "win"
        elif marines_alive == 0:
            result = "loss"
        else:
            result = "timeout"
        initial_enemies = self.initial_enemies if self.initial_enemies is not None else enemies_alive
        enemies_killed = initial_enemies - enemies_alive
        reward = (enemies_killed if marines_alive > 0 else 0) + 0.25 * marines_alive + 0.01 * hp_alive_sum
        if hasattr(self.policy, "end_episode"):
            self.policy.end_episode(reward)
        summary = {
            "policy": self.policy.name,
            "seed": self.seed,
            "realtime": self.realtime,
            "result": result,
            "marines_alive": marines_alive,
            "enemies_alive": enemies_alive,
            "initial_enemies": initial_enemies,
            "enemies_killed": enemies_killed,
            "training_reward": round(reward, 3),
            "training_advantage": getattr(self.policy, "last_advantage", None),
            "fight_seconds": round(elapsed / config.LOOPS_PER_SECOND, 2),
            "decisions": self.decisions,
            "late_decisions": self.late,
            "api_errors": self.api_errors,
            "latencies_ms": [round(x, 1) for x in self.latencies],
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "stutter_steps": self.stutter_steps,
            "skipped_decisions": self.scheduler.skipped if self.scheduler is not None else 0,
            "mean_apply_delay_loops": (
                round(sum(self.apply_delays) / len(self.apply_delays), 2) if self.realtime and self.apply_delays else None
            ),
            "step_reflex_changes": self.step_reflex_changes,
            "hp_alive_sum": round(hp_alive_sum, 1),
            "record_start_wall": getattr(self.recorder, "start_wall", None),
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self.log.close()
        if leave:
            await self.client.leave()

    def close(self) -> None:
        """Idempotent cleanup: safe to call multiple times, from on_end and/or
        a caller's try/finally around run_game()."""
        if self._closed:
            return
        self._closed = True
        self._close_scheduler()
        self.log.close()
        if self.recorder is not None:
            self.recorder.stop()

    def _close_scheduler(self) -> None:
        """Drop any in-flight background decision without waiting; never raises."""
        if self.scheduler is not None:
            try:
                self.scheduler.close()
            except Exception:
                pass

    async def on_end(self, game_result: Result):
        if not self.finished:
            # The SC2 engine can decide the game is over (e.g. all our Marines
            # died) in the same observation that on_step never gets called
            # for, so _finish() above never ran. Write the summary here from
            # the last state on_step did see, using the engine's own result.
            marines = self.units(UnitTypeId.MARINE)
            enemies = self.enemy_units.of_type(ENEMY_TYPES)
            if self.start_loop is None:
                # No fight ever started; the game ended anyway (e.g. resigned
                # or errored out during spawn). The engine's game_result isn't
                # a meaningful win/loss/timeout for a fight that never began,
                # and calling client.leave() here (after the game has already
                # ended) is unnecessary, so skip both.
                await self._finish(marines.amount, enemies.amount, 0, result_override="aborted", leave=False)
            else:
                elapsed = self.state.game_loop - self.start_loop
                await self._finish(
                    marines.amount,
                    enemies.amount,
                    elapsed,
                    RESULT_MAP.get(game_result, "timeout"),
                    hp_alive_sum=_hp_sum(marines),
                )
        self.close()
