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
from arena.actions import plan_marine_orders, plan_orders
from arena.log import DecisionLog
from arena.state import build_state
from arena.views import UnitView

KINDS = {UnitTypeId.MARINE: "marine", UnitTypeId.BANELING: "baneling", UnitTypeId.ZERGLING: "zergling"}
ENEMY_TYPES = {UnitTypeId.BANELING, UnitTypeId.ZERGLING}
TOTAL_ENEMIES = config.BANELING_COUNT + config.ZERGLING_COUNT
# Fallback for the case handled in on_end: the SC2 engine can declare a Result
# for our player (e.g. all Marines dead) in the same observation it stops
# calling on_step for, so on_step's own "not marines" / "not enemies" branch
# never runs. See ArenaBot.on_end.
RESULT_MAP = {Result.Victory: "win", Result.Defeat: "loss", Result.Tie: "timeout", Result.Undecided: "timeout"}


def to_view(unit) -> UnitView:
    return UnitView(
        id=unit.tag,
        kind=KINDS[unit.type_id],
        x=unit.position.x,
        y=unit.position.y,
        hp=unit.health,
        stimmed=unit.has_buff(BuffId.STIMPACK),
    )


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
            await self._finish(marines.amount, enemies.amount, elapsed)
            return
        if self.last_camera_loop is None or loop - self.last_camera_loop >= config.CAMERA_FOLLOW_INTERVAL_LOOPS:
            self.last_camera_loop = loop
            await self.client.move_camera(
                Point2(((marines.center.x + enemies.center.x) / 2, (marines.center.y + enemies.center.y) / 2))
            )
        if self.last_enemy_order_loop is None or loop - self.last_enemy_order_loop >= config.ENEMY_REORDER_INTERVAL_LOOPS:
            self.last_enemy_order_loop = loop
            for e in enemies:
                e.attack(marines.center)
        if self.last_decision_loop is None or loop - self.last_decision_loop >= config.DECISION_INTERVAL_LOOPS:
            self.last_decision_loop = loop
            self._decide(marines, enemies, elapsed)

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
        state = build_state(mv, ev, elapsed)
        d = self.policy.decide(state)
        orders = (
            plan_marine_orders(d.marine_actions, mv, ev) if d.marine_actions is not None else plan_orders(d.action, mv, ev)
        )
        for o in orders:
            unit = marines.find_by_tag(o.unit_id)
            if unit is None:
                continue
            if o.kind == "move":
                unit.move(Point2((o.x, o.y)))
            elif o.kind == "attack":
                unit.attack(Point2((o.x, o.y)))
            elif o.kind == "kite":
                unit.move(Point2((o.x, o.y)))
                unit.attack(Point2((o.tx, o.ty)), queue=True)
            else:
                unit(AbilityId.EFFECT_STIM_MARINE)
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
            "marine_confidences": d.marine_confidences,
            "marines_alive": len(mv),
            "enemies_alive": len(ev),
            "state": state,
        })

    async def _finish(
        self,
        marines_alive: int,
        enemies_alive: int,
        elapsed: int,
        result_override: str | None = None,
        leave: bool = True,
    ) -> None:
        self.finished = True
        if result_override is not None:
            result = result_override
        elif enemies_alive == 0 and marines_alive > 0:
            result = "win"
        elif marines_alive == 0:
            result = "loss"
        else:
            result = "timeout"
        initial_enemies = self.initial_enemies if self.initial_enemies is not None else enemies_alive
        summary = {
            "policy": self.policy.name,
            "seed": self.seed,
            "realtime": self.realtime,
            "result": result,
            "marines_alive": marines_alive,
            "enemies_alive": enemies_alive,
            "initial_enemies": initial_enemies,
            "enemies_killed": initial_enemies - enemies_alive,
            "fight_seconds": round(elapsed / config.LOOPS_PER_SECOND, 2),
            "decisions": self.decisions,
            "late_decisions": self.late,
            "latencies_ms": [round(x, 1) for x in self.latencies],
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
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
        self.log.close()
        if self.recorder is not None:
            self.recorder.stop()

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
                await self._finish(marines.amount, enemies.amount, elapsed, RESULT_MAP.get(game_result, "timeout"))
        self.close()
