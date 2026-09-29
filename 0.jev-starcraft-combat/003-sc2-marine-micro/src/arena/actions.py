"""Turn squad or per-Marine actions into per-Marine orders. Pure geometry, no SC2."""

from __future__ import annotations

import math
from dataclasses import dataclass

from arena import config
from arena.state import centroid
from arena.views import UnitView


@dataclass(frozen=True)
class Order:
    unit_id: int
    kind: str
    x: float | None = None
    y: float | None = None
    tx: float | None = None
    ty: float | None = None


def _unit_vector(dx: float, dy: float) -> tuple[float, float]:
    n = math.hypot(dx, dy)
    return (0.0, 0.0) if n == 0 else (dx / n, dy / n)


def _nearest(u: UnitView, others: list[UnitView]) -> UnitView:
    return min(others, key=lambda o: (o.x - u.x) ** 2 + (o.y - u.y) ** 2)


def _spread(marines: list[UnitView], enemies: list[UnitView], ex: float, ey: float) -> list[Order]:
    threats = [e for e in enemies if e.kind == "baneling"] or enemies
    orders = []
    for m in marines:
        t = _nearest(m, threats)
        vx, vy = _unit_vector(m.x - t.x, m.y - t.y)
        others = [o for o in marines if o.id != m.id]
        if others:
            n = _nearest(m, others)
            wx, wy = _unit_vector(m.x - n.x, m.y - n.y)
            vx, vy = vx + wx, vy + wy
        ux, uy = _unit_vector(vx, vy)
        if (ux, uy) == (0.0, 0.0):
            orders.append(Order(m.id, "attack", ex, ey))
        else:
            x, y = m.x + ux * config.SPREAD_STEP, m.y + uy * config.SPREAD_STEP
            orders.append(Order(m.id, "kite", x, y, tx=ex, ty=ey))
    return orders


def plan_orders(action: str, marines: list[UnitView], enemies: list[UnitView]) -> list[Order]:
    if action not in config.ACTIONS:
        raise ValueError(f"unknown action: {action}")
    if not marines or not enemies:
        return []
    ex, ey = centroid(enemies)
    if action == "attack":
        return [Order(m.id, "attack", ex, ey) for m in marines]
    if action == "clump":
        cx, cy = centroid(marines)
        return [Order(m.id, "move", cx, cy) for m in marines]
    if action == "retreat":
        orders = []
        for m in marines:
            ux, uy = _unit_vector(m.x - ex, m.y - ey)
            orders.append(Order(m.id, "move", m.x + ux * config.RETREAT_DISTANCE, m.y + uy * config.RETREAT_DISTANCE))
        return orders
    if action == "stim":
        return [Order(m.id, "stim") for m in marines if not m.stimmed and m.hp > config.STIM_MIN_HP]
    return _spread(marines, enemies, ex, ey)


def _step_away(m: UnitView, threat: UnitView, step: float) -> tuple[float, float] | None:
    ux, uy = _unit_vector(m.x - threat.x, m.y - threat.y)
    if (ux, uy) == (0.0, 0.0):
        return None
    return (m.x + ux * step, m.y + uy * step)


def _marine_order(
    action: str, m: UnitView, marines: list[UnitView], enemies: list[UnitView], ex: float, ey: float
) -> Order | None:
    nearest_enemy = _nearest(m, enemies)
    attack = Order(m.id, "attack", nearest_enemy.x, nearest_enemy.y)
    if action == "attack":
        return attack
    if action == "stim":
        return Order(m.id, "stim") if not m.stimmed and m.hp > config.STIM_MIN_HP else None
    if action == "retreat":
        ux, uy = _unit_vector(m.x - ex, m.y - ey)
        d = config.MARINE_RETREAT_DISTANCE
        return Order(m.id, "move", m.x + ux * d, m.y + uy * d)
    if action == "kite":
        threat = _nearest(m, [e for e in enemies if e.kind == "baneling"] or enemies)
        target = _step_away(m, threat, config.KITE_STEP)
    else:  # split
        others = [o for o in marines if o.id != m.id]
        target = _step_away(m, _nearest(m, others), config.SPLIT_STEP) if others else None
    if target is None:
        return attack
    return Order(m.id, "kite", target[0], target[1], tx=ex, ty=ey)


def plan_marine_orders(actions: dict[int, str], marines: list[UnitView], enemies: list[UnitView]) -> list[Order]:
    """One order per Marine that has an action. Kite/split reuse the `kite` order kind:
    move a step away, then queued attack-move to the enemy centroid."""
    for action in actions.values():
        if action not in config.MARINE_ACTIONS:
            raise ValueError(f"unknown marine action: {action}")
    if not marines or not enemies:
        return []
    ex, ey = centroid(enemies)
    orders = []
    for m in marines:
        action = actions.get(m.id)
        if action is None:
            continue
        o = _marine_order(action, m, marines, enemies, ex, ey)
        if o is not None:
            orders.append(o)
    return orders
