"""Turn one squad action into per-Marine orders. Pure geometry, no SC2."""

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
