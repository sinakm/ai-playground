"""Turn squad or per-Marine actions into per-Marine orders. Pure geometry, no SC2."""

from __future__ import annotations

import math
from dataclasses import dataclass

from arena import config
from arena.state import banelings_within, centroid, closest_to_banelings
from arena.views import UnitView


@dataclass(frozen=True)
class Order:
    unit_id: int
    kind: str
    x: float | None = None
    y: float | None = None
    tx: float | None = None
    ty: float | None = None
    target_id: int | None = None


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


def _attack_unit(m: UnitView, target: UnitView) -> Order:
    return Order(m.id, "attack_unit", target.x, target.y, target_id=target.id)


def _focus_target(marines: list[UnitView], enemies: list[UnitView], priority_target: int | None) -> UnitView | None:
    banes = [e for e in enemies if e.kind == "baneling"]
    for b in banes:
        if b.id == priority_target:
            return b
    if not banes:
        return None
    cx, cy = centroid(marines)
    return min(banes, key=lambda b: (b.x - cx) ** 2 + (b.y - cy) ** 2)


def _cover_target(m: UnitView, marines: list[UnitView], enemies: list[UnitView]) -> UnitView | None:
    near = [
        o for o in marines
        if o.id != m.id and math.hypot(o.x - m.x, o.y - m.y) <= config.COVER_RADIUS
    ]
    if not near:
        return None
    weakest = min(near, key=lambda o: (o.hp, math.hypot(o.x - m.x, o.y - m.y)))
    return _nearest(weakest, enemies)


def _bait_target(m: UnitView, marines: list[UnitView], enemies: list[UnitView]) -> tuple[float, float] | None:
    cx, cy = centroid(marines)
    ux, uy = _unit_vector(m.x - cx, m.y - cy)
    if (ux, uy) == (0.0, 0.0):
        # Alone or at the center: run straight away from the nearest threat instead.
        t = _nearest(m, [e for e in enemies if e.kind == "baneling"] or enemies)
        ux, uy = _unit_vector(m.x - t.x, m.y - t.y)
        if (ux, uy) == (0.0, 0.0):
            return None
    return (m.x + ux * config.BAIT_DISTANCE, m.y + uy * config.BAIT_DISTANCE)


def _marine_order(
    action: str,
    m: UnitView,
    marines: list[UnitView],
    enemies: list[UnitView],
    ex: float,
    ey: float,
    priority_target: int | None = None,
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
    if action == "focus_bane":
        t = _focus_target(marines, enemies, priority_target)
        return _attack_unit(m, t) if t is not None else attack
    if action == "cover_ally":
        t = _cover_target(m, marines, enemies)
        return _attack_unit(m, t) if t is not None else attack
    if action == "bait":
        target = _bait_target(m, marines, enemies)
        return Order(m.id, "move", target[0], target[1]) if target is not None else attack
    if action == "kite":
        threat = _nearest(m, [e for e in enemies if e.kind == "baneling"] or enemies)
        target = _step_away(m, threat, config.KITE_STEP)
    else:  # split
        others = [o for o in marines if o.id != m.id]
        target = _step_away(m, _nearest(m, others), config.SPLIT_STEP) if others else None
    if target is None:
        return attack
    return Order(m.id, "kite", target[0], target[1], tx=ex, ty=ey)


def resolve_bait(actions: dict[int, str], marines: list[UnitView], enemies: list[UnitView]) -> dict[int, str]:
    """At most one Marine baits per step: the one closest to the Banelings among those that
    chose it (any enemy if no Baneling is alive). Extra `bait` choices become `kite`."""
    by_id = {m.id: m for m in marines}
    baiters = [tag for tag, a in actions.items() if a == "bait" and tag in by_id]
    if len(baiters) <= 1 or not enemies:
        return dict(actions)
    threats = [e for e in enemies if e.kind == "baneling"] or enemies

    def threat_distance(tag: int) -> float:
        m = by_id[tag]
        return min(math.hypot(m.x - t.x, m.y - t.y) for t in threats)

    keep = min(baiters, key=lambda tag: (threat_distance(tag), tag))
    return {tag: ("kite" if a == "bait" and tag != keep else a) for tag, a in actions.items()}


def assign_roles(actions: dict[int, str], marines: list[UnitView], enemies: list[UnitView]) -> dict[int, str]:
    """bait_and_split roles, set in code: the Marine closest to the Banelings baits (overriding
    its Jev choice); any other Marine that chose focus_bane or attack with a Baneling within
    3 cells splits instead. Only Marines that have an action are touched."""
    bait_tag = closest_to_banelings(marines, enemies)
    if bait_tag is None:
        return dict(actions)
    by_id = {m.id: m for m in marines}
    out = {}
    for tag, a in actions.items():
        if tag == bait_tag:
            out[tag] = "bait"
        elif a in ("focus_bane", "attack") and tag in by_id and banelings_within(by_id[tag], enemies) > 0:
            out[tag] = "split"
        else:
            out[tag] = a
    return out


def cap_focus_bane(
    actions: dict[int, str], marines: list[UnitView], enemies: list[UnitView], priority_target: int | None
) -> dict[int, str]:
    """At most FOCUS_BANE_CAP Marines execute focus_bane: the closest to the focus target among
    those that chose it (ties: lowest tag). The rest attack."""
    by_id = {m.id: m for m in marines}
    chose = [tag for tag, a in actions.items() if a == "focus_bane" and tag in by_id]
    if len(chose) <= config.FOCUS_BANE_CAP or not marines:
        return dict(actions)
    target = _focus_target(marines, enemies, priority_target)
    if target is None:
        return dict(actions)
    chose.sort(key=lambda tag: (math.hypot(by_id[tag].x - target.x, by_id[tag].y - target.y), tag))
    keep = set(chose[: config.FOCUS_BANE_CAP])
    return {tag: ("attack" if a == "focus_bane" and tag not in keep else a) for tag, a in actions.items()}


def execute_actions(
    actions: dict[int, str],
    marines: list[UnitView],
    enemies: list[UnitView],
    squad_plan: str | None = None,
    priority_target: int | None = None,
) -> dict[int, str]:
    """The actions Marines actually execute this step: plan-driven roles (bait_and_split only),
    then one bait per step, then the focus_bane cap."""
    if not marines or not enemies:
        return dict(actions)
    if squad_plan == "bait_and_split":
        actions = assign_roles(actions, marines, enemies)
    actions = resolve_bait(actions, marines, enemies)
    return cap_focus_bane(actions, marines, enemies, priority_target)


def plan_marine_orders(
    actions: dict[int, str],
    marines: list[UnitView],
    enemies: list[UnitView],
    priority_target: int | None = None,
) -> list[Order]:
    """One order per Marine that has an action. Kite/split reuse the `kite` order kind:
    move a step away, then queued attack-move to the enemy centroid. focus_bane and
    cover_ally attack a specific unit (`attack_unit`); bait is a plain move."""
    for action in actions.values():
        if action not in config.MARINE_ACTIONS:
            raise ValueError(f"unknown marine action: {action}")
    if not marines or not enemies:
        return []
    actions = resolve_bait(actions, marines, enemies)
    ex, ey = centroid(enemies)
    orders = []
    for m in marines:
        action = actions.get(m.id)
        if action is None:
            continue
        o = _marine_order(action, m, marines, enemies, ex, ey, priority_target)
        if o is not None:
            orders.append(o)
    return orders
