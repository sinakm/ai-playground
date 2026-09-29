"""Turn squad or per-Marine actions into per-Marine orders. Pure geometry, no SC2."""

from __future__ import annotations

import math
from dataclasses import dataclass

from arena import config
from arena.state import (
    banelings_within,
    centroid,
    closest_to_banelings,
    in_contact,
    marines_within,
    zerglings_within,
)
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
        # Range guard: only shoot the focus Baneling when it is already in range; walking to
        # it means walking into Banelings.
        t = _focus_target(marines, enemies, priority_target)
        if t is None or math.hypot(t.x - m.x, t.y - m.y) > config.FOCUS_RANGE + 1e-9:
            return attack
        return _attack_unit(m, t)
    if action == "cover_ally":
        t = _cover_target(m, marines, enemies)
        return _attack_unit(m, t) if t is not None else attack
    if action == "retreat_to_squad":
        others = [o for o in marines if o.id != m.id]
        if not others:
            return attack
        cx, cy = centroid(others)
        d = math.hypot(cx - m.x, cy - m.y)
        if d == 0:
            return attack
        step = min(config.REGROUP_STEP, d)
        ux, uy = (cx - m.x) / d, (cy - m.y) / d
        return Order(m.id, "kite", m.x + ux * step, m.y + uy * step, tx=ex, ty=ey)
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
    3 cells splits instead, but only if another Marine is within CROWDED_DISTANCE.
    Only Marines that have an action are touched."""
    bait_tag = closest_to_banelings(marines, enemies)
    if bait_tag is None:
        return dict(actions)
    by_id = {m.id: m for m in marines}
    out = {}
    for tag, a in actions.items():
        if tag == bait_tag:
            out[tag] = "bait"
        elif (
            a in ("focus_bane", "attack")
            and tag in by_id
            and banelings_within(by_id[tag], enemies) > 0
            and _crowded(by_id[tag], marines)
        ):
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


def _crowded(m: UnitView, marines: list[UnitView]) -> bool:
    return marines_within(m, marines, config.CROWDED_DISTANCE) > 0


def pre_split(actions: dict[int, str], marines: list[UnitView]) -> dict[int, str]:
    """pre_split plan (before contact only): crowded Marines that chose attack or focus_bane split."""
    by_id = {m.id: m for m in marines}
    return {
        tag: ("split" if a in ("attack", "focus_bane") and tag in by_id and _crowded(by_id[tag], marines) else a)
        for tag, a in actions.items()
    }


def apply_reflex(
    actions: dict[int, str], marines: list[UnitView], enemies: list[UnitView]
) -> tuple[dict[int, str], int]:
    """Baneling reflex: any Marine with a Baneling within REFLEX_KITE_DISTANCE kites.
    Then the Zergling swarm reflex: a Marine with at least ZERGLING_SWARM_COUNT Zerglings within
    ZERGLING_SWARM_RADIUS (and no Baneling that close) executes retreat_to_squad.
    Returns the new actions and how many Marines the reflexes changed."""
    banes = [e for e in enemies if e.kind == "baneling"]
    by_id = {m.id: m for m in marines}
    out, count = {}, 0
    for tag, a in actions.items():
        m = by_id.get(tag)
        new = a
        if m is not None:
            if any(math.hypot(b.x - m.x, b.y - m.y) <= config.REFLEX_KITE_DISTANCE for b in banes):
                # Group escape: a Marine in a clump retreats (3 cells from the enemy centroid)
                # instead of kiting a single step.
                grouped = marines_within(m, marines, config.GROUP_ESCAPE_RADIUS) >= config.GROUP_ESCAPE_NEIGHBORS
                new = "retreat" if grouped else "kite"
            elif zerglings_within(m, enemies) >= config.ZERGLING_SWARM_COUNT:
                new = "retreat_to_squad"
            elif m.hp <= config.LOW_HP and zerglings_within(m, enemies) > 0:
                # Low-HP reflex: a wounded Marine with a Zergling on it falls back to the squad.
                new = "retreat_to_squad"
        out[tag] = new
        count += int(new != a)
    return out, count


def guard_cover_ally(actions: dict[int, str], marines: list[UnitView], enemies: list[UnitView]) -> dict[int, str]:
    """A Marine with a Baneling within 3 cells cannot execute cover_ally; it kites instead."""
    by_id = {m.id: m for m in marines}
    return {
        tag: (
            "kite"
            if a == "cover_ally"
            and tag in by_id
            and banelings_within(by_id[tag], enemies, config.COVER_ALLY_BANELING_GUARD) > 0
            else a
        )
        for tag, a in actions.items()
    }


def gate_low_confidence(
    actions: dict[int, str],
    marines: list[UnitView],
    enemies: list[UnitView],
    confidences: dict[int, float],
    min_confidence: float,
) -> tuple[dict[int, str], int]:
    """Replace choices Jev was unsure about (confidence < min_confidence) with a safe default:
    kite if a Baneling is within 3 cells, else retreat at LOW_HP or less, else attack.
    Returns new actions and how many changed."""
    by_id = {m.id: m for m in marines}
    out, count = {}, 0
    for tag, a in actions.items():
        c = confidences.get(tag)
        if c is not None and c < min_confidence and tag in by_id:
            near = banelings_within(by_id[tag], enemies, config.LOW_CONFIDENCE_KITE_DISTANCE) > 0
            m = by_id[tag]
            out[tag] = "kite" if near else "retreat" if m.hp <= config.LOW_HP else "attack"
            count += 1
        else:
            out[tag] = a
    return out, count


def squad_stim_orders(marines: list[UnitView]) -> list[Order]:
    """Commander stim_now = yes: stim every unstimmed Marine above COMMANDER_STIM_MIN_HP.
    `stimmed` is the live Stimpack buff, so a Marine can be stimmed again once it wears off."""
    return [Order(m.id, "stim") for m in marines if not m.stimmed and m.hp > config.COMMANDER_STIM_MIN_HP]


def execute_actions(
    actions: dict[int, str],
    marines: list[UnitView],
    enemies: list[UnitView],
    squad_plan: str | None = None,
    priority_target: int | None = None,
    reflex: bool = False,
    stats: dict | None = None,
    confidences: dict[int, float] | None = None,
    min_confidence: float | None = None,
) -> dict[int, str]:
    """The actions Marines actually execute this step: plan-driven roles (bait_and_split,
    pre_split), then the reflex kite (commander only), then one bait per step, then the
    focus_bane cap. With `min_confidence`, low-confidence choices are first replaced by a safe
    default. `stats` gets `reflex_count` and `low_confidence_marines`."""
    count = low = 0
    contact = in_contact(marines, enemies)
    pre_split_active = squad_plan == "pre_split" and not contact
    if marines and enemies:
        if min_confidence is not None and confidences:
            actions, low = gate_low_confidence(actions, marines, enemies, confidences, min_confidence)
        if squad_plan == "bait_and_split":
            actions = assign_roles(actions, marines, enemies)
        elif pre_split_active:
            actions = pre_split(actions, marines)
        if reflex:
            actions, count = apply_reflex(actions, marines, enemies)
        actions = guard_cover_ally(actions, marines, enemies)
        actions = resolve_bait(actions, marines, enemies)
        actions = cap_focus_bane(actions, marines, enemies, priority_target)
    else:
        actions = dict(actions)
    if stats is not None:
        stats["reflex_count"] = count
        stats["low_confidence_marines"] = low
        stats["contact"] = contact
        stats["pre_split_active"] = pre_split_active
    return actions


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
        if action not in config.MARINE_ACTIONS and action not in config.EXECUTED_ONLY_ACTIONS:
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
