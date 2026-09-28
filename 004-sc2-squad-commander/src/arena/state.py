"""Turn unit views into the JSON state Jev reads."""

from __future__ import annotations

import math

from arena import config
from arena.views import UnitView


def centroid(units: list[UnitView]) -> tuple[float, float]:
    if not units:
        raise ValueError("centroid of empty unit list")
    n = len(units)
    return (sum(u.x for u in units) / n, sum(u.y for u in units) / n)


def distance(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def _nearest_distance(u: UnitView, others: list[UnitView]) -> float | None:
    if not others:
        return None
    return round(min(distance(u.x, u.y, o.x, o.y) for o in others), 1)


def build_state(marines: list[UnitView], enemies: list[UnitView], fight_loop: int) -> dict:
    center = centroid(marines) if marines else None
    banes = [e for e in enemies if e.kind == "baneling"]
    spread = (
        sum(distance(m.x, m.y, *center) for m in marines) / len(marines) if center else 0.0
    )
    nearest_bane = (
        min(distance(b.x, b.y, m.x, m.y) for b in banes for m in marines)
        if banes and marines
        else None
    )
    nearest_bane_to_center = (
        min(distance(b.x, b.y, *center) for b in banes) if banes and center else None
    )
    marines_in_splash_danger = sum(
        1
        for m in marines
        if any(distance(m.x, m.y, b.x, b.y) <= config.SPLASH_DANGER_RADIUS for b in banes)
    )
    seconds_left = round((config.MAX_FIGHT_LOOPS - fight_loop) / config.LOOPS_PER_SECOND, 1)
    return {
        "game_time_s": round(fight_loop / config.LOOPS_PER_SECOND, 1),
        "marines": [
            {
                "id": m.id,
                "x": round(m.x, 1),
                "y": round(m.y, 1),
                "hp": round(m.hp),
                "stimmed": m.stimmed,
                "nearest_baneling_distance": _nearest_distance(m, banes),
                "nearest_marine_distance": _nearest_distance(m, [o for o in marines if o.id != m.id]),
                "nearest_enemy_distance": _nearest_distance(m, enemies),
            }
            for m in marines
        ],
        "enemies": [
            {
                "id": e.id,
                "kind": e.kind,
                "x": round(e.x, 1),
                "y": round(e.y, 1),
                "hp": round(e.hp),
                "distance_to_squad": round(distance(e.x, e.y, *center), 1) if center else None,
            }
            for e in enemies
        ],
        "summary": {
            "marines_alive": len(marines),
            "banelings_alive": len(banes),
            "zerglings_alive": sum(1 for e in enemies if e.kind == "zergling"),
            "squad_spread": round(spread, 1),
            "nearest_baneling_distance": round(nearest_bane, 1) if nearest_bane is not None else None,
            "nearest_baneling_to_squad_center": (
                round(nearest_bane_to_center, 1) if nearest_bane_to_center is not None else None
            ),
            "stimmed_marines": sum(1 for m in marines if m.stimmed),
            "marines_in_splash_danger": marines_in_splash_danger,
            "seconds_left": seconds_left,
        },
        "rules": config.STATE_RULES,
    }


class Blackboard:
    """Memory of the previous decision step, so the next state can report what each
    Marine did last and how much HP it lost since. Pure: fed views and actions."""

    def __init__(self) -> None:
        self.last_actions: dict[int, str] = {}
        self.last_hp: dict[int, float] = {}
        self.last_enemy_count: int | None = None

    def record(self, marines: list[UnitView], enemies: list[UnitView], actions: dict[int, str] | None) -> None:
        self.last_actions = dict(actions or {})
        self.last_hp = {m.id: m.hp for m in marines}
        self.last_enemy_count = len(enemies)


def _near(u: UnitView, others: list[UnitView], radius: float) -> bool:
    return any(distance(u.x, u.y, o.x, o.y) <= radius for o in others)


def _hp_lost(m: UnitView, memory: Blackboard) -> int:
    return max(0, round(memory.last_hp.get(m.id, m.hp) - m.hp))


def closest_to_banelings(marines: list[UnitView], enemies: list[UnitView]) -> int | None:
    """Tag of the living Marine with the smallest distance to any Baneling (ties: lowest tag)."""
    banes = [e for e in enemies if e.kind == "baneling"]
    if not banes or not marines:
        return None
    return min(marines, key=lambda m: (min(distance(m.x, m.y, b.x, b.y) for b in banes), m.id)).id


def banelings_within(m: UnitView, enemies: list[UnitView], radius: float = config.BANELING_NEAR_TEAMMATE) -> int:
    return sum(1 for e in enemies if e.kind == "baneling" and distance(m.x, m.y, e.x, e.y) <= radius)


def priority_candidates(marines: list[UnitView], enemies: list[UnitView]) -> list[dict]:
    """Up to PRIORITY_TARGET_CANDIDATES Banelings nearest the squad center."""
    banes = [e for e in enemies if e.kind == "baneling"]
    if not banes or not marines:
        return []
    cx, cy = centroid(marines)
    banes.sort(key=lambda b: distance(b.x, b.y, cx, cy))
    return [
        {
            "id": b.id,
            "distance_to_squad_center": round(distance(b.x, b.y, cx, cy), 1),
            "nearest_marine_distance": _nearest_distance(b, marines),
        }
        for b in banes[: config.PRIORITY_TARGET_CANDIDATES]
    ]


def build_commander_state(
    marines: list[UnitView], enemies: list[UnitView], fight_loop: int, memory: Blackboard
) -> dict:
    """003b state with the per-Marine blackboard, squad totals and priority candidates.
    The commander's squad_plan and priority_target are added by the Jev client for call 2."""
    base = build_state(marines, enemies, fight_loop)
    banes = [e for e in enemies if e.kind == "baneling"]
    closest = closest_to_banelings(marines, enemies)
    entries = []
    for m in marines:
        others = sorted((o for o in marines if o.id != m.id), key=lambda o: distance(m.x, m.y, o.x, o.y))
        entries.append({
            "id": m.id,
            "hp": round(m.hp),
            "stimmed": m.stimmed,
            "last_action": memory.last_actions.get(m.id),
            "hp_lost_last_step": _hp_lost(m, memory),
            "nearest_baneling_distance": _nearest_distance(m, banes),
            "nearest_marine_distance": _nearest_distance(m, others),
            "nearest_enemy_distance": _nearest_distance(m, enemies),
            "is_closest_to_banelings": m.id == closest,
            "banelings_within_3": banelings_within(m, enemies),
            "teammates": [
                {
                    "id": t.id,
                    "hp": round(t.hp),
                    "last_action": memory.last_actions.get(t.id),
                    "baneling_within_3": _near(t, banes, config.BANELING_NEAR_TEAMMATE),
                    "distance": round(distance(m.x, m.y, t.x, t.y), 1),
                    "hp_lost_last_step": _hp_lost(t, memory),
                }
                for t in others[: config.TEAMMATES_ON_BLACKBOARD]
            ],
        })
    killed = 0 if memory.last_enemy_count is None else max(0, memory.last_enemy_count - len(enemies))
    return {
        **base,
        "marines": entries,
        "squad": {
            "marines_alive": len(marines),
            "stimmed_count": sum(1 for m in marines if m.stimmed),
            "low_hp_count": sum(1 for m in marines if m.hp <= config.LOW_HP),
            "zerg_killed_last_step": killed,
        },
        "priority_candidates": priority_candidates(marines, enemies),
    }
