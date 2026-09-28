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
    marines_in_splash_danger = sum(
        1
        for m in marines
        if any(distance(m.x, m.y, b.x, b.y) <= config.SPLASH_DANGER_RADIUS for b in banes)
    )
    seconds_left = round((config.MAX_FIGHT_LOOPS - fight_loop) / config.LOOPS_PER_SECOND, 1)
    return {
        "game_time_s": round(fight_loop / config.LOOPS_PER_SECOND, 1),
        "marines": [
            {"id": m.id, "x": round(m.x, 1), "y": round(m.y, 1), "hp": round(m.hp), "stimmed": m.stimmed}
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
            "stimmed_marines": sum(1 for m in marines if m.stimmed),
            "marines_in_splash_danger": marines_in_splash_danger,
            "seconds_left": seconds_left,
        },
        "rules": config.STATE_RULES,
    }
