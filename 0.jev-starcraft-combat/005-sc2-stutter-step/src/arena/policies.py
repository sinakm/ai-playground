"""Policies: attack-move and per-Marine random baselines, scripted stutter-all, Jev commander (with or without stutter)."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from arena import config
from arena.jev import JevCommanderClient


@dataclass(frozen=True)
class Decision:
    action: str
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    model: str | None = None
    marine_actions: dict[int, str] | None = None
    marine_confidences: dict[int, float] | None = None
    squad_plan: str | None = None
    priority_target: int | None = None
    commander_latency_ms: float | None = None
    soldier_latency_ms: float | None = None
    stim_now: bool | None = None
    plan_kept_low_confidence: bool | None = None


def marine_tags(state: dict) -> list[int]:
    return [m["id"] for m in state.get("marines", [])]


def aggregate(
    marine_actions: dict[int, str], actions: dict[str, str] = config.MARINE_ACTIONS
) -> tuple[str, dict[str, float] | None]:
    """Most common Marine action (ties: first in `actions` order) and the share of Marines per action."""
    if not marine_actions:
        return "attack", None
    counts = Counter(marine_actions.values())
    n = len(marine_actions)
    order = list(actions)
    action = max(order, key=lambda a: (counts[a], -order.index(a)))
    return action, {a: counts[a] / n for a in order}


class AttackMove:
    name = "attack_move"

    def decide(self, state: dict) -> Decision:
        return Decision(action="attack")


class RandomMarine:
    """A random action per Marine each step (stutter included), seeded: same control granularity as Jev."""

    name = "random"

    def __init__(self, seed: int):
        self._rng = random.Random(seed)

    def decide(self, state: dict) -> Decision:
        choices = list(config.MARINE_ACTIONS)
        marine_actions = {tag: self._rng.choice(choices) for tag in marine_tags(state)}
        action, shares = aggregate(marine_actions)
        return Decision(action=action, probabilities=shares, marine_actions=marine_actions)


class StutterAll:
    """Scripted baseline: every living Marine executes `stutter` for the whole fight. No Jev."""

    name = "stutter_all"

    def decide(self, state: dict) -> Decision:
        marine_actions = {tag: "stutter" for tag in marine_tags(state)}
        action, shares = aggregate(marine_actions)
        return Decision(action=action, probabilities=shares, marine_actions=marine_actions)


WARMUP_MARINE = {
    "id": 1, "x": 0.0, "y": 0.0, "hp": 45, "stimmed": False,
    "nearest_baneling_distance": None, "nearest_marine_distance": None, "nearest_enemy_distance": None,
}


class JevCommander:
    """Call 1: the commander picks a squad plan and a priority Baneling.
    Call 2: every Marine picks one of the soldier actions, seeing the plan and the blackboard.
    Round 4h behavior: stutter is not offered."""

    name = "jev_commander"
    uses_blackboard = True
    stutter_default = False
    soldier_actions = config.COMMANDER_MARINE_ACTIONS

    def __init__(self, client: JevCommanderClient):
        self._client = client
        self._previous_plan: str | None = None
        self._contact_seen = False

    def warmup(self, state: dict) -> None:
        self._client.ask(
            {**state, "marines": [WARMUP_MARINE], "priority_candidates": []},
            [WARMUP_MARINE["id"]],
            soldier_actions=self.soldier_actions,
        )

    def decide(self, state: dict) -> Decision:
        nearest = (state.get("summary") or {}).get("nearest_baneling_distance")
        if nearest is not None and nearest <= config.PRE_CONTACT_DISTANCE:
            self._contact_seen = True
        a = self._client.ask(
            state,
            marine_tags(state),
            previous_plan=self._previous_plan,
            after_contact=self._contact_seen,
            soldier_actions=self.soldier_actions,
        )
        self._previous_plan = a.plan
        action, shares = aggregate(a.actions)
        confs = list(a.confidences.values())
        return Decision(
            action=action,
            probabilities=shares,
            confidence=sum(confs) / len(confs) if confs else None,
            latency_ms=a.commander_latency_ms + a.soldier_latency_ms,
            input_tokens=a.input_tokens,
            output_tokens=a.output_tokens,
            model=a.model,
            marine_actions=a.actions,
            marine_confidences=a.confidences,
            squad_plan=a.plan,
            priority_target=a.target_tag,
            commander_latency_ms=a.commander_latency_ms,
            soldier_latency_ms=a.soldier_latency_ms,
            stim_now=a.stim_now,
            plan_kept_low_confidence=a.plan_kept_low_confidence,
        )


class JevCommanderStutter(JevCommander):
    """Round 4h plus `stutter` as a soldier option. Low-confidence default: stutter when an enemy
    is within STUTTER_ENEMY_RADIUS and no Baneling within LOW_CONFIDENCE_KITE_DISTANCE."""

    name = "jev_commander_stutter"
    stutter_default = True
    soldier_actions = config.COMMANDER_STUTTER_MARINE_ACTIONS


POLICY_NAMES = ("attack_move", "random", "stutter_all", "jev_commander", "jev_commander_stutter")


def make_policy(name: str, seed: int, jev_client=None):
    if name == "attack_move":
        return AttackMove()
    if name == "random":
        return RandomMarine(seed)
    if name == "stutter_all":
        return StutterAll()
    if name == "jev_commander":
        return JevCommander(jev_client if jev_client is not None else JevCommanderClient())
    if name == "jev_commander_stutter":
        return JevCommanderStutter(jev_client if jev_client is not None else JevCommanderClient())
    raise ValueError(f"unknown policy: {name}")
