"""Policies: attack-move baseline, per-Marine random baseline, Jev per Marine, and Jev commander + Marines."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from arena import config
from arena.jev import JevCommanderClient, JevMarineClient


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
    """A random action per Marine each step, seeded: same control granularity as jev_marine."""

    name = "random"

    def __init__(self, seed: int):
        self._rng = random.Random(seed)

    def decide(self, state: dict) -> Decision:
        choices = list(config.MARINE_ACTIONS)
        marine_actions = {tag: self._rng.choice(choices) for tag in marine_tags(state)}
        action, shares = aggregate(marine_actions)
        return Decision(action=action, probabilities=shares, marine_actions=marine_actions)


WARMUP_MARINE = {
    "id": 1, "x": 0.0, "y": 0.0, "hp": 45, "stimmed": False,
    "nearest_baneling_distance": None, "nearest_marine_distance": None, "nearest_enemy_distance": None,
}


class JevMarine:
    name = "jev_marine"

    def __init__(self, client: JevMarineClient):
        self._client = client

    def warmup(self, state: dict) -> None:
        self._client.ask({**state, "marines": [WARMUP_MARINE]}, [WARMUP_MARINE["id"]])

    def decide(self, state: dict) -> Decision:
        a = self._client.ask(state, marine_tags(state))
        action, shares = aggregate(a.actions, config.JEV_MARINE_ACTIONS)
        confs = list(a.confidences.values())
        return Decision(
            action=action,
            probabilities=shares,
            confidence=sum(confs) / len(confs) if confs else None,
            latency_ms=a.latency_ms,
            input_tokens=a.input_tokens,
            output_tokens=a.output_tokens,
            model=a.model,
            marine_actions=a.actions,
            marine_confidences=a.confidences,
        )


class JevCommander:
    """Call 1: the commander picks a squad plan and a priority Baneling.
    Call 2: every Marine picks one of 8 actions, seeing the plan and the blackboard."""

    name = "jev_commander"
    uses_blackboard = True

    def __init__(self, client: JevCommanderClient):
        self._client = client

    def warmup(self, state: dict) -> None:
        self._client.ask({**state, "marines": [WARMUP_MARINE], "priority_candidates": []}, [WARMUP_MARINE["id"]])

    def decide(self, state: dict) -> Decision:
        a = self._client.ask(state, marine_tags(state))
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
        )


POLICY_NAMES = ("attack_move", "random", "jev_marine", "jev_commander")


def make_policy(name: str, seed: int, jev_client=None):
    if name == "attack_move":
        return AttackMove()
    if name == "random":
        return RandomMarine(seed)
    if name == "jev_marine":
        return JevMarine(jev_client if jev_client is not None else JevMarineClient())
    if name == "jev_commander":
        return JevCommander(jev_client if jev_client is not None else JevCommanderClient())
    raise ValueError(f"unknown policy: {name}")
