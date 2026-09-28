"""Policies: attack-move baseline, per-Marine random baseline, Jev squad, and Jev per Marine."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from arena import config
from arena.jev import JevMarineClient, JevSquadClient


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


def marine_tags(state: dict) -> list[int]:
    return [m["id"] for m in state.get("marines", [])]


def aggregate(marine_actions: dict[int, str]) -> tuple[str, dict[str, float] | None]:
    """Most common Marine action (ties: first in MARINE_ACTIONS order) and the share of Marines per action."""
    if not marine_actions:
        return "attack", None
    counts = Counter(marine_actions.values())
    n = len(marine_actions)
    action = max(config.MARINE_ACTIONS, key=lambda a: (counts[a], -list(config.MARINE_ACTIONS).index(a)))
    return action, {a: counts[a] / n for a in config.MARINE_ACTIONS}


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


class JevSquad:
    name = "jev_squad"

    def __init__(self, client: JevSquadClient):
        self._client = client

    def warmup(self, state: dict) -> None:
        self._client.ask(state)

    def decide(self, state: dict) -> Decision:
        a = self._client.ask(state)
        return Decision(
            action=a.choice,
            probabilities=a.probabilities,
            confidence=a.confidence,
            latency_ms=a.latency_ms,
            input_tokens=a.input_tokens,
            output_tokens=a.output_tokens,
            model=a.model,
        )


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
        action, shares = aggregate(a.actions)
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


POLICY_NAMES = ("attack_move", "random", "jev_squad", "jev_marine")


def make_policy(name: str, seed: int, jev_client=None):
    if name == "attack_move":
        return AttackMove()
    if name == "random":
        return RandomMarine(seed)
    if name == "jev_squad":
        return JevSquad(jev_client if jev_client is not None else JevSquadClient())
    if name == "jev_marine":
        return JevMarine(jev_client if jev_client is not None else JevMarineClient())
    raise ValueError(f"unknown policy: {name}")
