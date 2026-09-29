"""Squad policies: attack-move baseline, random baseline, and Jev."""

from __future__ import annotations

import random
from dataclasses import dataclass

from arena import config
from arena.jev import JevSquadClient


@dataclass(frozen=True)
class Decision:
    action: str
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    model: str | None = None


class AttackMove:
    name = "attack_move"

    def decide(self, state: dict) -> Decision:
        return Decision(action="attack")


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int):
        self._rng = random.Random(seed)

    def decide(self, state: dict) -> Decision:
        return Decision(action=self._rng.choice(list(config.ACTIONS)))


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


POLICY_NAMES = ("attack_move", "random", "jev_squad")


def make_policy(name: str, seed: int, jev_client: JevSquadClient | None = None):
    if name == "attack_move":
        return AttackMove()
    if name == "random":
        return RandomPolicy(seed)
    if name == "jev_squad":
        return JevSquad(jev_client if jev_client is not None else JevSquadClient())
    raise ValueError(f"unknown policy: {name}")
