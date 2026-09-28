"""Thin wrappers over the TypeSafe SDK: one squad question, or one question per Marine."""

from __future__ import annotations

import time
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv
from typesafe_sdk import Choice, TypeSafeClient

from arena import config

QUESTION_KEY = "squad_action"


@dataclass(frozen=True)
class JevAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float
    latency_ms: float
    input_tokens: int
    output_tokens: int
    model: str


def _default_client():
    load_dotenv(find_dotenv(usecwd=True))
    return TypeSafeClient()


class JevSquadClient:
    def __init__(self, client=None):
        self._client = client if client is not None else _default_client()
        self._questions = {
            QUESTION_KEY: Choice(instructions=config.SQUAD_INSTRUCTIONS, criteria=config.ACTIONS)
        }

    def ask(self, state: dict) -> JevAnswer:
        start = time.perf_counter()
        response = self._client.system_one(state=state, questions=self._questions)
        latency_ms = (time.perf_counter() - start) * 1000
        answer = response.answers[QUESTION_KEY]
        return JevAnswer(
            choice=answer.choice,
            probabilities=dict(answer.probabilities),
            confidence=float(answer.confidence),
            latency_ms=latency_ms,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
        )


def marine_key(tag: int) -> str:
    return f"marine_{tag}"


@dataclass(frozen=True)
class JevMarineAnswer:
    actions: dict[int, str]
    confidences: dict[int, float]
    latency_ms: float
    input_tokens: int
    output_tokens: int
    model: str


class JevMarineClient:
    """One system_one call per decision, one Choice per living Marine keyed `marine_<tag>`.
    The questions dict is rebuilt every call because Marines die."""

    def __init__(self, client=None):
        self._client = client if client is not None else _default_client()

    def ask(self, state: dict, marine_tags: list[int]) -> JevMarineAnswer:
        questions = {
            marine_key(tag): Choice(
                instructions=config.MARINE_INSTRUCTIONS_TEMPLATE.format(tag=tag), criteria=config.MARINE_ACTIONS
            )
            for tag in marine_tags
        }
        start = time.perf_counter()
        response = self._client.system_one(state=state, questions=questions)
        latency_ms = (time.perf_counter() - start) * 1000
        actions, confidences = {}, {}
        for tag in marine_tags:
            answer = response.answers.get(marine_key(tag))
            if answer is None:
                continue
            actions[tag] = answer.choice
            confidences[tag] = float(answer.confidence)
        return JevMarineAnswer(
            actions=actions,
            confidences=confidences,
            latency_ms=latency_ms,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
        )
