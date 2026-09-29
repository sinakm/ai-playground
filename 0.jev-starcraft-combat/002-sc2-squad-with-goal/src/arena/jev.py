"""Thin wrapper over the TypeSafe SDK for the squad-action question."""

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


class JevSquadClient:
    def __init__(self, client=None):
        if client is None:
            load_dotenv(find_dotenv(usecwd=True))
            client = TypeSafeClient()
        self._client = client
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
