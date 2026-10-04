"""Thin wrapper over the TypeSafe SDK: a commander call, then one question per Marine."""

from __future__ import annotations

import time
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv
from typesafe_sdk import (
    Choice,
    Noul,
    TypeSafeAPIConnectionError,
    TypeSafeAPITimeoutError,
    TypeSafeClient,
    TypeSafeInternalServerError,
    TypeSafeRateLimitError,
)

from arena import config

PLAN_KEY = "squad_plan"
TARGET_KEY = "priority_target"
STIM_KEY = "stim_now"


# Errors worth one retry: the server or network hiccuped, the request itself was fine.
TRANSIENT_ERRORS = (
    TypeSafeInternalServerError,
    TypeSafeAPIConnectionError,
    TypeSafeAPITimeoutError,
    TypeSafeRateLimitError,
)
RETRY_SLEEP_S = 0.2


def _default_client():
    load_dotenv(find_dotenv(usecwd=True))
    return TypeSafeClient()


def _system_one(client, state: dict, questions: dict):
    """system_one with one retry on a transient error; a second failure propagates."""
    try:
        return client.system_one(state=state, questions=questions)
    except TRANSIENT_ERRORS:
        time.sleep(RETRY_SLEEP_S)
        return client.system_one(state=state, questions=questions)


def marine_key(tag: int) -> str:
    return f"marine_{tag}"


def bane_key(tag: int) -> str:
    return f"bane_{tag}"


def priority_target_options(candidates: list[dict]) -> dict[str, str]:
    return {
        bane_key(c["id"]): (
            f"baneling {c['id']}: {c['distance_to_squad_center']} cells from squad center, "
            f"nearest marine {c['nearest_marine_distance']} cells"
        )
        for c in candidates
    }


@dataclass(frozen=True)
class JevCommanderAnswer:
    plan: str
    plan_confidence: float
    plan_raw: str
    plan_kept_low_confidence: bool
    stim_now: bool
    stim_now_p: float
    target_tag: int | None
    actions: dict[int, str]
    confidences: dict[int, float]
    commander_latency_ms: float
    soldier_latency_ms: float
    input_tokens: int
    output_tokens: int
    model: str


class JevCommanderClient:
    """Two sequential system_one calls per decision.
    Call 1 (commander): `squad_plan`, plus `priority_target` over `bane_<tag>` options built
    from state["priority_candidates"] (omitted when no Baneling is alive).
    Call 2 (soldiers): one `marine_<tag>` question per living Marine, with the plan and
    target added to the state."""

    def __init__(self, client=None):
        self._client = client if client is not None else _default_client()

    def ask(
        self,
        state: dict,
        marine_tags: list[int],
        previous_plan: str | None = None,
        after_contact: bool = False,
        soldier_actions: dict[str, str] = config.COMMANDER_MARINE_ACTIONS,
    ) -> JevCommanderAnswer:
        """`soldier_actions`: the criteria each Marine chooses from (stutter is offered only by
        jev_commander_stutter). `previous_plan`: when the new squad_plan answer has confidence below
        COMMANDER_MIN_CONFIDENCE and a previous plan exists, the previous plan is kept (and is
        what the soldiers see), except that a pre_split plan is never kept `after_contact`."""
        questions = {
            PLAN_KEY: Choice(instructions=config.COMMANDER_PLAN_INSTRUCTIONS, criteria=config.SQUAD_PLANS),
            STIM_KEY: Noul(instructions=config.STIM_NOW_INSTRUCTIONS),
        }
        options = priority_target_options(state.get("priority_candidates") or [])
        if options:
            questions[TARGET_KEY] = Choice(instructions=config.COMMANDER_TARGET_INSTRUCTIONS, criteria=options)
        start = time.perf_counter()
        r1 = _system_one(self._client, state, questions)
        commander_latency_ms = (time.perf_counter() - start) * 1000
        plan_answer = r1.answers[PLAN_KEY]
        stim_answer = r1.answers.get(STIM_KEY)
        stim_now_p = float(stim_answer.noul) if stim_answer is not None else 0.0
        target_answer = r1.answers.get(TARGET_KEY) if options else None
        target_key = target_answer.choice if target_answer is not None and target_answer.choice in options else None
        target_tag = int(target_key.removeprefix("bane_")) if target_key is not None else None

        plan_confidence = float(plan_answer.confidence)
        kept = (
            previous_plan is not None
            and plan_confidence < config.COMMANDER_MIN_CONFIDENCE
            and not (after_contact and previous_plan == "pre_split")
        )
        plan = previous_plan if kept else plan_answer.choice
        soldier_state = {**state, PLAN_KEY: plan, TARGET_KEY: target_key}
        soldier_questions = {
            marine_key(tag): Choice(
                instructions=config.COMMANDER_MARINE_INSTRUCTIONS_TEMPLATE.format(tag=tag),
                criteria=soldier_actions,
            )
            for tag in marine_tags
        }
        start = time.perf_counter()
        r2 = _system_one(self._client, soldier_state, soldier_questions)
        soldier_latency_ms = (time.perf_counter() - start) * 1000
        actions, confidences = {}, {}
        for tag in marine_tags:
            answer = r2.answers.get(marine_key(tag))
            if answer is None:
                continue
            actions[tag] = answer.choice
            confidences[tag] = float(answer.confidence)
        return JevCommanderAnswer(
            plan=plan,
            plan_confidence=plan_confidence,
            plan_raw=plan_answer.choice,
            plan_kept_low_confidence=kept,
            stim_now=stim_now_p >= 0.5,
            stim_now_p=stim_now_p,
            target_tag=target_tag,
            actions=actions,
            confidences=confidences,
            commander_latency_ms=commander_latency_ms,
            soldier_latency_ms=soldier_latency_ms,
            input_tokens=r1.usage.input_tokens + r2.usage.input_tokens,
            output_tokens=r1.usage.output_tokens + r2.usage.output_tokens,
            model=r2.model,
        )


@dataclass(frozen=True)
class JevSemanticAnswer:
    activations: dict[str, float]
    actions: dict[int, str]
    confidences: dict[int, float]
    perception_latency_ms: float
    soldier_latency_ms: float
    input_tokens: int
    output_tokens: int
    model: str


class JevSemanticClient:
    """Layer 1 emits semantic probabilities; layer 2 maps them plus local state to actions."""

    def __init__(self, client=None):
        self._client = client if client is not None else _default_client()

    def ask(
        self,
        state: dict,
        marine_tags: list[int],
        soldier_actions: dict[str, str] = config.SEMANTIC_MARINE_ACTIONS,
    ) -> JevSemanticAnswer:
        perception_questions = {
            key: Noul(instructions=text) for key, text in config.SEMANTIC_PERCEPTIONS.items()
        }
        start = time.perf_counter()
        r1 = _system_one(self._client, state, perception_questions)
        perception_latency_ms = (time.perf_counter() - start) * 1000
        activations = {
            key: float(r1.answers[key].noul)
            for key in perception_questions
            if r1.answers.get(key) is not None
        }

        soldier_state = {**state, "semantic_activations": activations}
        soldier_questions = {
            marine_key(tag): Choice(
                instructions=config.SEMANTIC_MARINE_INSTRUCTIONS_TEMPLATE.format(tag=tag),
                criteria=soldier_actions,
            )
            for tag in marine_tags
        }
        start = time.perf_counter()
        r2 = _system_one(self._client, soldier_state, soldier_questions)
        soldier_latency_ms = (time.perf_counter() - start) * 1000

        actions, confidences = {}, {}
        for tag in marine_tags:
            answer = r2.answers.get(marine_key(tag))
            if answer is not None:
                actions[tag] = answer.choice
                confidences[tag] = float(answer.confidence)

        return JevSemanticAnswer(
            activations=activations,
            actions=actions,
            confidences=confidences,
            perception_latency_ms=perception_latency_ms,
            soldier_latency_ms=soldier_latency_ms,
            input_tokens=r1.usage.input_tokens + r2.usage.input_tokens,
            output_tokens=r1.usage.output_tokens + r2.usage.output_tokens,
            model=r2.model,
        )


@dataclass(frozen=True)
class JevPerceptionAnswer:
    global_activations: dict[str, float]
    local_activations: dict[int, dict[str, float]]
    relational_activations: dict[int, dict[str, float]]
    priority_target: int | None
    node_latency_ms: float
    edge_latency_ms: float
    latency_ms: float
    input_tokens: int
    output_tokens: int
    model: str


PERCEPTION_GLOBALS = {
    **config.SEMANTIC_PERCEPTIONS,
    "stim_opportunity": (
        "Would using Stim now materially improve squad combat effectiveness, considering enemy proximity, "
        "Marine health, and whether healthy Marines are already stimmed?"
    ),
}


LOCAL_PERCEPTIONS = {
    "personal_danger": "Is Marine {tag} personally in immediate danger from nearby enemies, especially Banelings?",
    "isolation": "Is Marine {tag} meaningfully isolated from useful support by the rest of the squad?",
    "escape_pressure": "Does Marine {tag} have poor escape space or an urgent need to create distance?",
    "firing_opportunity": "Does Marine {tag} currently have a strong opportunity to deal useful damage without taking disproportionate risk?",
}

RELATIONAL_PERCEPTIONS = {
    "target_engagement_value": (
        "Would Marine {tag} materially help the squad by attacking semantic_priority_target "
        "during this decision window, considering range, target threat, other Marines and immediate danger?"
    ),
    "target_threat_to_local_group": (
        "Is semantic_priority_target an immediate threat to Marine {tag} or to one of Marine {tag}'s "
        "nearby listed teammates?"
    ),
    "ally_needs_cover": (
        "Does one of Marine {tag}'s nearby listed teammates currently need covering fire or protection "
        "because that teammate is weak, losing health, or under enemy pressure?"
    ),
    "cover_effectiveness": (
        "Could Marine {tag} effectively cover the nearby teammate who most needs help right now, "
        "without creating disproportionate danger for itself?"
    ),
}


class JevPerceptionClient:
    """Relational semantic perception.

    Call 1 ("nodes") emits seven global activations, four local activations per Marine,
    and one referential priority-Baneling choice. Call 2 ("edges") receives that exact
    selected target in the state and emits four Marine-level relational activations.
    """

    def __init__(self, client=None):
        self._client = client if client is not None else _default_client()

    def ask(self, state: dict, marine_tags: list[int]) -> JevPerceptionAnswer:
        node_questions = {
            key: Noul(instructions=text) for key, text in PERCEPTION_GLOBALS.items()
        }
        target_options = priority_target_options(state.get("priority_candidates") or [])
        if target_options:
            node_questions[TARGET_KEY] = Choice(
                instructions=(
                    "Which Baneling currently poses the strongest immediate threat to the squad? "
                    "Identify the threat itself; this is perception, not a squad command."
                ),
                criteria=target_options,
            )
        for tag in marine_tags:
            for key, text in LOCAL_PERCEPTIONS.items():
                node_questions[f"local_{tag}_{key}"] = Noul(instructions=text.format(tag=tag))

        start = time.perf_counter()
        r1 = _system_one(self._client, state, node_questions)
        node_latency_ms = (time.perf_counter() - start) * 1000

        global_a = {k: float(r1.answers[k].noul) for k in PERCEPTION_GLOBALS}
        local_a = {
            tag: {
                key: float(r1.answers[f"local_{tag}_{key}"].noul)
                for key in LOCAL_PERCEPTIONS
            }
            for tag in marine_tags
        }

        target_answer = r1.answers.get(TARGET_KEY) if target_options else None
        target_key = (
            target_answer.choice
            if target_answer is not None and target_answer.choice in target_options
            else None
        )
        priority_target = int(target_key.removeprefix("bane_")) if target_key is not None else None
        target_enemy = next(
            (e for e in state.get("enemies", []) if e.get("id") == priority_target),
            None,
        )

        edge_state = {
            **state,
            "semantic_priority_target": {
                "key": target_key,
                "id": priority_target,
                "description": target_options.get(target_key) if target_key else None,
                "enemy": target_enemy,
            },
            "semantic_global_activations": global_a,
            "semantic_local_activations": {str(k): v for k, v in local_a.items()},
        }
        edge_questions = {}
        for tag in marine_tags:
            for key, text in RELATIONAL_PERCEPTIONS.items():
                if priority_target is None and key.startswith("target_"):
                    continue
                edge_questions[f"edge_{tag}_{key}"] = Noul(instructions=text.format(tag=tag))

        relational_a = {
            tag: {key: 0.0 for key in RELATIONAL_PERCEPTIONS}
            for tag in marine_tags
        }
        edge_latency_ms = 0.0
        r2 = None
        if edge_questions:
            start = time.perf_counter()
            r2 = _system_one(self._client, edge_state, edge_questions)
            edge_latency_ms = (time.perf_counter() - start) * 1000
            for tag in marine_tags:
                for key in RELATIONAL_PERCEPTIONS:
                    answer = r2.answers.get(f"edge_{tag}_{key}")
                    if answer is not None:
                        relational_a[tag][key] = float(answer.noul)

        return JevPerceptionAnswer(
            global_activations=global_a,
            local_activations=local_a,
            relational_activations=relational_a,
            priority_target=priority_target,
            node_latency_ms=node_latency_ms,
            edge_latency_ms=edge_latency_ms,
            latency_ms=node_latency_ms + edge_latency_ms,
            input_tokens=r1.usage.input_tokens + (r2.usage.input_tokens if r2 is not None else 0),
            output_tokens=r1.usage.output_tokens + (r2.usage.output_tokens if r2 is not None else 0),
            model=(r2.model if r2 is not None else r1.model),
        )

