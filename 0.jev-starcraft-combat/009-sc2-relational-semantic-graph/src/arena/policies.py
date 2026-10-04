"""Policies: attack-move and per-Marine random baselines, scripted stutter-all, Jev commander (with or without stutter)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from collections import Counter
from dataclasses import dataclass

from arena import config
from arena.jev import JevCommanderClient, JevSemanticClient, JevPerceptionClient
from arena.semantic_policy import SemanticMLP
from arena.distill import RelationalSemanticMLP


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
    stim_probability: float | None = None
    plan_kept_low_confidence: bool | None = None
    semantic_activations: dict[str, float] | None = None
    local_activations: dict | None = None
    relational_activations: dict | None = None
    perception_node_latency_ms: float | None = None
    perception_edge_latency_ms: float | None = None


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





class JevTeacherCollector:
    """Successful commander controls; frozen Jev perception observes the identical state.

    After code reflexes are applied, observe_executed() stores the perception together with
    the action that actually ran. This produces aligned supervised data without changing
    teacher behavior.
    """

    name = "jev_teacher_collect"
    uses_blackboard = True
    stutter_default = True
    soldier_actions = config.COMMANDER_STUTTER_MARINE_ACTIONS

    def __init__(self, seed: int, dataset_path: str | None = None,
                 commander_client=None, perception_client=None):
        self.seed = seed
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self._commander = commander_client if commander_client is not None else JevCommanderClient()
        self._perception = perception_client if perception_client is not None else JevPerceptionClient()
        self._previous_plan = None
        self._contact_seen = False
        self._pending = None

    def warmup(self, state: dict) -> None:
        warm = {**state, "marines": [WARMUP_MARINE], "priority_candidates": []}
        self._perception.ask(warm, [WARMUP_MARINE["id"]])
        self._commander.ask(warm, [WARMUP_MARINE["id"]], soldier_actions=self.soldier_actions)

    def decide(self, state: dict) -> Decision:
        tags = marine_tags(state)
        nearest = (state.get("summary") or {}).get("nearest_baneling_distance")
        if nearest is not None and nearest <= config.PRE_CONTACT_DISTANCE:
            self._contact_seen = True

        p = self._perception.ask(state, tags)
        a = self._commander.ask(
            state, tags, previous_plan=self._previous_plan,
            after_contact=self._contact_seen, soldier_actions=self.soldier_actions,
        )
        self._previous_plan = a.plan
        self._pending = {
            "episode_seed": self.seed,
            "global_activations": p.global_activations,
            "local_activations": {str(k): v for k, v in p.local_activations.items()},
            "relational_activations": {str(k): v for k, v in p.relational_activations.items()},
            "stim_now": bool(a.stim_now),
            "perception_priority_target": p.priority_target,
            "teacher_priority_target": a.target_tag,
            "teacher_plan": a.plan,
        }
        action, shares = aggregate(a.actions)
        confs = list(a.confidences.values())
        return Decision(
            action=action, probabilities=shares,
            confidence=sum(confs) / len(confs) if confs else None,
            latency_ms=p.latency_ms + a.commander_latency_ms + a.soldier_latency_ms,
            input_tokens=p.input_tokens + a.input_tokens,
            output_tokens=p.output_tokens + a.output_tokens,
            model=a.model,
            marine_actions=a.actions,
            marine_confidences=a.confidences,
            squad_plan=a.plan,
            priority_target=a.target_tag,
            commander_latency_ms=a.commander_latency_ms,
            soldier_latency_ms=a.soldier_latency_ms,
            stim_now=a.stim_now,
            plan_kept_low_confidence=a.plan_kept_low_confidence,
            perception_node_latency_ms=p.node_latency_ms,
            perception_edge_latency_ms=p.edge_latency_ms,
            semantic_activations=p.global_activations,
            local_activations=p.local_activations,
            relational_activations=p.relational_activations,
        )

    def observe_executed(self, executed: dict | None) -> None:
        if self.dataset_path is None or self._pending is None or executed is None:
            return
        row = {
            **self._pending,
            "executed_actions": {str(k): v for k, v in executed.items()},
        }
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        with self.dataset_path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(row, separators=(",", ":")) + "\n")
        self._pending = None


class JevDistilledSemantic:
    """Frozen Jev perception -> behavior-cloned numerical student."""

    name = "jev_distilled_semantic"
    uses_blackboard = True
    stutter_default = True

    def __init__(self, client: JevPerceptionClient, seed: int,
                 weights_path: str | None, temperature: float = 1.0, stim_threshold: float = 0.5):
        if not weights_path:
            raise ValueError("jev_distilled_semantic requires --weights")
        self._client = client
        self._rng = np.random.default_rng(seed)
        self.temperature = float(temperature)
        self.stim_threshold = float(stim_threshold)
        if not 0.0 <= self.stim_threshold <= 1.0:
            raise ValueError("stim_threshold must be between 0 and 1")
        self.net = RelationalSemanticMLP(seed=seed)
        self.net.load(weights_path)

    def warmup(self, state: dict) -> None:
        self._client.ask({**state, "marines": [WARMUP_MARINE], "priority_candidates": []},
                         [WARMUP_MARINE["id"]])

    def decide(self, state: dict) -> Decision:
        tags = marine_tags(state)
        p = self._client.ask(state, tags)
        actions, confidences = {}, {}
        chosen_probs = []
        for tag in tags:
            action, prob, _ = self.net.act(
                p.global_activations,
                p.local_activations[tag],
                p.relational_activations[tag],
                self._rng,
                temperature=self.temperature,
            )
            actions[tag] = action
            confidences[tag] = 1.0
            chosen_probs.append(prob)

        # Stim is a rare global decision, not part of action exploration. Episode 009a
        # showed that Bernoulli sampling from a weakly separated p(stim) around 0.4 caused
        # massive over-stimming. Decode it deterministically with a separate threshold.
        stim_p = self.net.stim_prob(self.net.global_vector(p.global_activations))
        stim_now = stim_p >= self.stim_threshold
        action, shares = aggregate(actions)
        return Decision(
            action=action, probabilities=shares,
            confidence=sum(chosen_probs) / len(chosen_probs) if chosen_probs else None,
            latency_ms=p.latency_ms, input_tokens=p.input_tokens, output_tokens=p.output_tokens,
            model=p.model, marine_actions=actions, marine_confidences=confidences,
            priority_target=p.priority_target, stim_now=stim_now, stim_probability=stim_p,
            semantic_activations=p.global_activations,
            local_activations=p.local_activations,
            relational_activations=p.relational_activations,
            perception_node_latency_ms=p.node_latency_ms,
            perception_edge_latency_ms=p.edge_latency_ms,
        )


class JevTrainableSemantic:
    """Jev perceives; a tiny local MLP chooses actions and can learn from battle reward."""

    name = "jev_trainable_semantic"
    uses_blackboard = True
    stutter_default = True

    def __init__(self, client: JevPerceptionClient, seed: int, training: bool = False, weights_path: str | None = None):
        self._client = client
        self._rng = np.random.default_rng(seed)
        self.training = training
        self.weights_path = Path(weights_path) if weights_path else None
        self.net = SemanticMLP(seed=seed)
        if self.weights_path is not None and self.weights_path.exists():
            self.net.load(self.weights_path)

    def warmup(self, state: dict) -> None:
        self._client.ask({**state, "marines": [WARMUP_MARINE]}, [WARMUP_MARINE["id"]])

    def decide(self, state: dict) -> Decision:
        tags = marine_tags(state)
        p = self._client.ask(state, tags)
        actions, confidences = {}, {}
        self._step_start = len(self.net.trajectory)
        for tag in tags:
            action, prob, _ = self.net.act(
                p.global_activations, p.local_activations[tag], self._rng, training=self.training, key=tag
            )
            actions[tag] = action
            # Do not route learned-policy actions through Jev's confidence fallback.
            confidences[tag] = 1.0
        action, shares = aggregate(actions)
        return Decision(
            action=action, probabilities=shares,
            confidence=sum(confidences.values()) / len(confidences) if confidences else None,
            latency_ms=p.latency_ms, input_tokens=p.input_tokens, output_tokens=p.output_tokens,
            model=p.model, marine_actions=actions, marine_confidences=confidences,
            semantic_activations=p.global_activations,
            local_activations=p.local_activations,
            relational_activations=p.relational_activations,
        )

    def observe_executed(self, executed: dict | None) -> None:
        """Called by the bot after code rules ran: train only on Marine steps that executed the
        network's own choice."""
        if self.training:
            self.net.drop_unexecuted(getattr(self, "_step_start", 0), executed)

    def end_episode(self, reward: float) -> None:
        if not self.training:
            return
        self.last_advantage = self.net.finish_episode(reward)
        if self.weights_path is not None:
            self.weights_path.parent.mkdir(parents=True, exist_ok=True)
            self.net.save(self.weights_path)


class JevSemanticNet:
    """Shared semantic perceptions -> independent Marine actions; no discrete squad plan."""

    name = "jev_semantic_net"
    uses_blackboard = True
    stutter_default = True
    soldier_actions = config.SEMANTIC_MARINE_ACTIONS

    def __init__(self, client: JevSemanticClient):
        self._client = client

    def warmup(self, state: dict) -> None:
        self._client.ask(
            {**state, "marines": [WARMUP_MARINE], "priority_candidates": []},
            [WARMUP_MARINE["id"]],
            soldier_actions=self.soldier_actions,
        )

    def decide(self, state: dict) -> Decision:
        a = self._client.ask(state, marine_tags(state), soldier_actions=self.soldier_actions)
        action, shares = aggregate(a.actions)
        confs = list(a.confidences.values())
        return Decision(
            action=action,
            probabilities=shares,
            confidence=sum(confs) / len(confs) if confs else None,
            latency_ms=a.perception_latency_ms + a.soldier_latency_ms,
            input_tokens=a.input_tokens,
            output_tokens=a.output_tokens,
            model=a.model,
            marine_actions=a.actions,
            marine_confidences=a.confidences,
            commander_latency_ms=a.perception_latency_ms,
            soldier_latency_ms=a.soldier_latency_ms,
            semantic_activations=a.activations,
        )


class JevCommanderStutter(JevCommander):
    """Round 4h plus `stutter` as a soldier option. Low-confidence default: stutter when an enemy
    is within STUTTER_ENEMY_RADIUS and no Baneling within LOW_CONFIDENCE_KITE_DISTANCE."""

    name = "jev_commander_stutter"
    stutter_default = True
    soldier_actions = config.COMMANDER_STUTTER_MARINE_ACTIONS


POLICY_NAMES = ("attack_move", "random", "stutter_all", "jev_commander", "jev_commander_stutter", "jev_semantic_net", "jev_trainable_semantic", "jev_teacher_collect", "jev_distilled_semantic")


def make_policy(name: str, seed: int, jev_client=None, training: bool = False, weights_path: str | None = None, dataset_path: str | None = None, temperature: float = 1.0, stim_threshold: float = 0.5):
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
    if name == "jev_semantic_net":
        return JevSemanticNet(jev_client if jev_client is not None else JevSemanticClient())
    if name == "jev_trainable_semantic":
        return JevTrainableSemantic(jev_client if jev_client is not None else JevPerceptionClient(), seed, training, weights_path)
    if name == "jev_teacher_collect":
        return JevTeacherCollector(seed, dataset_path=dataset_path)
    if name == "jev_distilled_semantic":
        return JevDistilledSemantic(jev_client if jev_client is not None else JevPerceptionClient(), seed, weights_path, temperature, stim_threshold)
    raise ValueError(f"unknown policy: {name}")
