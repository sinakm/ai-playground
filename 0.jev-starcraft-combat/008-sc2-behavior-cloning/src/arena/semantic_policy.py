"""Trainable policy for episode 008.

Jev is frozen perception. A small shared MLP maps semantic activations to per-Marine
actions. Stim is a separate global binary head because the successful commander can
stim and issue a movement/fire action in the same decision window.

Episode 008 adds supervised behavior cloning while retaining the episode-007
REINFORCE hooks for optional fine-tuning.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# Stim is deliberately not mutually exclusive with the Marine action.
ACTIONS = ("kite", "split", "attack", "retreat", "focus_bane", "cover_ally", "bait", "stutter")
GLOBAL_KEYS = (
    "baneling_pressure",
    "clumping_danger",
    "encirclement_risk",
    "focus_fire_opportunity",
    "retreat_pressure",
    "formation_instability",
)
LOCAL_KEYS = ("personal_danger", "isolation", "escape_pressure", "firing_opportunity")
INPUT_DIM = len(GLOBAL_KEYS) + len(LOCAL_KEYS)
HIDDEN_DIM = 16


def _sigmoid(x):
    x = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x))


class SemanticMLP:
    def __init__(self, seed=0, learning_rate=0.05, entropy=0.01, baseline_alpha=0.1):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, 0.15, (HIDDEN_DIM, INPUT_DIM))
        self.b1 = np.zeros(HIDDEN_DIM)
        self.w2 = rng.normal(0, 0.15, (len(ACTIONS), HIDDEN_DIM))
        self.b2 = np.zeros(len(ACTIONS))
        self.w_stim = rng.normal(0, 0.15, len(GLOBAL_KEYS))
        self.b_stim = 0.0
        self.lr = learning_rate
        self.entropy = entropy
        self.trajectory = []
        self.baseline_alpha = baseline_alpha
        self.baseline = None
        self.reward_var = 25.0
        self.episodes = 0

    def vector(self, global_a, local_a):
        return np.asarray(
            [global_a.get(k, 0.5) for k in GLOBAL_KEYS]
            + [local_a.get(k, 0.5) for k in LOCAL_KEYS],
            dtype=float,
        )

    def global_vector(self, global_a):
        return np.asarray([global_a.get(k, 0.5) for k in GLOBAL_KEYS], dtype=float)

    def probs(self, x):
        h = np.tanh(self.w1 @ x + self.b1)
        z = self.w2 @ h + self.b2
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
        return p, h

    def stim_probability(self, global_a):
        g = self.global_vector(global_a)
        return float(_sigmoid(g @ self.w_stim + self.b_stim))

    def act(self, global_a, local_a, rng, training=True, key=None, sample_eval=False, temperature=1.0):
        x = self.vector(global_a, local_a)
        p, h = self.probs(x)
        if not training and sample_eval:
            t = max(0.05, float(temperature))
            q = np.power(np.maximum(p, 1e-12), 1.0 / t)
            q /= q.sum()
            idx = int(rng.choice(len(ACTIONS), p=q))
        else:
            idx = int(rng.choice(len(ACTIONS), p=p)) if training else int(np.argmax(p))
        if training:
            self.trajectory.append((x, h, p, idx, key))
        return ACTIONS[idx], float(p[idx]), {a: float(p[i]) for i, a in enumerate(ACTIONS)}

    def drop_unexecuted(self, start, executed):
        if executed is None:
            return
        tail = [
            t for t in self.trajectory[start:]
            if executed.get(t[4]) == ACTIONS[t[3]]
        ]
        del self.trajectory[start:]
        self.trajectory.extend(tail)

    def advantage(self, reward):
        if self.baseline is None:
            return 0.0
        return float(
            np.clip(
                (reward - self.baseline) / max(1.0, math.sqrt(self.reward_var)),
                -3.0,
                3.0,
            )
        )

    def update_baseline(self, reward):
        if self.baseline is None:
            self.baseline = float(reward)
        else:
            dev = reward - self.baseline
            self.baseline += self.baseline_alpha * dev
            self.reward_var = (
                (1 - self.baseline_alpha) * self.reward_var
                + self.baseline_alpha * dev * dev
            )
        self.episodes += 1

    def finish_episode(self, reward):
        advantage = self.advantage(reward)
        self.update_baseline(reward)
        if not self.trajectory or advantage == 0.0:
            self.trajectory.clear()
            return advantage
        gw1 = np.zeros_like(self.w1)
        gb1 = np.zeros_like(self.b1)
        gw2 = np.zeros_like(self.w2)
        gb2 = np.zeros_like(self.b2)
        n = len(self.trajectory)
        for x, h, p, idx, _ in self.trajectory:
            d = -p.copy()
            d[idx] += 1.0
            d *= advantage
            d += self.entropy * (1.0 / len(ACTIONS) - p)
            gw2 += np.outer(d, h)
            gb2 += d
            dh = (self.w2.T @ d) * (1 - h * h)
            gw1 += np.outer(dh, x)
            gb1 += dh
        scale = self.lr / max(1, n)
        self.w1 += scale * gw1
        self.b1 += scale * gb1
        self.w2 += scale * gw2
        self.b2 += scale * gb2
        self.trajectory.clear()
        return advantage

    def fit_supervised(
        self,
        x,
        y,
        stim_x,
        stim_y,
        epochs=150,
        learning_rate=0.03,
        batch_size=512,
        balance_power=0.5,
        seed=0,
    ):
        """Behavior-clone action and global stim heads with mini-batch cross entropy."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=int)
        stim_x = np.asarray(stim_x, dtype=float)
        stim_y = np.asarray(stim_y, dtype=float)
        if len(x) == 0:
            raise ValueError("no action samples")
        rng = np.random.default_rng(seed)
        counts = np.bincount(y, minlength=len(ACTIONS)).astype(float)
        nonzero = counts[counts > 0]
        ref = float(nonzero.mean()) if len(nonzero) else 1.0
        class_w = np.ones(len(ACTIONS))
        for i, c in enumerate(counts):
            if c > 0:
                class_w[i] = (ref / c) ** float(balance_power)

        for _ in range(int(epochs)):
            order = rng.permutation(len(x))
            for start in range(0, len(x), int(batch_size)):
                ids = order[start:start + int(batch_size)]
                xb, yb = x[ids], y[ids]
                h = np.tanh(xb @ self.w1.T + self.b1)
                z = h @ self.w2.T + self.b2
                z -= z.max(axis=1, keepdims=True)
                p = np.exp(z)
                p /= p.sum(axis=1, keepdims=True)
                d = p
                d[np.arange(len(ids)), yb] -= 1.0
                sw = class_w[yb]
                d *= (sw / max(1e-12, sw.sum()))[:, None]
                gw2 = d.T @ h
                gb2 = d.sum(axis=0)
                dh = (d @ self.w2) * (1.0 - h * h)
                gw1 = dh.T @ xb
                gb1 = dh.sum(axis=0)
                self.w2 -= learning_rate * gw2
                self.b2 -= learning_rate * gb2
                self.w1 -= learning_rate * gw1
                self.b1 -= learning_rate * gb1

            if len(stim_x):
                order_s = rng.permutation(len(stim_x))
                for start in range(0, len(stim_x), int(batch_size)):
                    ids = order_s[start:start + int(batch_size)]
                    xb, yb = stim_x[ids], stim_y[ids]
                    p = _sigmoid(xb @ self.w_stim + self.b_stim)
                    d = (p - yb) / max(1, len(ids))
                    self.w_stim -= learning_rate * (xb.T @ d)
                    self.b_stim -= learning_rate * float(d.sum())

        return {
            "action_counts": {ACTIONS[i]: int(counts[i]) for i in range(len(ACTIONS))},
            **self.supervised_metrics(x, y, stim_x, stim_y),
        }

    def supervised_metrics(self, x, y, stim_x=None, stim_y=None):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=int)
        per_class_recall = {}
        predicted_action_share = {}
        if len(x):
            h = np.tanh(x @ self.w1.T + self.b1)
            z = h @ self.w2.T + self.b2
            pred = np.argmax(z, axis=1)
            action_accuracy = float(np.mean(pred == y))
            majority_baseline = float(np.max(np.bincount(y, minlength=len(ACTIONS))) / len(y))
            for i, action in enumerate(ACTIONS):
                mask = y == i
                per_class_recall[action] = float(np.mean(pred[mask] == i)) if np.any(mask) else None
                predicted_action_share[action] = float(np.mean(pred == i))
        else:
            action_accuracy = None
            majority_baseline = None
        recalls = [v for v in per_class_recall.values() if v is not None]
        balanced_accuracy = float(np.mean(recalls)) if recalls else None
        stim_accuracy = None
        stim_positive_rate = None
        stim_predicted_positive_rate = None
        if stim_x is not None and len(stim_x):
            sx = np.asarray(stim_x, dtype=float)
            sy = np.asarray(stim_y, dtype=int)
            sp = _sigmoid(sx @ self.w_stim + self.b_stim) >= 0.5
            stim_accuracy = float(np.mean(sp == sy))
            stim_positive_rate = float(np.mean(sy))
            stim_predicted_positive_rate = float(np.mean(sp))
        return {
            "action_accuracy": action_accuracy,
            "majority_baseline": majority_baseline,
            "balanced_accuracy": balanced_accuracy,
            "per_class_recall": per_class_recall,
            "predicted_action_share": predicted_action_share,
            "stim_accuracy": stim_accuracy,
            "stim_positive_rate": stim_positive_rate,
            "stim_predicted_positive_rate": stim_predicted_positive_rate,
        }

    def save(self, path):
        Path(path).write_text(
            json.dumps(
                {
                    "version": 2,
                    "actions": list(ACTIONS),
                    "w1": self.w1.tolist(),
                    "b1": self.b1.tolist(),
                    "w2": self.w2.tolist(),
                    "b2": self.b2.tolist(),
                    "w_stim": self.w_stim.tolist(),
                    "b_stim": self.b_stim,
                    "baseline": self.baseline,
                    "reward_var": self.reward_var,
                    "episodes": self.episodes,
                }
            ),
            encoding="utf-8",
        )

    def load(self, path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        stored_actions = tuple(d.get("actions", ()))
        if stored_actions and stored_actions != ACTIONS:
            raise ValueError(
                "incompatible policy checkpoint: episode 008 uses a separate stim head; "
                "behavior-clone into a fresh episode-008 weights file instead"
            )
        w2 = np.asarray(d["w2"])
        if w2.shape[0] != len(ACTIONS):
            raise ValueError(
                "episode-007 9-way checkpoint is not directly compatible with episode 008"
            )
        self.w1 = np.asarray(d["w1"])
        self.b1 = np.asarray(d["b1"])
        self.w2 = w2
        self.b2 = np.asarray(d["b2"])
        self.w_stim = np.asarray(d.get("w_stim", np.zeros(len(GLOBAL_KEYS))))
        self.b_stim = float(d.get("b_stim", 0.0))
        self.baseline = d.get("baseline")
        self.reward_var = d.get("reward_var", 25.0)
        self.episodes = d.get("episodes", 0)
