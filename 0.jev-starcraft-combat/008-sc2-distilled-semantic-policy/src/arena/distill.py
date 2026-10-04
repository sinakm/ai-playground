"""Behavior-cloned policy for episode 008.

Jev stays frozen and supplies semantic perception. The numerical student has:
- a shared 10 -> 16 -> 8 per-Marine action head (stim excluded), and
- a separate 6 -> 1 squad-stim head.

Teacher data are grouped by episode seed so validation never sees decisions from a
trajectory used for training.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

GLOBAL_KEYS = (
    "baneling_pressure", "clumping_danger", "encirclement_risk",
    "focus_fire_opportunity", "retreat_pressure", "formation_instability", "stim_opportunity",
)
LOCAL_KEYS = ("personal_danger", "isolation", "escape_pressure", "firing_opportunity")
ACTIONS = ("kite", "split", "attack", "retreat", "focus_bane", "cover_ally", "bait", "stutter")
INPUT_DIM = len(GLOBAL_KEYS) + len(LOCAL_KEYS)
HIDDEN_DIM = 16


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class DistilledSemanticMLP:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, 0.15, (HIDDEN_DIM, INPUT_DIM))
        self.b1 = np.zeros(HIDDEN_DIM)
        self.w2 = rng.normal(0, 0.15, (len(ACTIONS), HIDDEN_DIM))
        self.b2 = np.zeros(len(ACTIONS))
        self.ws = rng.normal(0, 0.15, len(GLOBAL_KEYS))
        self.bs = 0.0

    @staticmethod
    def vector(global_a, local_a):
        return np.asarray(
            [global_a.get(k, 0.5) for k in GLOBAL_KEYS]
            + [local_a.get(k, 0.5) for k in LOCAL_KEYS],
            dtype=float,
        )

    @staticmethod
    def global_vector(global_a):
        return np.asarray([global_a.get(k, 0.5) for k in GLOBAL_KEYS], dtype=float)

    def action_probs(self, x, temperature=1.0):
        x = np.asarray(x, dtype=float)
        h = np.tanh(x @ self.w1.T + self.b1)
        logits = h @ self.w2.T + self.b2
        t = max(float(temperature), 1e-4)
        return _softmax(logits / t)

    def stim_prob(self, global_x):
        return float(_sigmoid(np.asarray(global_x, dtype=float) @ self.ws + self.bs))

    def act(self, global_a, local_a, rng, temperature=0.7):
        x = self.vector(global_a, local_a)
        p = self.action_probs(x, temperature=temperature)
        if temperature <= 0:
            idx = int(np.argmax(p))
        else:
            idx = int(rng.choice(len(ACTIONS), p=p))
        return ACTIONS[idx], float(p[idx]), {a: float(p[i]) for i, a in enumerate(ACTIONS)}

    def stim(self, global_a, rng, sample=True):
        p = self.stim_prob(self.global_vector(global_a))
        return (bool(rng.random() < p) if sample else p >= 0.5), p

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "format": "episode008-distilled-v1",
            "actions": list(ACTIONS),
            "global_keys": list(GLOBAL_KEYS),
            "local_keys": list(LOCAL_KEYS),
            "w1": self.w1.tolist(), "b1": self.b1.tolist(),
            "w2": self.w2.tolist(), "b2": self.b2.tolist(),
            "ws": self.ws.tolist(), "bs": float(self.bs),
        }), encoding="utf-8")

    def load(self, path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        self.w1 = np.asarray(d["w1"], dtype=float); self.b1 = np.asarray(d["b1"], dtype=float)
        self.w2 = np.asarray(d["w2"], dtype=float); self.b2 = np.asarray(d["b2"], dtype=float)
        self.ws = np.asarray(d["ws"], dtype=float); self.bs = float(d["bs"])


def load_teacher_dataset(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        g = r["global_activations"]
        stim = int(bool(r.get("stim_now")))
        seed = int(r["episode_seed"])
        for tag, action in r.get("executed_actions", {}).items():
            if action not in ACTIONS:
                continue
            local = r["local_activations"].get(str(tag), r["local_activations"].get(int(tag), {}))
            rows.append((DistilledSemanticMLP.vector(g, local), ACTIONS.index(action),
                         DistilledSemanticMLP.global_vector(g), stim, seed))
    if not rows:
        raise ValueError(f"no usable teacher samples in {path}")
    X = np.stack([r[0] for r in rows]); y = np.asarray([r[1] for r in rows], dtype=int)
    G = np.stack([r[2] for r in rows]); s = np.asarray([r[3] for r in rows], dtype=float)
    seeds = np.asarray([r[4] for r in rows], dtype=int)
    return X, y, G, s, seeds


def _split_by_episode(seeds, validation_fraction=0.2):
    unique = np.unique(seeds)
    if len(unique) < 2:
        n = len(seeds)
        cut = max(1, int(n * (1 - validation_fraction)))
        idx = np.arange(n)
        return idx[:cut], idx[cut:]
    n_val = max(1, int(round(len(unique) * validation_fraction)))
    val_seeds = set(unique[-n_val:].tolist())
    val = np.asarray([i for i, s in enumerate(seeds) if int(s) in val_seeds], dtype=int)
    train = np.asarray([i for i, s in enumerate(seeds) if int(s) not in val_seeds], dtype=int)
    return train, val


def _metrics(model, X, y, G, stim):
    p = model.action_probs(X)
    eps = 1e-9
    action_loss = float(-np.log(p[np.arange(len(y)), y] + eps).mean())
    action_acc = float((p.argmax(axis=1) == y).mean())
    sp = _sigmoid(G @ model.ws + model.bs)
    stim_loss = float(-(stim * np.log(sp + eps) + (1 - stim) * np.log(1 - sp + eps)).mean())
    stim_acc = float(((sp >= 0.5) == stim).mean())
    return {"action_loss": action_loss, "action_accuracy": action_acc,
            "stim_loss": stim_loss, "stim_accuracy": stim_acc}


def train_behavior_clone(dataset_path, output_path, epochs=120, batch_size=256, learning_rate=0.03,
                         seed=0, validation_fraction=0.2, weight_decay=1e-4):
    X, y, G, stim, seeds = load_teacher_dataset(dataset_path)
    train_idx, val_idx = _split_by_episode(seeds, validation_fraction)
    if len(val_idx) == 0:
        val_idx = train_idx
    model = DistilledSemanticMLP(seed=seed)
    rng = np.random.default_rng(seed)

    for _ in range(int(epochs)):
        order = rng.permutation(train_idx)
        for start in range(0, len(order), int(batch_size)):
            idx = order[start:start + int(batch_size)]
            xb, yb, gb, sb = X[idx], y[idx], G[idx], stim[idx]
            h = np.tanh(xb @ model.w1.T + model.b1)
            p = _softmax(h @ model.w2.T + model.b2)
            dz = p
            dz[np.arange(len(idx)), yb] -= 1.0
            dz /= len(idx)
            gw2 = dz.T @ h + weight_decay * model.w2
            gb2 = dz.sum(axis=0)
            dh = (dz @ model.w2) * (1 - h * h)
            gw1 = dh.T @ xb + weight_decay * model.w1
            gb1 = dh.sum(axis=0)

            sp = _sigmoid(gb @ model.ws + model.bs)
            ds = (sp - sb) / len(idx)
            gws = gb.T @ ds + weight_decay * model.ws
            gbs = float(ds.sum())

            model.w2 -= learning_rate * gw2; model.b2 -= learning_rate * gb2
            model.w1 -= learning_rate * gw1; model.b1 -= learning_rate * gb1
            model.ws -= learning_rate * gws; model.bs -= learning_rate * gbs

    model.save(output_path)
    counts = {a: int((y == i).sum()) for i, a in enumerate(ACTIONS)}
    return {
        "samples": int(len(y)),
        "episodes": int(len(np.unique(seeds))),
        "train": _metrics(model, X[train_idx], y[train_idx], G[train_idx], stim[train_idx]),
        "validation": _metrics(model, X[val_idx], y[val_idx], G[val_idx], stim[val_idx]),
        "action_counts": counts,
        "stim_positive_rate": float(stim.mean()),
        "output": str(output_path),
    }
