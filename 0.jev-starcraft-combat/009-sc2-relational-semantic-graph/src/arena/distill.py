"""Behavior cloning over Jev node + edge semantics for episode 009.

Jev stays frozen. Each Marine receives:
- 7 global node activations
- 4 local Marine node activations
- 4 relational edge activations

The action student is a shared 15 -> 16 -> 8 MLP.
Stim remains a separate decision-level 7 -> 1 logistic head.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

GLOBAL_KEYS = (
    "baneling_pressure", "clumping_danger", "encirclement_risk",
    "focus_fire_opportunity", "retreat_pressure", "formation_instability",
    "stim_opportunity",
)
LOCAL_KEYS = ("personal_danger", "isolation", "escape_pressure", "firing_opportunity")
RELATIONAL_KEYS = (
    "target_engagement_value",
    "target_threat_to_local_group",
    "ally_needs_cover",
    "cover_effectiveness",
)
ACTIONS = ("kite", "split", "attack", "retreat", "focus_bane", "cover_ally", "bait", "stutter")
INPUT_DIM = len(GLOBAL_KEYS) + len(LOCAL_KEYS) + len(RELATIONAL_KEYS)
HIDDEN_DIM = 16


def _softmax(z):
    z = np.asarray(z, dtype=float)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class RelationalSemanticMLP:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, 0.15, (HIDDEN_DIM, INPUT_DIM))
        self.b1 = np.zeros(HIDDEN_DIM)
        self.w2 = rng.normal(0, 0.15, (len(ACTIONS), HIDDEN_DIM))
        self.b2 = np.zeros(len(ACTIONS))
        self.ws = rng.normal(0, 0.15, len(GLOBAL_KEYS))
        self.bs = 0.0

    @staticmethod
    def vector(global_a, local_a, relational_a):
        return np.asarray(
            [global_a.get(k, 0.5) for k in GLOBAL_KEYS]
            + [local_a.get(k, 0.5) for k in LOCAL_KEYS]
            + [relational_a.get(k, 0.5) for k in RELATIONAL_KEYS],
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

    def act(self, global_a, local_a, relational_a, rng, temperature=1.0):
        x = self.vector(global_a, local_a, relational_a)
        p = self.action_probs(x, temperature=temperature)
        idx = int(np.argmax(p)) if temperature <= 0 else int(rng.choice(len(ACTIONS), p=p))
        return ACTIONS[idx], float(p[idx]), {a: float(p[i]) for i, a in enumerate(ACTIONS)}

    def stim(self, global_a, rng, sample=True):
        p = self.stim_prob(self.global_vector(global_a))
        return (bool(rng.random() < p) if sample else p >= 0.5), p

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "format": "episode009-relational-v1",
            "actions": list(ACTIONS),
            "global_keys": list(GLOBAL_KEYS),
            "local_keys": list(LOCAL_KEYS),
            "relational_keys": list(RELATIONAL_KEYS),
            "w1": self.w1.tolist(), "b1": self.b1.tolist(),
            "w2": self.w2.tolist(), "b2": self.b2.tolist(),
            "ws": self.ws.tolist(), "bs": float(self.bs),
        }), encoding="utf-8")

    def load(self, path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        expected = list(RELATIONAL_KEYS)
        if d.get("relational_keys") != expected:
            raise ValueError(
                f"weights relational_keys={d.get('relational_keys')} do not match {expected}; "
                "episode 009 needs newly trained weights"
            )
        self.w1 = np.asarray(d["w1"], dtype=float)
        self.b1 = np.asarray(d["b1"], dtype=float)
        self.w2 = np.asarray(d["w2"], dtype=float)
        self.b2 = np.asarray(d["b2"], dtype=float)
        self.ws = np.asarray(d["ws"], dtype=float)
        self.bs = float(d["bs"])


def load_teacher_dataset(path):
    action_rows = []
    decision_rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        g = r["global_activations"]
        seed = int(r["episode_seed"])
        decision_rows.append((
            RelationalSemanticMLP.global_vector(g),
            float(bool(r.get("stim_now"))),
            seed,
            r.get("perception_priority_target"),
            r.get("teacher_priority_target"),
        ))
        local_all = r.get("local_activations", {})
        rel_all = r.get("relational_activations", {})
        for tag, action in r.get("executed_actions", {}).items():
            if action not in ACTIONS:
                continue
            local = local_all.get(str(tag), {})
            rel = rel_all.get(str(tag), {})
            action_rows.append((
                RelationalSemanticMLP.vector(g, local, rel),
                ACTIONS.index(action),
                seed,
            ))

    if not action_rows:
        raise ValueError(f"no usable action samples in {path}")
    if not decision_rows:
        raise ValueError(f"no decision rows in {path}")

    X = np.stack([r[0] for r in action_rows])
    y = np.asarray([r[1] for r in action_rows], dtype=int)
    action_seeds = np.asarray([r[2] for r in action_rows], dtype=int)

    G = np.stack([r[0] for r in decision_rows])
    stim = np.asarray([r[1] for r in decision_rows], dtype=float)
    decision_seeds = np.asarray([r[2] for r in decision_rows], dtype=int)
    target_pairs = [(r[3], r[4]) for r in decision_rows]
    return X, y, action_seeds, G, stim, decision_seeds, target_pairs


def _split_seeds(seeds, validation_fraction=0.2):
    unique = np.unique(seeds)
    if len(unique) < 2:
        return set(unique.tolist()), set()
    n_val = max(1, int(round(len(unique) * validation_fraction)))
    return set(unique[:-n_val].tolist()), set(unique[-n_val:].tolist())


def _indices_for_seeds(seeds, selected):
    return np.asarray([i for i, s in enumerate(seeds) if int(s) in selected], dtype=int)


def _action_class_weights(y, mode="none"):
    w = np.ones(len(ACTIONS), dtype=float)
    if mode == "none":
        return w
    counts = np.bincount(y, minlength=len(ACTIONS)).astype(float)
    present = counts > 0
    if mode == "sqrt":
        w[present] = np.sqrt(counts[present].sum() / counts[present])
    elif mode == "inverse":
        w[present] = counts[present].sum() / counts[present]
    else:
        raise ValueError(f"unknown class_balance: {mode}")
    w /= w[present].mean() if present.any() else 1.0
    return w


def _action_metrics(model, X, y):
    p = model.action_probs(X)
    eps = 1e-9
    pred = p.argmax(axis=1)
    result = {
        "loss": float(-np.log(p[np.arange(len(y)), y] + eps).mean()),
        "accuracy": float((pred == y).mean()),
        "per_class": {},
    }
    for i, action in enumerate(ACTIONS):
        mask = y == i
        n = int(mask.sum())
        result["per_class"][action] = {
            "n": n,
            "recall": float((pred[mask] == i).mean()) if n else None,
            "predicted_share": float((pred == i).mean()),
            "mean_p_true": float(p[mask, i].mean()) if n else None,
        }
    return result


def _stim_metrics(model, G, stim):
    p = _sigmoid(G @ model.ws + model.bs)
    eps = 1e-9
    return {
        "loss": float(-(stim * np.log(p + eps) + (1 - stim) * np.log(1 - p + eps)).mean()),
        "accuracy": float(((p >= 0.5) == stim).mean()),
        "positive_rate": float(stim.mean()),
        "mean_p_positive": float(p[stim == 1].mean()) if np.any(stim == 1) else None,
        "mean_p_negative": float(p[stim == 0].mean()) if np.any(stim == 0) else None,
    }


def train_behavior_clone(
    dataset_path,
    output_path,
    epochs=120,
    batch_size=256,
    learning_rate=0.03,
    seed=0,
    validation_fraction=0.2,
    weight_decay=1e-4,
    class_balance="none",
    balance_stim=True,
):
    X, y, action_seeds, G, stim, decision_seeds, target_pairs = load_teacher_dataset(dataset_path)
    train_seeds, val_seeds = _split_seeds(action_seeds, validation_fraction)
    train_idx = _indices_for_seeds(action_seeds, train_seeds)
    val_idx = _indices_for_seeds(action_seeds, val_seeds)
    train_dec = _indices_for_seeds(decision_seeds, train_seeds)
    val_dec = _indices_for_seeds(decision_seeds, val_seeds)
    if len(val_idx) == 0:
        val_idx = train_idx
    if len(val_dec) == 0:
        val_dec = train_dec

    model = RelationalSemanticMLP(seed=seed)
    rng = np.random.default_rng(seed)
    class_weights = _action_class_weights(y[train_idx], mode=class_balance)

    positive = float(stim[train_dec].sum())
    negative = float(len(train_dec) - positive)
    stim_pos_weight = (negative / positive) if balance_stim and positive > 0 else 1.0

    for _ in range(int(epochs)):
        order = rng.permutation(train_idx)
        for start in range(0, len(order), int(batch_size)):
            idx = order[start:start + int(batch_size)]
            xb, yb = X[idx], y[idx]
            h = np.tanh(xb @ model.w1.T + model.b1)
            p = _softmax(h @ model.w2.T + model.b2)
            dz = p.copy()
            dz[np.arange(len(idx)), yb] -= 1.0
            sample_w = class_weights[yb]
            dz *= sample_w[:, None]
            dz /= max(float(sample_w.sum()), 1.0)

            gw2 = dz.T @ h + weight_decay * model.w2
            gb2 = dz.sum(axis=0)
            dh = (dz @ model.w2) * (1 - h * h)
            gw1 = dh.T @ xb + weight_decay * model.w1
            gb1 = dh.sum(axis=0)

            model.w2 -= learning_rate * gw2
            model.b2 -= learning_rate * gb2
            model.w1 -= learning_rate * gw1
            model.b1 -= learning_rate * gb1

        # Stim is a decision-level label, trained once per battlefield decision rather
        # than once per living Marine. Positive weighting avoids the 007/008 "never stim"
        # majority-class solution.
        dec_order = rng.permutation(train_dec)
        for start in range(0, len(dec_order), int(batch_size)):
            idx = dec_order[start:start + int(batch_size)]
            gb, sb = G[idx], stim[idx]
            sp = _sigmoid(gb @ model.ws + model.bs)
            weights = np.where(sb > 0.5, stim_pos_weight, 1.0)
            ds = (sp - sb) * weights
            ds /= max(float(weights.sum()), 1.0)
            gws = gb.T @ ds + weight_decay * model.ws
            gbs = float(ds.sum())
            model.ws -= learning_rate * gws
            model.bs -= learning_rate * gbs

    model.save(output_path)

    counts = {a: int((y == i).sum()) for i, a in enumerate(ACTIONS)}
    relation_offset = len(GLOBAL_KEYS) + len(LOCAL_KEYS)
    relational_means_by_action = {}
    for i, action in enumerate(ACTIONS):
        mask = y == i
        relational_means_by_action[action] = {
            key: (float(X[mask, relation_offset + j].mean()) if np.any(mask) else None)
            for j, key in enumerate(RELATIONAL_KEYS)
        }

    valid_target_pairs = [(a, b) for a, b in target_pairs if b is not None]
    target_matches = sum(a == b for a, b in valid_target_pairs)

    return {
        "samples": int(len(y)),
        "decision_rows": int(len(G)),
        "episodes": int(len(np.unique(action_seeds))),
        "input_dim": INPUT_DIM,
        "class_balance": class_balance,
        "stim_positive_weight": float(stim_pos_weight),
        "train": {
            "action": _action_metrics(model, X[train_idx], y[train_idx]),
            "stim": _stim_metrics(model, G[train_dec], stim[train_dec]),
        },
        "validation": {
            "action": _action_metrics(model, X[val_idx], y[val_idx]),
            "stim": _stim_metrics(model, G[val_dec], stim[val_dec]),
        },
        "action_counts": counts,
        "relational_means_by_action": relational_means_by_action,
        "priority_target_agreement": (
            float(target_matches / len(valid_target_pairs)) if valid_target_pairs else None
        ),
        "output": str(output_path),
    }
