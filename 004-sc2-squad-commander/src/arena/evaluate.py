"""Aggregate run summaries into the results table, JSON, and chart."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from arena import config  # noqa: E402
from arena.policies import POLICY_NAMES  # noqa: E402


def cost_usd(input_tokens: int, output_tokens: int) -> float:
    cost = (input_tokens * config.PRICE_INPUT_PER_MTOK + output_tokens * config.PRICE_OUTPUT_PER_MTOK) / 1_000_000
    return round(cost, 6)


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def score(run: dict) -> int:
    return run["enemies_killed"] if run["marines_alive"] >= 1 else 0


def summarize(runs: list[dict]) -> dict[str, dict]:
    metrics = {}
    for policy in POLICY_NAMES:
        rs = [r for r in runs if r["policy"] == policy]
        if not rs:
            continue
        latencies = [x for r in rs for x in r["latencies_ms"]]
        tin = sum(r["input_tokens"] for r in rs)
        tout = sum(r["output_tokens"] for r in rs)
        wins = sum(1 for r in rs if r["result"] == "win")
        metrics[policy] = {
            "runs": len(rs),
            "wins": wins,
            "win_rate": round(wins / len(rs), 3),
            "mean_marines_alive": round(statistics.mean(r["marines_alive"] for r in rs), 2),
            "mean_enemies_killed": round(statistics.mean(r["enemies_killed"] for r in rs), 2),
            "mean_fight_seconds": round(statistics.mean(r["fight_seconds"] for r in rs), 2),
            "mean_score": round(statistics.mean(score(r) for r in rs), 2),
            "median_latency_ms": percentile(latencies, 50),
            "p90_latency_ms": percentile(latencies, 90),
            "late_decisions": sum(r["late_decisions"] for r in rs),
            "api_errors": sum(r.get("api_errors", 0) for r in rs),
            "input_tokens": tin,
            "output_tokens": tout,
            "cost_usd": cost_usd(tin, tout),
        }
    return metrics


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def format_table(metrics: dict) -> str:
    cols = [
        ("Policy", None), ("Runs", "runs"), ("Wins", "wins"), ("Score", "mean_score"), ("Win rate", "win_rate"),
        ("Marines alive", "mean_marines_alive"), ("Enemies killed", "mean_enemies_killed"),
        ("Fight s", "mean_fight_seconds"), ("Median ms", "median_latency_ms"),
        ("p90 ms", "p90_latency_ms"), ("Cost USD", "cost_usd"),
    ]
    lines = ["| " + " | ".join(c for c, _ in cols) + " |", "|" + "---|" * len(cols)]
    for policy, m in metrics.items():
        cells = [policy] + [_fmt(m[k]) for _, k in cols[1:]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _chart(metrics: dict, path: Path) -> None:
    names = list(metrics)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    axes[0].bar(names, [metrics[n]["win_rate"] for n in names], color="#4C72B0")
    axes[0].set_title("Win rate")
    axes[0].set_ylim(0, 1)
    axes[1].bar(names, [metrics[n]["mean_marines_alive"] for n in names], color="#55A868")
    axes[1].set_title("Marines alive (mean)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def evaluate(runs_dir: Path, results_dir: Path) -> dict:
    runs = []
    for p in sorted(runs_dir.glob("*/summary.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if not r["realtime"] and r["result"] != "aborted":
            runs.append(r)
    metrics = summarize(runs)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "table.md").write_text(format_table(metrics), encoding="utf-8")
    (results_dir / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _chart(metrics, results_dir / "chart.png")
    return metrics
