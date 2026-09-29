import json

from arena.evaluate import cost_usd, evaluate, format_table, percentile, summarize
from arena.log import DecisionLog, read_jsonl


def run(policy, result, marines, killed, secs, lat=(), tin=0, tout=0, realtime=False):
    return {
        "policy": policy, "seed": 0, "realtime": realtime, "result": result,
        "marines_alive": marines, "enemies_alive": 16 - killed, "enemies_killed": killed,
        "fight_seconds": secs, "decisions": 10, "late_decisions": 0,
        "latencies_ms": list(lat), "input_tokens": tin, "output_tokens": tout, "record_start_wall": None,
    }


def test_log_roundtrip(tmp_path):
    log = DecisionLog(tmp_path / "d.jsonl")
    log.write({"a": 1})
    log.write({"a": 2})
    log.close()
    assert read_jsonl(tmp_path / "d.jsonl") == [{"a": 1}, {"a": 2}]


def test_cost_and_percentile():
    assert cost_usd(1_000_000, 500) == 0.042
    assert percentile([], 90) is None
    assert percentile([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 90) == 90
    assert percentile([10, 20, 30], 50) == 20


def test_summarize():
    runs = [
        run("jev_squad", "win", 6, 16, 20.0, lat=[100, 200], tin=1_000_000),
        run("jev_squad", "loss", 0, 10, 30.0, lat=[300], tin=1_000_000),
        run("attack_move", "loss", 0, 11, 8.0),
    ]
    m = summarize(runs)
    assert list(m) == ["attack_move", "jev_squad"]
    j = m["jev_squad"]
    assert (j["runs"], j["wins"], j["win_rate"]) == (2, 1, 0.5)
    assert (j["mean_marines_alive"], j["mean_enemies_killed"], j["mean_fight_seconds"]) == (3.0, 13.0, 25.0)
    assert (j["median_latency_ms"], j["p90_latency_ms"]) == (200, 300)
    assert j["cost_usd"] == 0.084
    assert m["attack_move"]["median_latency_ms"] is None


def test_format_table_has_header_and_rows():
    t = format_table(summarize([run("random", "win", 3, 16, 12.0)]))
    assert t.splitlines()[0].startswith("| Policy |")
    assert "| random |" in t


def test_evaluate_skips_realtime_and_writes_files(tmp_path):
    runs_dir, results = tmp_path / "runs", tmp_path / "results"
    for i, r in enumerate([run("random", "win", 3, 16, 12.0), run("jev_squad", "win", 5, 16, 9.0, realtime=True)]):
        d = runs_dir / f"r{i}"
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps(r))
    m = evaluate(runs_dir, results)
    assert list(m) == ["random"]
    assert (results / "table.md").exists()
    assert (results / "chart.png").stat().st_size > 0
    assert json.loads((results / "summary.json").read_text())["random"]["runs"] == 1


def test_evaluate_skips_aborted_runs(tmp_path):
    runs_dir, results = tmp_path / "runs", tmp_path / "results"
    runs = [run("random", "win", 3, 16, 12.0), run("random", "aborted", 12, 0, 0.0)]
    for i, r in enumerate(runs):
        d = runs_dir / f"r{i}"
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps(r))
    m = evaluate(runs_dir, results)
    assert m["random"]["runs"] == 1
