import json

import pytest

from arena.log import read_jsonl
from arena.run import play_one


@pytest.mark.sc2
def test_attack_move_fight_writes_log_and_summary(tmp_path):
    run_dir = play_one("attack_move", seed=0, runs_dir=tmp_path, realtime=False)
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["result"] in {"win", "loss", "timeout"}
    assert summary["decisions"] > 0
    records = read_jsonl(run_dir / "decisions.jsonl")
    assert records[0]["action"] == "attack"
    assert records[0]["marines_alive"] == 12
