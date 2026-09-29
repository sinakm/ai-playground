"""Transient TypeSafe errors: one retry in the clients, no crash in the bot. No SC2, no network."""

import json
from types import SimpleNamespace


import pytest
from typesafe_sdk import TypeSafeAPIConnectionError, TypeSafeInternalServerError, TypeSafeRateLimitError

from arena import jev
from arena.bot import ArenaBot, decide_or_error
from arena.evaluate import summarize
from arena.jev import JevCommanderClient
from arena.policies import Decision


def server_error():
    return TypeSafeInternalServerError(520, "", {}, message="520")


class Flaky:
    """Raises the given errors first, then answers every question with `attack`."""

    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0

    def system_one(self, state, questions):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        answers = {}
        for k, q in questions.items():
            choice = "hold_and_shoot" if k == "squad_plan" else next(iter(q.criteria)) if k == "priority_target" else "attack"
            answers[k] = SimpleNamespace(choice=choice, probabilities={choice: 1.0}, confidence=0.9, noul=0.0)
        return SimpleNamespace(answers=answers, usage=SimpleNamespace(input_tokens=10, output_tokens=1), model="m")


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(jev.time, "sleep", slept.append)
    return slept


def test_client_retries_once_then_succeeds(no_sleep):
    fake = Flaky([server_error()])
    a = JevCommanderClient(client=fake).ask({"priority_candidates": []}, [1, 2])
    assert fake.calls == 3 and a.actions == {1: "attack", 2: "attack"}
    assert no_sleep == [0.2]


def test_commander_client_retries_each_call_once(no_sleep):
    fake = Flaky([TypeSafeAPIConnectionError("reset")])
    a = JevCommanderClient(client=fake).ask({"priority_candidates": []}, [1])
    assert fake.calls == 3 and a.plan == "hold_and_shoot" and a.actions == {1: "attack"}


def test_client_gives_up_after_one_retry():
    rate = TypeSafeRateLimitError(429, "", {}, message="429")
    fake = Flaky([server_error(), rate])
    with pytest.raises(TypeSafeRateLimitError):
        JevCommanderClient(client=fake).ask({"priority_candidates": []}, [1])
    assert fake.calls == 2


def test_client_does_not_retry_other_errors():
    fake = Flaky([ValueError("bad")])
    with pytest.raises(ValueError):
        JevCommanderClient(client=fake).ask({"priority_candidates": []}, [1])
    assert fake.calls == 1


class RaisingPolicy:
    name = "jev_commander"

    def decide(self, state):
        raise server_error()


class OkPolicy:
    name = "ok"

    def decide(self, state):
        return Decision(action="attack")


def test_decide_or_error():
    d, err = decide_or_error(OkPolicy(), {})
    assert d.action == "attack" and err is None
    d, err = decide_or_error(RaisingPolicy(), {})
    assert d is None and err == "TypeSafeInternalServerError"
    with pytest.raises(ZeroDivisionError):
        decide_or_error(SimpleNamespace(decide=lambda s: 1 / 0), {})


def test_bot_logs_api_error_record_and_counts(tmp_path):
    bot = ArenaBot(RaisingPolicy(), tmp_path, seed=0, realtime=True)
    d, err = decide_or_error(bot.policy, {"k": 1})
    bot._record_api_error(err, elapsed=24, state={"k": 1}, marines_alive=12, enemies_alive=24)
    bot.close()
    [rec] = [json.loads(line) for line in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert rec["action"] == "api_error" and rec["error"] == "TypeSafeInternalServerError"
    assert rec["fight_loop"] == 24 and rec["late"] is False and rec["marine_actions"] is None
    assert (rec["input_tokens"], rec["output_tokens"]) == (0, 0)
    assert bot.api_errors == 1 and bot.decisions == 0 and bot.latencies == [] and bot.late == 0


def run(policy, api_errors=None):
    r = {
        "policy": policy, "result": "loss", "marines_alive": 0, "enemies_killed": 3, "fight_seconds": 10.0,
        "latencies_ms": [], "late_decisions": 0, "input_tokens": 0, "output_tokens": 0,
    }
    if api_errors is not None:
        r["api_errors"] = api_errors
    return r


def test_summarize_sums_api_errors():
    m = summarize([run("jev_commander", 2), run("jev_commander", 1), run("random")])
    assert m["jev_commander"]["api_errors"] == 3
    assert m["random"]["api_errors"] == 0


def test_panel_frame_renders_api_error_record():
    from arena.overlay import PANEL_SIZE, cumulative, panel_frame

    rec = {
        "policy": "jev_commander", "action": "api_error", "error": "TypeSafeInternalServerError",
        "probabilities": None, "confidence": None, "latency_ms": 0.0, "input_tokens": 0, "output_tokens": 0,
        "late": False, "marine_actions": None, "marine_confidences": None, "squad_plan": None,
        "priority_target": None, "commander_latency_ms": None, "soldier_latency_ms": None,
        "marines_alive": 12, "enemies_alive": 24,
    }
    assert panel_frame(rec, cumulative([rec])[0]).size == PANEL_SIZE
