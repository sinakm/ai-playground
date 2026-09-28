"""ArenaBot.close() idempotence, without needing a running SC2 game."""

from __future__ import annotations

from arena.bot import ArenaBot


class FakePolicy:
    name = "fake"


class FakeRecorder:
    def __init__(self):
        self.start_wall = None
        self.stop_calls = 0

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self.stop_calls += 1


def test_close_stops_recorder_and_log_exactly_once(tmp_path):
    recorder = FakeRecorder()
    bot = ArenaBot(FakePolicy(), tmp_path, seed=0, realtime=False, recorder=recorder)

    bot.close()
    bot.close()
    bot.close()

    assert recorder.stop_calls == 1
    assert bot.log._closed is True


def test_close_without_recorder_is_safe_and_idempotent(tmp_path):
    bot = ArenaBot(FakePolicy(), tmp_path, seed=0, realtime=False)

    bot.close()
    bot.close()

    assert bot.log._closed is True
