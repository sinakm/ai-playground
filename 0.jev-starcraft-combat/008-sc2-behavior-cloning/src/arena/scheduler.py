"""Background decisions for realtime mode: at most one policy call in flight.

The bot hands `maybe_start` a snapshot taken at a decision step; the callable runs on a
single worker thread. Every bot step calls `poll`; when the call has finished it returns the
snapshot and the result once, so the bot can apply it against the units it sees now.
No SC2 here, so the scheduling can be tested on its own.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Completed:
    snapshot: Any
    result: Any


class DecisionScheduler:
    def __init__(self, fn: Callable[[Any], Any]):
        self._fn = fn
        self._executor: ThreadPoolExecutor | None = None
        self._future: Future | None = None
        self._snapshot: Any = None
        self.started = 0
        self.skipped = 0
        self._closed = False

    @property
    def pending(self) -> bool:
        return self._future is not None

    def maybe_start(self, snapshot: Any) -> bool:
        """Start `fn(snapshot)` in the background. If a call is still in flight (or the
        scheduler is closed), start nothing, count a skipped decision and return False."""
        if self._closed or self._future is not None:
            self.skipped += 1
            return False
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="arena-decide")
        self._snapshot = snapshot
        self._future = self._executor.submit(self._fn, snapshot)
        self.started += 1
        return True

    def poll(self) -> Completed | None:
        """The finished call (once), or None while it is running or when nothing is in flight.
        An exception raised by `fn` is re-raised here, in the caller's thread."""
        f = self._future
        if f is None or not f.done():
            return None
        snapshot = self._snapshot
        self._future = None
        self._snapshot = None
        return Completed(snapshot, f.result())

    def close(self) -> None:
        """Idempotent. Drops the in-flight call without waiting for it: a queued call is
        cancelled, a running one finishes on its thread and its result is discarded."""
        if self._closed:
            return
        self._closed = True
        if self._future is not None:
            self._future.cancel()
            self._future = None
            self._snapshot = None
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
