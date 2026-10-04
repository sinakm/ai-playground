"""JSON Lines decision log."""

from __future__ import annotations

import json
from pathlib import Path


class DecisionLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = path.open("a", encoding="utf-8")
        self._closed = False

    def write(self, record: dict) -> None:
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._f.close()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
