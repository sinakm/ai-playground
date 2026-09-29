"""SC2-independent view of a unit, so pure logic can be tested without the game."""

from dataclasses import dataclass


@dataclass(frozen=True)
class UnitView:
    id: int
    kind: str
    x: float
    y: float
    hp: float
    stimmed: bool = False
