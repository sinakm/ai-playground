import dataclasses

import pytest

from arena.views import UnitView


def test_unitview_defaults_and_frozen():
    u = UnitView(id=1, kind="marine", x=1.0, y=2.0, hp=45.0)
    assert u.stimmed is False
    with pytest.raises(dataclasses.FrozenInstanceError):
        u.hp = 10.0
