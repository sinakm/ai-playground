from types import SimpleNamespace

from arena.jev import JevPerceptionClient, PERCEPTION_GLOBALS, LOCAL_PERCEPTIONS, RELATIONAL_PERCEPTIONS


class FakeSystemOne:
    def system_one(self, state, questions):
        answers = {}
        for key, q in questions.items():
            if key == "priority_target":
                choice = next(iter(q.criteria))
                answers[key] = SimpleNamespace(choice=choice, confidence=0.9, probabilities={choice: 0.9})
            else:
                answers[key] = SimpleNamespace(noul=0.6)
        return SimpleNamespace(
            answers=answers,
            usage=SimpleNamespace(input_tokens=123, output_tokens=4),
            model="jev-test",
        )


def test_perception_batches_scalars_locals_and_referential_target():
    state = {
        "priority_candidates": [
            {"id": 20, "distance_to_squad_center": 2.0, "nearest_marine_distance": 1.0},
            {"id": 21, "distance_to_squad_center": 4.0, "nearest_marine_distance": 2.0},
        ]
    }
    a = JevPerceptionClient(client=FakeSystemOne()).ask(state, [1, 2])
    assert set(a.global_activations) == set(PERCEPTION_GLOBALS)
    assert "stim_opportunity" in a.global_activations
    assert set(a.local_activations[1]) == set(LOCAL_PERCEPTIONS) | set(RELATIONAL_PERCEPTIONS)
    assert set(a.local_activations[2]) == set(LOCAL_PERCEPTIONS) | set(RELATIONAL_PERCEPTIONS)
    assert "should_focus_priority_target" in a.local_activations[1]
    assert "ally_needs_cover" in a.local_activations[1]
    assert a.priority_target == 20
    assert a.input_tokens == 123 and a.output_tokens == 4
