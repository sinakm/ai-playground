from types import SimpleNamespace

from arena.jev import (
    JevPerceptionClient, PERCEPTION_GLOBALS, LOCAL_PERCEPTIONS, RELATIONAL_PERCEPTIONS,
)


class FakeSystemOne:
    def __init__(self):
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        answers = {}
        for key, q in questions.items():
            if key == "priority_target":
                choice = next(iter(q.criteria))
                answers[key] = SimpleNamespace(
                    choice=choice, confidence=0.9, probabilities={choice: 0.9}
                )
            else:
                answers[key] = SimpleNamespace(noul=0.6)
        return SimpleNamespace(
            answers=answers,
            usage=SimpleNamespace(input_tokens=123, output_tokens=4),
            model="jev-test",
        )


def test_perception_builds_node_then_edge_graph():
    state = {
        "marines": [
            {"id": 1, "x": 0.0, "y": 0.0, "teammates": []},
            {"id": 2, "x": 1.0, "y": 0.0, "teammates": []},
        ],
        "enemies": [
            {"id": 20, "kind": "baneling", "x": 3.0, "y": 0.0, "hp": 30},
            {"id": 21, "kind": "baneling", "x": 5.0, "y": 0.0, "hp": 30},
        ],
        "priority_candidates": [
            {"id": 20, "distance_to_squad_center": 2.0, "nearest_marine_distance": 1.0},
            {"id": 21, "distance_to_squad_center": 4.0, "nearest_marine_distance": 2.0},
        ],
    }
    fake = FakeSystemOne()
    a = JevPerceptionClient(client=fake).ask(state, [1, 2])

    assert len(fake.calls) == 2
    node_state, node_questions = fake.calls[0]
    edge_state, edge_questions = fake.calls[1]

    assert "semantic_priority_target" not in node_state
    assert edge_state["semantic_priority_target"]["id"] == 20
    assert edge_state["semantic_priority_target"]["enemy"]["id"] == 20

    assert set(a.global_activations) == set(PERCEPTION_GLOBALS)
    assert set(a.local_activations[1]) == set(LOCAL_PERCEPTIONS)
    assert set(a.relational_activations[1]) == set(RELATIONAL_PERCEPTIONS)
    assert set(a.relational_activations[2]) == set(RELATIONAL_PERCEPTIONS)
    assert a.relational_activations[1]["target_engagement_value"] == 0.6
    assert a.priority_target == 20
    assert a.input_tokens == 246 and a.output_tokens == 8
    assert a.latency_ms == a.node_latency_ms + a.edge_latency_ms
