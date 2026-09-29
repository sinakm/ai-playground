from arena.overlay import (
    PANEL_SIZE,
    POLICY_LABELS,
    _results_lines,
    card,
    cumulative,
    decision_index_for_frames,
    marine_headline,
    panel_frame,
    trim_offset,
)

REC = {
    "action": "spread", "probabilities": {"spread": 0.9, "clump": 0.02, "retreat": 0.03, "attack": 0.04, "stim": 0.01},
    "confidence": 0.88, "latency_ms": 171.0, "input_tokens": 1_000_000, "output_tokens": 0,
    "marines_alive": 12, "enemies_alive": 16, "late": False,
}


def test_decision_index_for_frames():
    idx = decision_index_for_frames([100.05, 100.19], start_wall=100.0, n_frames=10, fps=30)
    # frame t = 0, .033, .067, .1, .133, .167, .2, .233, ...
    assert idx == [-1, -1, 0, 0, 0, 0, 1, 1, 1, 1]


def test_cumulative_totals():
    c = cumulative([REC, REC])
    assert c[1]["decisions"] == 2
    assert c[1]["cost_usd"] == 0.084


def test_panel_and_card_sizes():
    assert panel_frame(None, None).size == PANEL_SIZE
    assert panel_frame(REC, cumulative([REC])[0]).size == PANEL_SIZE
    assert card(["Jev vs Banelings", "episode 001"]).size == (1920, 1080)


def test_panel_frame_accepts_custom_header():
    assert panel_frame(None, None, header="RANDOM PICKS").size == PANEL_SIZE


def test_panel_frame_handles_none_confidence_and_probabilities():
    rec = {**REC, "confidence": None, "probabilities": None}
    assert panel_frame(rec, cumulative([rec])[0]).size == PANEL_SIZE


def test_trim_offset():
    assert trim_offset([], record_start_wall=100.0) == 0.0
    assert trim_offset([{**REC, "wall_time": 105.0}], record_start_wall=100.0) == 4.0
    assert trim_offset([{**REC, "wall_time": 100.5}], record_start_wall=100.0) == 0.0


def test_results_lines_include_score():
    results = {
        "jev_commander": {
            "wins": 3, "runs": 10, "mean_marines_alive": 2.5, "mean_enemies_killed": 9.1, "mean_score": 7.3,
        },
    }
    lines = _results_lines(results)
    assert any("score 7.3" in line for line in lines)


MARINE_REC = {
    **REC,
    "action": "kite",
    "probabilities": {"kite": 0.5, "split": 0.0, "attack": 0.25, "stim": 0.0, "retreat": 0.25},
    "confidence": 0.71,
    "marines_alive": 4,
    "marine_actions": {"1": "kite", "2": "kite", "3": "attack", "4": "retreat"},
    "marine_confidences": {"1": 0.9, "2": 0.5, "3": 0.7, "4": 0.74},
}


def test_panel_frame_per_marine_record():
    assert panel_frame(MARINE_REC, cumulative([MARINE_REC])[0], header="JEV DECIDES PER MARINE").size == PANEL_SIZE


def test_panel_frame_per_marine_random_record():
    rec = {**MARINE_REC, "confidence": None, "marine_confidences": None, "action": "retreat",
           "marine_actions": {str(i): "retreat" for i in range(12)}}
    assert panel_frame(rec, cumulative([rec])[0]).size == PANEL_SIZE


def test_marine_headline():
    assert marine_headline(MARINE_REC) == "KITE 2/4"


def test_per_marine_policy_labels():
    assert POLICY_LABELS["random"][:2] == ("RANDOM PICKS PER MARINE", "Random vs Banelings")
