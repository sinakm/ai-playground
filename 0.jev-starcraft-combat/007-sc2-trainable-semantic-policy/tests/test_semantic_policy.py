import numpy as np
from arena.semantic_policy import ACTIONS, INPUT_DIM, HIDDEN_DIM, SemanticMLP

def test_shape_and_probabilities():
    m=SemanticMLP(seed=1)
    x=m.vector({}, {})
    p,h=m.probs(x)
    assert x.shape==(INPUT_DIM,) and h.shape==(HIDDEN_DIM,)
    assert len(p)==len(ACTIONS) and np.isclose(p.sum(),1)

def test_training_changes_weights():
    m=SemanticMLP(seed=1,learning_rate=0.1)
    m.finish_episode(15)  # first episode only sets the baseline
    before=m.w2.copy(); rng=np.random.default_rng(2)
    for _ in range(20): m.act({}, {}, rng, training=True)
    assert m.finish_episode(20) > 0
    assert not np.allclose(before,m.w2)


def test_first_episode_sets_baseline_without_update():
    m=SemanticMLP(seed=1); before=m.w2.copy(); rng=np.random.default_rng(2)
    for _ in range(5): m.act({}, {}, rng, training=True)
    assert m.finish_episode(22) == 0.0
    assert m.baseline == 22 and np.allclose(before, m.w2) and m.trajectory == []


def test_advantage_sign_and_clip():
    m=SemanticMLP(seed=1); m.finish_episode(20)
    assert m.advantage(25) > 0 and m.advantage(0) == -3.0 and m.advantage(20) == 0.0


def test_positive_advantage_raises_chosen_action_probability():
    m=SemanticMLP(seed=1,learning_rate=0.5,entropy=0.0); m.finish_episode(10)
    x=m.vector({}, {}); idx=ACTIONS.index("stutter")
    before=m.probs(x)[0][idx]
    for _ in range(30):
        p,h=m.probs(x); m.trajectory.append((x,h,p,idx,None))
    m.finish_episode(20)
    assert m.probs(x)[0][idx] > before


def test_drop_unexecuted_keeps_only_executed_choices():
    m=SemanticMLP(seed=1); rng=np.random.default_rng(0)
    m.act({}, {}, rng, training=True, key=1)          # earlier step, untouched
    start=len(m.trajectory)
    a2,_,_=m.act({}, {}, rng, training=True, key=2)
    m.act({}, {}, rng, training=True, key=3)
    m.drop_unexecuted(start, {2: a2, 3: "retreat_to_squad"})
    assert [t[4] for t in m.trajectory] == [1, 2]


def test_save_load_keeps_baseline(tmp_path):
    m=SemanticMLP(seed=1); m.finish_episode(18); m.finish_episode(24)
    f=tmp_path/"w.json"; m.save(f)
    n=SemanticMLP(seed=9); n.load(f)
    assert n.baseline == m.baseline and n.episodes == 2 and np.allclose(n.w1, m.w1)
