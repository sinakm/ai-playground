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
    before=m.w2.copy(); rng=np.random.default_rng(2)
    for _ in range(20): m.act({}, {}, rng, training=True)
    m.finish_episode(20)
    assert not np.allclose(before,m.w2)
