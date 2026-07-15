import numpy as np
from evonas.operators import mutate

def test_mutate_changes_exactly_one_edge():
    rng = np.random.default_rng(1)
    parent = (0, 0, 0, 0, 0, 0)
    for _ in range(50):
        child = mutate(parent, rng)
        diffs = [i for i in range(6) if child[i] != parent[i]]
        assert len(diffs) == 1

def test_mutated_edge_takes_a_different_value():
    rng = np.random.default_rng(2)
    parent = (1, 2, 3, 4, 0, 1)
    for _ in range(50):
        child = mutate(parent, rng)
        i = next(i for i in range(6) if child[i] != parent[i])
        assert child[i] != parent[i]
        assert 0 <= child[i] <= 4

def test_mutate_returns_length_six_tuple():
    rng = np.random.default_rng(3)
    child = mutate((0, 1, 2, 3, 4, 0), rng)
    assert isinstance(child, tuple) and len(child) == 6
