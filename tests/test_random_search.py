import numpy as np
from evonas.benchmark import FakeBenchmark
from evonas.archive import Archive, param_bin_edges, DEPTH_Y_EDGES
from evonas.genome import index_to_genome
from evonas.random_search import random_search

def make_factory():
    all_params = [FakeBenchmark().query(index_to_genome(i))["params"] for i in range(5**6)]
    edges = param_bin_edges(all_params, x_bins=20)
    return lambda: Archive(edges, DEPTH_Y_EDGES)

def test_history_length_equals_budget():
    _, history = random_search(FakeBenchmark(), make_factory(), budget=200,
                               rng=np.random.default_rng(0), n_reachable=140)
    assert len(history) == 200

def test_reproducible_under_same_seed():
    f = make_factory()
    a1, _ = random_search(FakeBenchmark(), f, 200, np.random.default_rng(3), 140)
    a2, _ = random_search(FakeBenchmark(), f, 200, np.random.default_rng(3), 140)
    assert a1.qd_score() == a2.qd_score()
