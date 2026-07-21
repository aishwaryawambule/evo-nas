import numpy as np
from evonas.benchmark import FakeBenchmark
from evonas.archive import Archive, param_bin_edges, DEPTH_Y_EDGES
from evonas.genome import index_to_genome
from evonas.map_elites import map_elites

def make_factory():
    all_params = [FakeBenchmark().query(index_to_genome(i))["params"] for i in range(5**6)]
    edges = param_bin_edges(all_params, x_bins=20)
    return lambda: Archive(edges, DEPTH_Y_EDGES)

def test_history_length_equals_budget():
    _, history = map_elites(FakeBenchmark(), make_factory(), budget=200,
                            init_random=20, rng=np.random.default_rng(0), n_reachable=140)
    assert len(history) == 200

def test_history_logs_at_least_one_insert():
    _, history = map_elites(FakeBenchmark(), make_factory(), budget=200,
                            init_random=20, rng=np.random.default_rng(0), n_reachable=140)
    assert any(h["insert"] is not None for h in history)

def test_best_val_is_monotonic_nondecreasing():
    _, history = map_elites(FakeBenchmark(), make_factory(), budget=300,
                            init_random=20, rng=np.random.default_rng(1), n_reachable=140)
    bests = [h["best_val"] for h in history]
    assert all(b2 >= b1 for b1, b2 in zip(bests, bests[1:]))

def test_run_is_reproducible_under_same_seed():
    f = make_factory()
    a1, _ = map_elites(FakeBenchmark(), f, 200, 20, np.random.default_rng(7), 140)
    a2, _ = map_elites(FakeBenchmark(), f, 200, 20, np.random.default_rng(7), 140)
    assert sorted(a1.cells) == sorted(a2.cells)
    assert a1.qd_score() == a2.qd_score()

def test_coverage_grows_beyond_initial_seed():
    archive, history = map_elites(FakeBenchmark(), make_factory(), budget=500,
                                  init_random=20, rng=np.random.default_rng(2), n_reachable=140)
    assert len(archive.cells) > 1
