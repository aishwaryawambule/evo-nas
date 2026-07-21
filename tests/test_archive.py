import numpy as np
from evonas.archive import Archive, param_bin_edges, DEPTH_Y_EDGES

def make_archive():
    x = param_bin_edges([0.0, 1.8], x_bins=4)  # params range 0..1.8
    return Archive(x, DEPTH_Y_EDGES)

def rec(val, params, cc):
    return {"val_accuracy": val, "test_accuracy": val - 0.01,
            "params": params, "flops": params * 12, "conv_count": cc}

def test_depth_y_edges_give_four_bins():
    assert len(DEPTH_Y_EDGES) == 5  # 4 bins (depth 0..3) => 5 edges

def test_insert_into_empty_cell_returns_true():
    a = make_archive()
    assert a.insert((0,0,0,0,0,0), rec(0.8, 0.0, 0)) is True
    assert len(a.cells) == 1

def test_better_val_replaces_incumbent():
    a = make_archive()
    a.insert((0,0,0,0,0,0), rec(0.80, 0.0, 0))
    assert a.insert((1,0,0,0,0,0), rec(0.90, 0.0, 0)) is True  # same cell, better
    assert len(a.cells) == 1
    assert a.elites()[0]["genome"] == (1,0,0,0,0,0)

def test_worse_or_equal_val_is_rejected():
    a = make_archive()
    a.insert((0,0,0,0,0,0), rec(0.90, 0.0, 0))
    assert a.insert((1,0,0,0,0,0), rec(0.90, 0.0, 0)) is False  # equal
    assert a.insert((1,0,0,0,0,0), rec(0.80, 0.0, 0)) is False  # worse

def test_qd_score_sums_elite_val_accuracy():
    a = make_archive()
    a.insert((0,0,0,0,0,0), rec(0.80, 0.0, 0))
    a.insert((3,3,0,0,0,0), rec(0.70, 0.6, 2))  # different params => different x-cell
    assert abs(a.qd_score() - 1.50) < 1e-9

def test_depth_splits_same_size_designs_into_different_cells():
    a = make_archive()
    # identical size, different wiring: a direct input->output edge (depth 1)
    # vs a two-hop chain (depth 2). Under the old conv-count axis these could
    # collide; depth keeps them apart — the whole reason for the descriptor.
    a.insert((0, 0, 0, 2, 0, 0), rec(0.80, 0.5, 1))  # node0->node3, depth 1
    a.insert((2, 0, 0, 0, 2, 0), rec(0.70, 0.5, 2))  # node0->node1->node3, depth 2
    assert len(a.cells) == 2
    assert {c["depth"] for c in a.elites()} == {1, 2}

def test_coverage_is_filled_over_reachable():
    a = make_archive()
    a.insert((0,0,0,0,0,0), rec(0.80, 0.0, 0))
    assert a.coverage(4) == 0.25

def test_random_elite_returns_a_stored_genome():
    a = make_archive()
    a.insert((0,0,0,0,0,0), rec(0.80, 0.0, 0))
    assert a.random_elite(np.random.default_rng(0)) == (0,0,0,0,0,0)
