from evonas.benchmark import FakeBenchmark
from evonas.archive import Archive, param_bin_edges, CONV_Y_EDGES
from evonas.genome import index_to_genome
from evonas.ground_truth import enumerate_all, ground_truth_archive, reachable_cells

def make_factory():
    all_params = [FakeBenchmark().query(index_to_genome(i))["params"] for i in range(5**6)]
    edges = param_bin_edges(all_params, x_bins=20)
    return lambda: Archive(edges, CONV_Y_EDGES)

def test_enumerate_visits_whole_space():
    assert sum(1 for _ in enumerate_all(FakeBenchmark())) == 15625

def test_ground_truth_elite_is_true_max_per_cell():
    b = FakeBenchmark()
    factory = make_factory()
    gt = ground_truth_archive(b, factory)
    probe = factory()
    best = {}
    for idx in range(5 ** 6):
        g = index_to_genome(idx)
        rec = b.query(g)
        key = probe.cell_index(rec["params"], rec["conv_count"])
        if key not in best or rec["val_accuracy"] > best[key]:
            best[key] = rec["val_accuracy"]
    assert {k: c["val_accuracy"] for k, c in gt.cells.items()} == best

def test_reachable_cells_is_positive_and_bounded():
    n = reachable_cells(FakeBenchmark(), make_factory())
    assert 0 < n <= 140
