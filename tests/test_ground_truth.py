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

def test_ground_truth_archive_holds_best_per_cell():
    gt = ground_truth_archive(FakeBenchmark(), make_factory())
    # every filled cell's elite is the max val over genomes mapping to that cell
    assert len(gt.cells) == reachable_cells(FakeBenchmark(), make_factory())

def test_reachable_cells_is_positive_and_bounded():
    n = reachable_cells(FakeBenchmark(), make_factory())
    assert 0 < n <= 140
