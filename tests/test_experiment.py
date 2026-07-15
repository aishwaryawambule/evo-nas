from evonas.benchmark import FakeBenchmark
from evonas.experiment import build_factory, run_experiment, to_json, from_json

CONFIG = {
    "dataset": "fake",
    "budget": 150,
    "map": {"x_bins": 20},
    "init_random": 20,
    "seeds": [0, 1],
}

def test_run_produces_results_for_each_seed():
    res = run_experiment(CONFIG, FakeBenchmark())
    assert len(res["seeds"]) == 2
    assert res["n_reachable"] > 0
    assert res["ground_truth"]["n_cells"] == res["n_reachable"]

def test_each_seed_has_both_methods_with_full_history():
    res = run_experiment(CONFIG, FakeBenchmark())
    s0 = res["seeds"][0]
    assert len(s0["map_elites"]["history"]) == 150
    assert len(s0["random"]["history"]) == 150
    assert len(s0["map_elites"]["elites"]) > 0

def test_json_round_trip_preserves_values():
    res = run_experiment(CONFIG, FakeBenchmark())
    back = from_json(to_json(res))
    assert back["n_reachable"] == res["n_reachable"]
    # genomes come back as lists
    assert isinstance(back["seeds"][0]["map_elites"]["elites"][0]["genome"], list)

def test_build_factory_uses_whole_space_for_edges():
    factory = build_factory(FakeBenchmark(), x_bins=20)
    archive = factory()
    assert len(archive.x_edges) == 21
