import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

from evonas.benchmark import FakeBenchmark
from evonas.experiment import run_experiment, to_json, from_json
from evonas.uidata import replay_archive, comparison_figures

def test_replay_keeps_best_per_cell_up_to_step():
    history = [
        {"eval": 1, "insert": {"cell": [0, 0], "val_accuracy": 0.8, "genome": [0,0,0,0,0,0]}},
        {"eval": 2, "insert": None},
        {"eval": 3, "insert": {"cell": [0, 0], "val_accuracy": 0.9, "genome": [1,0,0,0,0,0]}},
        {"eval": 4, "insert": {"cell": [1, 0], "val_accuracy": 0.7, "genome": [2,0,0,0,0,0]}},
    ]
    at2 = replay_archive(history, up_to=2)
    assert at2[(0, 0)]["val_accuracy"] == 0.8
    at4 = replay_archive(history, up_to=4)
    assert at4[(0, 0)]["val_accuracy"] == 0.9   # replaced by the better one
    assert (1, 0) in at4

def test_comparison_figures_from_round_tripped_results():
    config = {
        "dataset": "fake",
        "budget": 30,
        "map": {"x_bins": 20},
        "init_random": 10,
        "seeds": [0, 1],
    }
    results = from_json(to_json(run_experiment(config, FakeBenchmark())))
    figs = comparison_figures(results)
    assert set(figs) == {"qd", "coverage", "frontier"}
    assert all(isinstance(f, Figure) for f in figs.values())
