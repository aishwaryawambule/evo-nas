import math

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

from evonas.benchmark import FakeBenchmark
from evonas.experiment import run_experiment, to_json, from_json
from evonas.uidata import (replay_archive, comparison_figures, describe_genome,
                           replay_grid, archive_table)

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

def test_describe_genome_renders_every_edge_readably():
    rows = describe_genome((3, 0, 1, 2, 4, 0))
    assert len(rows) == 6
    assert rows[0] == {"edge": "node1 ← node0", "operation": "conv 3×3"}
    assert rows[1]["operation"] == "none (edge removed)"
    assert rows[2]["operation"] == "skip connection"
    # no raw op indices leak into the display
    assert all(not r["operation"][0].isdigit() for r in rows)

def test_replay_grid_leaves_undiscovered_niches_nan():
    results = {"config": {"map": {"x_bins": 4}}}
    history = [
        {"eval": 1, "insert": {"cell": [0, 0], "val_accuracy": 0.8, "genome": [0,0,0,0,0,0]}},
        {"eval": 2, "insert": {"cell": [2, 3], "val_accuracy": 0.9, "genome": [1,0,0,0,0,0]}},
    ]
    grid, n_filled = replay_grid(results, history, up_to=2)
    assert grid.shape == (7, 4)          # 7 conv-count rows
    assert n_filled == 2
    assert grid[0, 0] == 0.8
    assert grid[3, 2] == 0.9
    # an undiscovered niche must be NaN, not 0.0 (0.0 would render as a real bad score)
    assert math.isnan(grid[6, 3])

def _elite(genome, params, acc, conv):
    return {"genome": list(genome), "params": params, "test_accuracy": acc,
            "val_accuracy": acc, "conv_count": conv}

def test_archive_table_shows_every_elite_smallest_first():
    elites = [_elite((3, 3, 3, 3, 3, 3), 1.532, 0.9376, 6),
              _elite((4, 0, 0, 1, 0, 0), 0.073, 0.8663, 0),
              _elite((3, 2, 3, 1, 2, 2), 0.643, 0.9431, 5)]
    rows, _ = archive_table(elites)
    assert len(rows) == 3                       # the whole archive, not one row
    assert [r["params (M)"] for r in rows] == [0.073, 0.643, 1.532]
    assert rows[0]["ops on the 6 edges"] == "pool / – / – / skip / – / –"
    assert rows[0]["test acc %"] == 86.63       # a percentage, not a fraction
    # no blank marker column — the selection is reported as an index instead
    assert "" not in rows[0]

def test_archive_table_reports_the_selected_row_index():
    elites = [_elite((3, 3, 3, 3, 3, 3), 1.532, 0.9376, 6),
              _elite((4, 0, 0, 1, 0, 0), 0.073, 0.8663, 0)]
    # rows are sorted smallest-first, so the 0.073M design lands at index 0
    _, idx = archive_table(elites, selected=(4, 0, 0, 1, 0, 0))
    assert idx == 0
    # a genome the query didn't resolve to, and no selection, both yield None
    assert archive_table(elites, selected=(0, 0, 0, 0, 0, 0))[1] is None
    assert archive_table(elites, selected=None)[1] is None

def test_x_edges_persisted_for_labelling_the_size_axis():
    config = {"dataset": "fake", "budget": 20, "map": {"x_bins": 20},
              "init_random": 10, "seeds": [0]}
    results = from_json(to_json(run_experiment(config, FakeBenchmark())))
    assert len(results["x_edges"]) == 21
    assert results["x_edges"] == sorted(results["x_edges"])
