import numpy as np

from evonas.experiment import from_json
from evonas.genome import OPS
from evonas.archive import CONV_Y_EDGES

# NAS-Bench-201 cell edges, in genome order: node i receives from every node j < i.
EDGE_LABELS = ("node1 ← node0", "node2 ← node0", "node2 ← node1",
               "node3 ← node0", "node3 ← node1", "node3 ← node2")

OP_DISPLAY = {
    "none": "none (edge removed)",
    "skip_connect": "skip connection",
    "nor_conv_1x1": "conv 1×1",
    "nor_conv_3x3": "conv 3×3",
    "avg_pool_3x3": "avg-pool 3×3",
}

def load_results(path):
    with open(path) as f:
        return from_json(f.read())

def describe_genome(genome):
    """Turn the 6 raw op indices into readable (edge, operation) rows."""
    return [{"edge": EDGE_LABELS[i], "operation": OP_DISPLAY[OPS[op]]}
            for i, op in enumerate(genome)]

def replay_grid(results, history, up_to):
    """Archive grid as of `up_to` evaluations, plus how many niches are filled.

    Undiscovered niches stay NaN so they render blank rather than as accuracy 0.
    """
    cells = replay_archive(history, up_to)
    grid = np.full((len(CONV_Y_EDGES) - 1, results["config"]["map"]["x_bins"]), np.nan)
    for (i, j), c in cells.items():
        grid[j, i] = c["val_accuracy"]
    return grid, len(cells)

def replay_archive(history, up_to):
    cells = {}
    for snap in history[:up_to]:
        ins = snap.get("insert")
        if ins is None:
            continue
        key = tuple(ins["cell"])
        cur = cells.get(key)
        if cur is None or ins["val_accuracy"] > cur["val_accuracy"]:
            cells[key] = {"val_accuracy": ins["val_accuracy"], "genome": ins["genome"]}
    return cells

def comparison_figures(results):
    from evonas.plots import convergence_figure, frontier_figure
    me = [s["map_elites"]["history"] for s in results["seeds"]]
    rs = [s["random"]["history"] for s in results["seeds"]]
    elites = results["seeds"][0]["map_elites"]["elites"]
    return {
        "qd": convergence_figure(me, rs, metric="qd_score"),
        "coverage": convergence_figure(me, rs, metric="coverage"),
        "frontier": frontier_figure(elites, results["ground_truth"]["pareto"]),
    }
