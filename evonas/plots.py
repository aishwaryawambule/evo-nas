import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def grid_heatmap_figure(grid, x_edges=None):
    """Render an archive grid. NaN cells (undiscovered niches) stay blank.

    x_edges labels the size axis with real param values when available.
    """
    fig, ax = plt.subplots()
    if x_edges is not None and len(x_edges) >= 2:
        extent = (float(x_edges[0]), float(x_edges[-1]), -0.5, grid.shape[0] - 0.5)
        im = ax.imshow(grid, origin="lower", aspect="auto", extent=extent)
        ax.set_xlabel("model size (millions of parameters)")
    else:
        im = ax.imshow(grid, origin="lower", aspect="auto")
        ax.set_xlabel("model size bin (small → large)")
    ax.set_ylabel("cell depth (longest input→output path)")
    ax.set_yticks(range(grid.shape[0]))
    fig.colorbar(im, ax=ax, label="best val accuracy in niche")
    return fig

def heatmap_figure(archive):
    nx = len(archive.x_edges) - 1
    ny = len(archive.y_edges) - 1
    grid = np.full((ny, nx), np.nan)
    for (i, j), cell in archive.cells.items():
        grid[j, i] = cell["val_accuracy"]
    return grid_heatmap_figure(grid, archive.x_edges)

METRIC_LABELS = {
    "qd_score": "QD-score (Σ best val accuracy over filled niches)",
    "coverage": "coverage (fraction of reachable niches filled)",
    "best_val": "best val accuracy found so far",
}

def convergence_figure(me_histories, rs_histories, metric="qd_score"):
    fig, ax = plt.subplots()
    for label, hists in (("MAP-Elites", me_histories), ("random search", rs_histories)):
        arr = np.array([[h[metric] for h in hist] for hist in hists])
        mean = arr.mean(axis=0)
        x = np.arange(1, len(mean) + 1)
        ax.plot(x, mean, label=label)
        if arr.shape[0] > 1:
            sd = arr.std(axis=0)
            ax.fill_between(x, mean - sd, mean + sd, alpha=0.2)
    ax.set_xlabel("architectures evaluated")
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.legend()
    return fig

def frontier_figure(discovered_elites, true_pareto):
    fig, ax = plt.subplots()
    d = sorted(discovered_elites, key=lambda e: e["params"])
    ax.scatter([e["params"] for e in d], [e["test_accuracy"] for e in d],
               s=12, label="archive elites (found by MAP-Elites)")
    t = sorted(true_pareto, key=lambda e: e["params"])
    ax.plot([e["params"] for e in t], [e["test_accuracy"] for e in t],
            color="black", label="true Pareto front (all 15,625 enumerated)")
    ax.set_xlabel("model size (millions of parameters)")
    ax.set_ylabel("test accuracy")
    ax.legend()
    return fig
