import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def heatmap_figure(archive):
    nx = len(archive.x_edges) - 1
    ny = len(archive.y_edges) - 1
    grid = np.full((ny, nx), np.nan)
    for (i, j), cell in archive.cells.items():
        grid[j, i] = cell["val_accuracy"]
    fig, ax = plt.subplots()
    im = ax.imshow(grid, origin="lower", aspect="auto")
    ax.set_xlabel("model size bin (params)")
    ax.set_ylabel("conv count")
    fig.colorbar(im, ax=ax, label="val accuracy")
    return fig

def convergence_figure(me_histories, rs_histories, metric="qd_score"):
    fig, ax = plt.subplots()
    for label, hists in (("MAP-Elites", me_histories), ("random", rs_histories)):
        arr = np.array([[h[metric] for h in hist] for hist in hists])
        mean = arr.mean(axis=0)
        x = np.arange(1, len(mean) + 1)
        ax.plot(x, mean, label=label)
        if arr.shape[0] > 1:
            sd = arr.std(axis=0)
            ax.fill_between(x, mean - sd, mean + sd, alpha=0.2)
    ax.set_xlabel("evaluations")
    ax.set_ylabel(metric)
    ax.legend()
    return fig

def frontier_figure(discovered_elites, true_pareto):
    fig, ax = plt.subplots()
    d = sorted(discovered_elites, key=lambda e: e["params"])
    ax.scatter([e["params"] for e in d], [e["test_accuracy"] for e in d],
               s=12, label="discovered elites")
    t = sorted(true_pareto, key=lambda e: e["params"])
    ax.plot([e["params"] for e in t], [e["test_accuracy"] for e in t],
            color="black", label="true pareto front")
    ax.set_xlabel("params")
    ax.set_ylabel("test accuracy")
    ax.legend()
    return fig
