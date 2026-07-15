import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import numpy as np
from evonas.benchmark import FakeBenchmark
from evonas.experiment import build_factory
from evonas.map_elites import map_elites
from evonas.plots import heatmap_figure, convergence_figure, frontier_figure

def _run():
    f = build_factory(FakeBenchmark(), 20)
    return map_elites(FakeBenchmark(), f, 200, 20, np.random.default_rng(0), 140)

def test_heatmap_returns_figure():
    archive, _ = _run()
    assert isinstance(heatmap_figure(archive), Figure)

def test_convergence_returns_figure():
    _, h = _run()
    assert isinstance(convergence_figure([h], [h], metric="qd_score"), Figure)

def test_frontier_returns_figure():
    archive, _ = _run()
    pareto = [{"params": e["params"], "test_accuracy": e["test_accuracy"]} for e in archive.elites()]
    assert isinstance(frontier_figure(archive.elites(), pareto), Figure)
