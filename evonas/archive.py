import numpy as np

from evonas.genome import cell_depth

DEPTH_Y_EDGES = np.arange(-0.5, 4.5, 1.0)  # 4 bins for cell depth 0..3

def param_bin_edges(all_params, x_bins):
    lo, hi = float(min(all_params)), float(max(all_params))
    if hi <= lo:
        hi = lo + 1e-9
    return np.linspace(lo, hi, x_bins + 1)

def _bin(value, edges):
    i = int(np.digitize([value], edges)[0]) - 1
    return int(np.clip(i, 0, len(edges) - 2))

class Archive:
    def __init__(self, x_edges, y_edges):
        self.x_edges = np.asarray(x_edges, dtype=float)
        self.y_edges = np.asarray(y_edges, dtype=float)
        self.cells = {}

    def cell_index(self, params, y_value):
        return (_bin(params, self.x_edges), _bin(y_value, self.y_edges))

    def cell_of(self, genome, record):
        """The (x, y) niche a design occupies: model size on x, cell depth on y.
        Depth comes from the genome's wiring, so it is independent of size."""
        return self.cell_index(record["params"], cell_depth(genome))

    def insert(self, genome, record):
        depth = cell_depth(genome)
        key = self.cell_index(record["params"], depth)
        cur = self.cells.get(key)
        if cur is None or record["val_accuracy"] > cur["val_accuracy"]:
            self.cells[key] = {"genome": genome, "depth": depth, **record}
            return True
        return False

    def random_elite(self, rng):
        if not self.cells:
            raise ValueError(
                "archive is empty — seed it with random genomes before "
                "selecting a parent (see map_elites' init_random)")
        keys = list(self.cells.keys())
        return self.cells[keys[int(rng.integers(len(keys)))]]["genome"]

    def coverage(self, n_reachable):
        if n_reachable <= 0:
            raise ValueError(f"n_reachable must be positive, got {n_reachable}")
        return len(self.cells) / n_reachable

    def qd_score(self):
        return sum(c["val_accuracy"] for c in self.cells.values())

    def elites(self):
        return list(self.cells.values())
