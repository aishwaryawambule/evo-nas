import numpy as np

CONV_Y_EDGES = np.arange(-0.5, 7.5, 1.0)  # 7 bins for conv_count 0..6

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

    def cell_index(self, params, conv_count):
        return (_bin(params, self.x_edges), _bin(conv_count, self.y_edges))

    def insert(self, genome, record):
        key = self.cell_index(record["params"], record["conv_count"])
        cur = self.cells.get(key)
        if cur is None or record["val_accuracy"] > cur["val_accuracy"]:
            self.cells[key] = {"genome": genome, **record}
            return True
        return False

    def random_elite(self, rng):
        keys = list(self.cells.keys())
        return self.cells[keys[int(rng.integers(len(keys)))]]["genome"]

    def coverage(self, n_reachable):
        return len(self.cells) / n_reachable

    def qd_score(self):
        return sum(c["val_accuracy"] for c in self.cells.values())

    def elites(self):
        return list(self.cells.values())
