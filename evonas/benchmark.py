from typing import Protocol
from evonas.genome import genome_to_index, conv_count

class Benchmark(Protocol):
    def query(self, g) -> dict: ...

class FakeBenchmark:
    """Deterministic stand-in for NAS-Bench-201 used in tests and demos.

    Accuracy rises with conv ops and falls with 'none' ops; a tiny
    index-derived jitter breaks ties. No randomness, no data file.
    """
    OP_PARAMS = (0.0, 0.0, 0.10, 0.30, 0.0)  # per-op params in millions

    def __init__(self):
        self._cache = {}

    def query(self, g):
        if g in self._cache:
            return self._cache[g]
        cc = conv_count(g)
        n_none = sum(1 for op in g if op == 0)
        params = round(sum(self.OP_PARAMS[op] for op in g), 6)
        flops = round(params * 12.0, 6)
        jitter = (genome_to_index(g) % 97) / 10000.0  # 0..0.0096
        val = 0.50 + 0.06 * cc - 0.015 * n_none + jitter
        val = round(max(0.10, min(0.95, val)), 6)
        rec = {
            "val_accuracy": val,
            "test_accuracy": round(val - 0.008, 6),
            "params": params,
            "flops": flops,
            "conv_count": cc,
        }
        self._cache[g] = rec
        return rec
