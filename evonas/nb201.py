import pandas as pd
from evonas.genome import genome_to_index, conv_count

class Nb201Benchmark:
    def __init__(self, csv_path, dataset="cifar10"):
        self.dataset = dataset
        df = pd.read_csv(csv_path)
        self._rows = {int(r["index"]): r for _, r in df.iterrows()}

    def query(self, g):
        r = self._rows[genome_to_index(g)]
        return {
            "val_accuracy": float(r["val_accuracy"]),
            "test_accuracy": float(r["test_accuracy"]),
            "params": float(r["params"]),
            "flops": float(r["flops"]),
            "conv_count": conv_count(g),
        }
