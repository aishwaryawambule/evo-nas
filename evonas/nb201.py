import pandas as pd

from evonas.genome import NUM_ARCHS, conv_count, genome_to_index

REQUIRED_COLUMNS = ("index", "val_accuracy", "test_accuracy", "params", "flops")

class Nb201Benchmark:
    """Real NAS-Bench-201 scores, read from a CSV exported by
    scripts/export_nb201_csv.py.

    The CSV is keyed by our base-5 genome index (NOT NATS-Bench's internal
    index, which differs). Accuracies are fractions in [0, 1]; params are in
    millions.
    """

    def __init__(self, csv_path, dataset="cifar10"):
        self.dataset = dataset
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"no NAS-Bench-201 CSV at {csv_path!r}. Export one first:\n"
                "  python scripts/export_nb201_csv.py "
                "--nats data/NATS-tss-v1_0-3ffb9-simple --out data/nb201_cifar10.csv"
            ) from None

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"{csv_path}: missing required column(s) {missing}. "
                f"Expected schema: {', '.join(REQUIRED_COLUMNS)}")

        # to_dict is ~50x faster than iterrows and gives plain floats, not Series
        self._rows = {
            int(r["index"]): r
            for r in df[list(REQUIRED_COLUMNS)].to_dict("records")
        }
        if len(self._rows) != NUM_ARCHS:
            raise ValueError(
                f"{csv_path}: expected {NUM_ARCHS} architectures, found "
                f"{len(self._rows)}. The export looks truncated — re-run "
                "scripts/export_nb201_csv.py.")

    def query(self, g):
        idx = genome_to_index(g)
        try:
            r = self._rows[idx]
        except KeyError:
            raise KeyError(
                f"genome {tuple(g)} (index {idx}) is not in {self.dataset} "
                "CSV — the export is incomplete") from None
        return {
            "val_accuracy": float(r["val_accuracy"]),
            "test_accuracy": float(r["test_accuracy"]),
            "params": float(r["params"]),
            "flops": float(r["flops"]),
            "conv_count": conv_count(g),
        }
