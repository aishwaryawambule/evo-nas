"""Export NAS-Bench-201 (via NATS-Bench) to the CSV schema Nb201Benchmark reads.

One-off. Needs `nats_bench` (numpy-only, no torch) and the NATS-tss "simple"
archive:

    pip install nats_bench gdown
    gdown "https://drive.google.com/uc?id=17_saCsj_krKjlCBLOJEpNtzPXArMCqxU" \
        -O data/NATS-tss-simple.tar
    tar xf data/NATS-tss-simple.tar -C data/
    python scripts/export_nb201_csv.py \
        --nats data/NATS-tss-v1_0-3ffb9-simple --out data/nb201_cifar10.csv

Keyed by OUR base-5 genome index (what Nb201Benchmark looks up), NOT NATS-Bench's
internal index — the two differ, so each of our genomes is resolved by its
architecture string via `query_index_by_arch`.

Protocol (cifar10): val_accuracy = the held-out validation accuracy of a model
trained on the train split only (`cifar10-valid` -> `valid-accuracy`); test_accuracy
= the test accuracy of a model trained on train+valid (`cifar10` -> `test-accuracy`).
This keeps search (val) and reporting (test) on separate splits.
"""
import argparse
import csv

from nats_bench import create

from evonas.genome import index_to_genome, genome_to_arch_str

TOTAL = 5 ** 6  # our whole search space

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nats", required=True,
                   help="extracted NATS-tss-*-simple directory (fast_mode)")
    p.add_argument("--out", required=True)
    p.add_argument("--dataset", default="cifar10",
                   help="only cifar10 is wired for the train/valid split protocol")
    args = p.parse_args()

    if args.dataset != "cifar10":
        raise SystemExit("only --dataset cifar10 is supported by this exporter's "
                         "val/test split logic; other datasets use different split names")

    api = create(args.nats, "tss", fast_mode=True, verbose=False)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "val_accuracy", "test_accuracy", "params", "flops"])
        for our_idx in range(TOTAL):
            arch = genome_to_arch_str(index_to_genome(our_idx))
            nats_idx = api.query_index_by_arch(arch)
            if nats_idx < 0:
                raise SystemExit(f"arch not found in benchmark: {arch}")
            val = api.get_more_info(nats_idx, "cifar10-valid", hp="200", is_random=False)
            test = api.get_more_info(nats_idx, "cifar10", hp="200", is_random=False)
            cost = api.get_cost_info(nats_idx, "cifar10")
            w.writerow([our_idx,
                        val["valid-accuracy"] / 100.0,
                        test["test-accuracy"] / 100.0,
                        cost["params"], cost["flops"]])
            if our_idx % 2000 == 0:
                print(f"  {our_idx}/{TOTAL}")
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
