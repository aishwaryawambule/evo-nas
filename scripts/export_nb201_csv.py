"""Export NAS-Bench-201 to the CSV schema used by Nb201Benchmark.

Usage:
    pip install nats_bench
    # download NATS-tss-v1_0-3ffb9.pickle.pbz2 per the NATS-Bench README
    python scripts/export_nb201_csv.py --nats <path-to-pickle> --out data/nb201_cifar10.csv

Accuracy fields for cifar10: 'x-valid' (val) and 'ori-test' (test), scaled to [0,1].
Architecture index i maps to our genome via index_to_genome(i); both use the
same edge ordering (node1<-0, node2<-0, node2<-1, node3<-0, node3<-1, node3<-2).
"""
import argparse, csv
from nats_bench import create
from evonas.genome import index_to_genome, genome_to_arch_str

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nats", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dataset", default="cifar10")
    args = p.parse_args()
    api = create(args.nats, "tss", fast_mode=True, verbose=False)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "val_accuracy", "test_accuracy", "params", "flops"])
        for idx in range(len(api)):
            # sanity: our genome's arch string must match the API's
            assert api.arch(idx) == genome_to_arch_str(index_to_genome(idx)), idx
            info = api.get_more_info(idx, args.dataset, hp="200")
            cost = api.get_cost_info(idx, args.dataset)
            w.writerow([idx,
                        info["valid-accuracy"] / 100.0,
                        info["test-accuracy"] / 100.0,
                        cost["params"], cost["flops"]])
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
