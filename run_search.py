import argparse
from evonas.experiment import load_config, run_experiment, to_json
from evonas.benchmark import FakeBenchmark

def make_benchmark(config):
    if config["dataset"] == "fake":
        return FakeBenchmark()
    from evonas.nb201 import Nb201Benchmark  # imported lazily
    return Nb201Benchmark(config["data_csv"], dataset=config["dataset"])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    config = load_config(args.config)
    results = run_experiment(config, make_benchmark(config))
    with open(args.out, "w") as f:
        f.write(to_json(results))
    print(f"wrote {args.out}: {results['n_reachable']} reachable cells")

if __name__ == "__main__":
    main()
