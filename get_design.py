import argparse
from evonas.experiment import from_json
from evonas.select import select_design

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--map", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-params", type=float, default=None)
    p.add_argument("--min-accuracy", type=float, default=None)
    p.add_argument("--smallest", action="store_true")
    p.add_argument("--best", action="store_true")
    args = p.parse_args()

    with open(args.map) as f:
        results = from_json(f.read())
    seeds = results["seeds"]
    entry = next((s for s in seeds if s["seed"] == args.seed), seeds[0])
    elites = entry["map_elites"]["elites"]

    d = select_design(elites, max_params=args.max_params,
                      min_accuracy=args.min_accuracy,
                      smallest=args.smallest, best=args.best)
    if d is None:
        print("No design matches those constraints.")
        return
    print(f"genome:   {d['genome']}")
    print(f"test acc: {d['test_accuracy']:.4f}")
    print(f"params:   {d['params']}")
    print(f"conv:     {d['conv_count']}")

if __name__ == "__main__":
    main()
