import argparse
from evonas.experiment import from_json
from evonas.select import select_design

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--map", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-params", type=float, default=None)
    p.add_argument("--min-accuracy", type=float, default=None)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--smallest", action="store_true",
                      help="smallest model meeting the constraints")
    mode.add_argument("--best", action="store_true",
                      help="most accurate model meeting the constraints (default)")
    args = p.parse_args()

    with open(args.map) as f:
        results = from_json(f.read())
    seeds = results["seeds"]
    if args.seed is None:
        entry = seeds[0]
    else:
        entry = next((s for s in seeds if s["seed"] == args.seed), None)
        if entry is None:
            available = ", ".join(str(s["seed"]) for s in seeds)
            raise SystemExit(
                f"seed {args.seed} is not in {args.map} (available: {available})")
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
