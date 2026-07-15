from evonas.experiment import from_json

def load_results(path):
    with open(path) as f:
        return from_json(f.read())

def replay_archive(history, up_to):
    cells = {}
    for snap in history[:up_to]:
        ins = snap.get("insert")
        if ins is None:
            continue
        key = tuple(ins["cell"])
        cur = cells.get(key)
        if cur is None or ins["val_accuracy"] > cur["val_accuracy"]:
            cells[key] = {"val_accuracy": ins["val_accuracy"], "genome": ins["genome"]}
    return cells

def comparison_figures(results):
    from evonas.plots import convergence_figure, frontier_figure
    me = [s["map_elites"]["history"] for s in results["seeds"]]
    rs = [s["random"]["history"] for s in results["seeds"]]
    elites = results["seeds"][0]["map_elites"]["elites"]
    return {
        "qd": convergence_figure(me, rs, metric="qd_score"),
        "coverage": convergence_figure(me, rs, metric="coverage"),
        "frontier": frontier_figure(elites, results["ground_truth"]["pareto"]),
    }
