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
