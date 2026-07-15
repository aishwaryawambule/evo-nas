def select_design(elites, max_params=None, min_accuracy=None, smallest=False, best=False):
    cand = list(elites)
    if max_params is not None:
        cand = [e for e in cand if e["params"] <= max_params]
    if min_accuracy is not None:
        cand = [e for e in cand if e["test_accuracy"] >= min_accuracy]
    if not cand:
        return None
    if smallest:
        return min(cand, key=lambda e: e["params"])
    # `best` and the default both maximize accuracy within constraints
    return max(cand, key=lambda e: e["test_accuracy"])
