def metrics_snapshot(archive, eval_index, best_val, n_reachable):
    return {
        "eval": eval_index,
        "coverage": archive.coverage(n_reachable),
        "qd_score": archive.qd_score(),
        "best_val": best_val,
    }

def evaluate(archive, gt_archive, n_reachable):
    gt_qd = gt_archive.qd_score()
    return {
        "coverage_ratio": archive.coverage(n_reachable),
        "qd_ratio": (archive.qd_score() / gt_qd) if gt_qd else 0.0,
    }

def pareto_front(records):
    ordered = sorted(records, key=lambda r: (r["params"], -r["test_accuracy"]))
    front, best = [], float("-inf")
    for r in ordered:
        if r["test_accuracy"] > best:
            front.append(r)
            best = r["test_accuracy"]
        elif (r["test_accuracy"] == best and front
              and r["params"] == front[-1]["params"]):
            front.append(r)
    return front
