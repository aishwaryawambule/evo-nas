from evonas.archive import Archive, param_bin_edges, DEPTH_Y_EDGES
from evonas.metrics import metrics_snapshot, evaluate, pareto_front

def make_archive():
    return Archive(param_bin_edges([0.0, 1.8], 4), DEPTH_Y_EDGES)

def rec(val, params, cc):
    return {"val_accuracy": val, "test_accuracy": val - 0.01,
            "params": params, "flops": params * 12, "conv_count": cc}

def test_snapshot_reports_eval_coverage_qd_best():
    a = make_archive()
    a.insert((0,0,0,0,0,0), rec(0.8, 0.0, 0))
    s = metrics_snapshot(a, eval_index=5, best_val=0.8, n_reachable=4)
    assert s == {"eval": 5, "coverage": 0.25, "qd_score": 0.8, "best_val": 0.8}

def test_evaluate_ratios_against_ground_truth():
    got = make_archive(); got.insert((0,0,0,0,0,0), rec(0.8, 0.0, 0))
    gt = make_archive()
    gt.insert((0,0,0,0,0,0), rec(0.9, 0.0, 0))
    gt.insert((3,3,0,0,0,0), rec(0.7, 0.6, 2))
    out = evaluate(got, gt, n_reachable=2)
    assert out["coverage_ratio"] == 0.5      # 1 of 2 cells
    assert abs(out["qd_ratio"] - 0.8/1.6) < 1e-9

def test_pareto_front_keeps_only_nondominated():
    records = [
        {"params": 0.2, "test_accuracy": 0.80},
        {"params": 0.4, "test_accuracy": 0.79},  # dominated (bigger, worse)
        {"params": 0.6, "test_accuracy": 0.90},
        {"params": 0.6, "test_accuracy": 0.85},  # dominated (same size, worse)
    ]
    front = pareto_front(records)
    assert {(r["params"], r["test_accuracy"]) for r in front} == {(0.2, 0.80), (0.6, 0.90)}

def test_pareto_front_keeps_exact_duplicate_coordinates():
    records = [
        {"params": 0.2, "test_accuracy": 0.80},
        {"params": 0.6, "test_accuracy": 0.90, "id": "a"},
        {"params": 0.6, "test_accuracy": 0.90, "id": "b"},
    ]
    assert len(pareto_front(records)) == 3

def test_pareto_front_drops_dominated_accuracy_tie():
    records = [
        {"params": 0.2, "test_accuracy": 0.90},
        {"params": 0.6, "test_accuracy": 0.90},  # bigger params, same acc => dominated
    ]
    front = pareto_front(records)
    assert len(front) == 1 and front[0]["params"] == 0.2
