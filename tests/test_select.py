from evonas.select import select_design

ELITES = [
    {"genome": (0,0,0,0,0,0), "params": 0.2, "test_accuracy": 0.80},
    {"genome": (2,0,0,0,0,0), "params": 0.6, "test_accuracy": 0.90},
    {"genome": (3,3,0,0,0,0), "params": 1.2, "test_accuracy": 0.93},
]

def test_max_params_returns_best_within_budget():
    d = select_design(ELITES, max_params=0.6)
    assert d["genome"] == (2,0,0,0,0,0)

def test_smallest_meeting_accuracy():
    d = select_design(ELITES, min_accuracy=0.90, smallest=True)
    assert d["genome"] == (2,0,0,0,0,0)

def test_best_ignores_size():
    d = select_design(ELITES, best=True)
    assert d["genome"] == (3,3,0,0,0,0)

def test_no_match_returns_none():
    assert select_design(ELITES, max_params=0.1) is None
