from evonas.uidata import replay_archive

def test_replay_keeps_best_per_cell_up_to_step():
    history = [
        {"eval": 1, "insert": {"cell": [0, 0], "val_accuracy": 0.8, "genome": [0,0,0,0,0,0]}},
        {"eval": 2, "insert": None},
        {"eval": 3, "insert": {"cell": [0, 0], "val_accuracy": 0.9, "genome": [1,0,0,0,0,0]}},
        {"eval": 4, "insert": {"cell": [1, 0], "val_accuracy": 0.7, "genome": [2,0,0,0,0,0]}},
    ]
    at2 = replay_archive(history, up_to=2)
    assert at2[(0, 0)]["val_accuracy"] == 0.8
    at4 = replay_archive(history, up_to=4)
    assert at4[(0, 0)]["val_accuracy"] == 0.9   # replaced by the better one
    assert (1, 0) in at4
