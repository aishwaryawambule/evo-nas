from evonas.benchmark import FakeBenchmark

def test_query_returns_all_record_keys():
    rec = FakeBenchmark().query((2, 3, 0, 1, 4, 0))
    assert set(rec) == {"val_accuracy", "test_accuracy", "params", "flops", "conv_count"}

def test_query_is_deterministic():
    b = FakeBenchmark()
    assert b.query((1, 2, 3, 4, 0, 1)) == b.query((1, 2, 3, 4, 0, 1))

def test_two_instances_agree():
    assert FakeBenchmark().query((0, 1, 2, 3, 4, 0)) == FakeBenchmark().query((0, 1, 2, 3, 4, 0))

def test_more_conv_ops_generally_scores_higher():
    b = FakeBenchmark()
    low = b.query((0, 0, 0, 0, 0, 0))["val_accuracy"]
    high = b.query((3, 3, 3, 3, 3, 3))["val_accuracy"]
    assert high > low

def test_accuracy_in_valid_range_and_test_below_val():
    b = FakeBenchmark()
    for idx_genome in [(0,0,0,0,0,0), (3,3,3,3,3,3), (2,1,4,0,3,2)]:
        rec = b.query(idx_genome)
        assert 0.1 <= rec["val_accuracy"] <= 0.95
        assert rec["test_accuracy"] <= rec["val_accuracy"]

def test_conv_count_matches_genome():
    assert FakeBenchmark().query((2, 3, 0, 0, 0, 0))["conv_count"] == 2

def test_query_result_is_not_mutable_via_cache():
    b = FakeBenchmark()
    first = b.query((1, 2, 3, 4, 0, 1))
    first["val_accuracy"] = 999.0
    assert b.query((1, 2, 3, 4, 0, 1))["val_accuracy"] != 999.0
