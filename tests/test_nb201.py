import pytest

from evonas.genome import NUM_ARCHS, index_to_genome
from evonas.nb201 import Nb201Benchmark

def _write_csv(path, n=NUM_ARCHS, columns=None):
    """A complete, well-formed export unless a test asks for otherwise."""
    header = columns or "index,val_accuracy,test_accuracy,params,flops"
    rows = [header]
    for idx in range(n):
        # vary the values a little so tests can't pass on a constant
        rows.append(f"{idx},0.9,0.89,1.1,55.0" if idx != 1
                    else f"{idx},0.5,0.49,0.3,12.0")
    path.write_text("\n".join(rows) + "\n")
    return path

def test_query_reads_row_by_index(tmp_path):
    b = Nb201Benchmark(str(_write_csv(tmp_path / "nb.csv")))
    rec = b.query(index_to_genome(1))
    assert rec["val_accuracy"] == 0.5
    assert rec["test_accuracy"] == 0.49
    assert rec["params"] == 0.3

def test_query_derives_conv_count_from_genome(tmp_path):
    b = Nb201Benchmark(str(_write_csv(tmp_path / "nb.csv")))
    g = index_to_genome(0)  # all-none => conv_count 0
    assert b.query(g)["conv_count"] == 0

def test_truncated_export_is_rejected_at_load(tmp_path):
    # a partial CSV would otherwise fail with a bare KeyError mid-search
    csv = _write_csv(tmp_path / "short.csv", n=100)
    with pytest.raises(ValueError, match="truncated"):
        Nb201Benchmark(str(csv))

def test_missing_column_is_reported_by_name(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("index,val_accuracy,params,flops\n0,0.9,1.1,55.0\n")
    with pytest.raises(ValueError, match="test_accuracy"):
        Nb201Benchmark(str(csv))

def test_missing_file_names_the_export_command(tmp_path):
    with pytest.raises(FileNotFoundError, match="export_nb201_csv"):
        Nb201Benchmark(str(tmp_path / "nope.csv"))
