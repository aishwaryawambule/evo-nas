from evonas.nb201 import Nb201Benchmark
from evonas.genome import genome_to_index, index_to_genome

def _write_csv(path):
    rows = ["index,val_accuracy,test_accuracy,params,flops"]
    for idx in (0, 1, 15624):
        rows.append(f"{idx},0.9,0.89,1.1,55.0")
    path.write_text("\n".join(rows) + "\n")

def test_query_reads_row_by_index(tmp_path):
    csv = tmp_path / "nb.csv"; _write_csv(csv)
    b = Nb201Benchmark(str(csv))
    rec = b.query(index_to_genome(1))
    assert rec["val_accuracy"] == 0.9
    assert rec["test_accuracy"] == 0.89
    assert rec["params"] == 1.1

def test_query_derives_conv_count_from_genome(tmp_path):
    csv = tmp_path / "nb.csv"; _write_csv(csv)
    b = Nb201Benchmark(str(csv))
    g = index_to_genome(0)  # all-none => conv_count 0
    assert b.query(g)["conv_count"] == 0
