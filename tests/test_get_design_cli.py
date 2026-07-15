import subprocess, sys, textwrap
from pathlib import Path

def _make_map(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(textwrap.dedent("""
        dataset: fake
        budget: 200
        map: {x_bins: 20}
        init_random: 20
        seeds: [0]
    """))
    out = tmp_path / "res.json"
    subprocess.run([sys.executable, "run_search.py", "--config", str(cfg),
                    "--out", str(out)], check=True, cwd=Path.cwd())
    return out

def test_get_design_prints_a_genome(tmp_path):
    out = _make_map(tmp_path)
    r = subprocess.run([sys.executable, "get_design.py", "--map", str(out), "--best"],
                       check=True, capture_output=True, text=True, cwd=Path.cwd())
    assert "genome" in r.stdout.lower()

def test_get_design_reports_no_match(tmp_path):
    out = _make_map(tmp_path)
    r = subprocess.run([sys.executable, "get_design.py", "--map", str(out),
                        "--max-params", "-1"], capture_output=True, text=True, cwd=Path.cwd())
    assert "no design" in r.stdout.lower()
