import json, subprocess, sys, textwrap
from pathlib import Path

def test_cli_writes_results_json(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(textwrap.dedent("""
        dataset: fake
        budget: 120
        map: {x_bins: 20}
        init_random: 20
        seeds: [0]
    """))
    out = tmp_path / "res.json"
    subprocess.run([sys.executable, "run_search.py", "--config", str(cfg),
                    "--out", str(out)], check=True, cwd=Path.cwd())
    data = json.loads(out.read_text())
    assert data["n_reachable"] > 0
    assert len(data["seeds"][0]["map_elites"]["history"]) == 120
