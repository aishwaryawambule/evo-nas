# QD-NAS (MAP-Elites over NAS-Bench-201) Implementation Plan

**Goal:** Build a Quality-Diversity (MAP-Elites) evolutionary search over the NAS-Bench-201 architecture space that illuminates a map of the best network at every size, verifies itself against the enumerable ground truth, beats a random-search baseline, and ships an interactive Streamlit UI — all CPU-only via table lookup.

**Architecture:** Small single-responsibility Python modules under `evonas/`. A genome is a length-6 integer tuple (one operation per cell edge). A pluggable `Benchmark` returns each genome's accuracy/size by lookup; a deterministic `FakeBenchmark` backs all tests so no data file is needed. The MAP-Elites loop mutates elites and files winners into an `Archive` grid. Enumerating all 15,625 genomes gives ground truth. CLIs run experiments and query saved maps; a Streamlit app visualizes them.

**Tech Stack:** Python 3.11+, numpy, pandas, PyYAML, matplotlib, streamlit, pytest. No GPU, no PyTorch for the core (PyTorch/`nats_bench` only for the one-off data export).

## Global Constraints

- Python **3.11+**. Core runtime deps: `numpy`, `pandas`, `pyyaml`, `matplotlib`, `streamlit`. Dev: `pytest`. `nats_bench`/`torch` used **only** by the optional data-export script, never imported by `evonas/` or tests.
- **CPU only. No network training anywhere.** All architecture scores come from lookup.
- **Genome** = `tuple[int, ...]` of length **6**, each value in `0..4`. Operations, in index order: `OPS = ("none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3")`. Conv ops are indices **2 and 3**.
- **Search selects on `val_accuracy`; only `test_accuracy` is reported.** Never select on the test set.
- All randomness via a passed `numpy.random.Generator` (`np.random.default_rng(seed)`). Every run reproducible from its seed.
- Determinism in tests must not rely on Python's `hash()`; derive any pseudo-values from `genome_to_index`.
- TDD throughout: failing test first, minimal code, commit per task.

---

## File Structure

| Path | Responsibility |
|---|---|
| `evonas/__init__.py` | Package marker |
| `evonas/genome.py` | Ops, random genome, index↔genome, arch-string, conv_count |
| `evonas/operators.py` | `mutate` (single-edge) |
| `evonas/benchmark.py` | `Benchmark` protocol + `FakeBenchmark` |
| `evonas/archive.py` | Bin edges, `Archive` (insert-if-better, coverage, qd_score) |
| `evonas/metrics.py` | `metrics_snapshot`, `evaluate`, `pareto_front` |
| `evonas/map_elites.py` | Shared `_search` loop + `map_elites` |
| `evonas/random_search.py` | `random_search` baseline (reuses `_search`) |
| `evonas/ground_truth.py` | Full enumeration → ground-truth archive, reachable cells |
| `evonas/select.py` | `select_design` (query elites by constraint) |
| `evonas/nb201.py` | `Nb201Benchmark` (CSV loader) |
| `evonas/plots.py` | Heatmap, convergence, frontier figures |
| `evonas/experiment.py` | `load_config`, `run_experiment`, JSON (de)serialization |
| `run_search.py` | CLI: build a map from a config, save JSON |
| `get_design.py` | CLI: query a saved map by constraint |
| `app.py` | Streamlit UI |
| `scripts/export_nb201_csv.py` | One-off: export NAS-Bench-201 to CSV (optional) |
| `tests/` | One test module per `evonas/` module |
| `requirements.txt`, `requirements-dev.txt`, `pyproject.toml`, `.gitignore` | Project setup |

---

## Task 0: Project scaffolding

**Files:**
- Create: `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`, `.gitignore`, `evonas/__init__.py`, `tests/__init__.py`, `tests/test_smoke.py`

**Interfaces:**
- Produces: importable package `evonas` with `evonas.__version__`.

- [ ] **Step 1: Write the failing test**

`tests/test_smoke.py`:
```python
def test_package_imports():
    import evonas
    assert evonas.__version__ == "0.1.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_smoke.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evonas'`

- [ ] **Step 3: Create the package and setup files**

`evonas/__init__.py`:
```python
__version__ = "0.1.0"
```

`tests/__init__.py`: (empty file)

`requirements.txt`:
```
numpy>=1.26
pandas>=2.1
pyyaml>=6.0
matplotlib>=3.8
streamlit>=1.30
```

`requirements-dev.txt`:
```
-r requirements.txt
pytest>=8.0
```

`pyproject.toml`:
```toml
[project]
name = "evonas"
version = "0.1.0"
requires-python = ">=3.11"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`.gitignore`:
```
__pycache__/
*.pyc
.venv/
venv/
results/
data/
*.egg-info/
.pytest_cache/
```

- [ ] **Step 4: Install deps and run the test**

Run: `python -m pip install -r requirements-dev.txt && python -m pytest tests/test_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml requirements.txt requirements-dev.txt .gitignore evonas tests
git commit -m "chore: project scaffolding and smoke test"
```

---

## Task 1: Genome

**Files:**
- Create: `evonas/genome.py`, `tests/test_genome.py`

**Interfaces:**
- Produces:
  - `OPS: tuple[str, ...]` (len 5), `CONV_OPS = (2, 3)`, `NUM_EDGES = 6`, `NUM_OPS = 5`
  - `random_genome(rng) -> tuple[int, ...]`
  - `genome_to_index(g) -> int` (0..15624)
  - `index_to_genome(idx) -> tuple[int, ...]`
  - `genome_to_arch_str(g) -> str`
  - `conv_count(g) -> int` (0..6)

- [ ] **Step 1: Write the failing tests**

`tests/test_genome.py`:
```python
import numpy as np
from evonas.genome import (
    OPS, NUM_EDGES, random_genome, genome_to_index,
    index_to_genome, genome_to_arch_str, conv_count,
)

def test_ops_are_the_five_nb201_operations():
    assert OPS == ("none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3")

def test_random_genome_has_six_valid_edges():
    rng = np.random.default_rng(0)
    g = random_genome(rng)
    assert len(g) == NUM_EDGES
    assert all(0 <= op <= 4 for op in g)

def test_index_round_trips_over_whole_space():
    for idx in (0, 1, 4, 5, 3125, 15624):
        assert genome_to_index(index_to_genome(idx)) == idx

def test_index_covers_exactly_15625():
    seen = {genome_to_index(index_to_genome(i)) for i in range(5**6)}
    assert seen == set(range(5**6))

def test_conv_count_counts_only_conv_ops():
    assert conv_count((2, 3, 0, 1, 4, 2)) == 3  # ops 2,3,2 are conv
    assert conv_count((0, 1, 4, 0, 1, 4)) == 0

def test_arch_str_matches_nb201_format():
    g = (0, 0, 0, 0, 0, 0)
    assert genome_to_arch_str(g) == "|none~0|+|none~0|none~1|+|none~0|none~1|none~2|"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_genome.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evonas.genome'`

- [ ] **Step 3: Implement `evonas/genome.py`**

```python
OPS = ("none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3")
CONV_OPS = (2, 3)
NUM_OPS = 5
NUM_EDGES = 6

def random_genome(rng):
    return tuple(int(x) for x in rng.integers(0, NUM_OPS, size=NUM_EDGES))

def genome_to_index(g):
    return sum(op * (NUM_OPS ** i) for i, op in enumerate(g))

def index_to_genome(idx):
    g = []
    for _ in range(NUM_EDGES):
        g.append(idx % NUM_OPS)
        idx //= NUM_OPS
    return tuple(g)

def conv_count(g):
    return sum(1 for op in g if op in CONV_OPS)

def genome_to_arch_str(g):
    node1 = f"|{OPS[g[0]]}~0|"
    node2 = f"|{OPS[g[1]]}~0|{OPS[g[2]]}~1|"
    node3 = f"|{OPS[g[3]]}~0|{OPS[g[4]]}~1|{OPS[g[5]]}~2|"
    return f"{node1}+{node2}+{node3}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_genome.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add evonas/genome.py tests/test_genome.py
git commit -m "feat: genome representation for NAS-Bench-201 cell"
```

---

## Task 2: Mutation operator

**Files:**
- Create: `evonas/operators.py`, `tests/test_operators.py`

**Interfaces:**
- Consumes: `evonas.genome.NUM_OPS`, `NUM_EDGES`
- Produces: `mutate(g, rng) -> tuple[int, ...]` — changes exactly one edge to a *different* operation.

- [ ] **Step 1: Write the failing tests**

`tests/test_operators.py`:
```python
import numpy as np
from evonas.operators import mutate

def test_mutate_changes_exactly_one_edge():
    rng = np.random.default_rng(1)
    parent = (0, 0, 0, 0, 0, 0)
    for _ in range(50):
        child = mutate(parent, rng)
        diffs = [i for i in range(6) if child[i] != parent[i]]
        assert len(diffs) == 1

def test_mutated_edge_takes_a_different_value():
    rng = np.random.default_rng(2)
    parent = (1, 2, 3, 4, 0, 1)
    for _ in range(50):
        child = mutate(parent, rng)
        i = next(i for i in range(6) if child[i] != parent[i])
        assert child[i] != parent[i]
        assert 0 <= child[i] <= 4

def test_mutate_returns_length_six_tuple():
    rng = np.random.default_rng(3)
    child = mutate((0, 1, 2, 3, 4, 0), rng)
    assert isinstance(child, tuple) and len(child) == 6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_operators.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evonas.operators'`

- [ ] **Step 3: Implement `evonas/operators.py`**

```python
from evonas.genome import NUM_OPS, NUM_EDGES

def mutate(g, rng):
    edge = int(rng.integers(0, NUM_EDGES))
    # pick a different op in 0..NUM_OPS-1 without rejection sampling
    new_op = int(rng.integers(0, NUM_OPS - 1))
    if new_op >= g[edge]:
        new_op += 1
    child = list(g)
    child[edge] = new_op
    return tuple(child)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_operators.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add evonas/operators.py tests/test_operators.py
git commit -m "feat: single-edge mutation operator"
```

---

## Task 3: Benchmark interface + FakeBenchmark

**Files:**
- Create: `evonas/benchmark.py`, `tests/test_benchmark.py`

**Interfaces:**
- Consumes: `evonas.genome.genome_to_index`, `conv_count`
- Produces:
  - A record is a `dict` with keys: `val_accuracy, test_accuracy, params, flops, conv_count` (all floats except `conv_count` int).
  - `FakeBenchmark().query(g) -> dict` — deterministic, cached, accuracy correlated with conv count. Used by all downstream tests.

- [ ] **Step 1: Write the failing tests**

`tests/test_benchmark.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_benchmark.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evonas.benchmark'`

- [ ] **Step 3: Implement `evonas/benchmark.py`**

```python
from typing import Protocol
from evonas.genome import genome_to_index, conv_count

class Benchmark(Protocol):
    def query(self, g) -> dict: ...

class FakeBenchmark:
    """Deterministic stand-in for NAS-Bench-201 used in tests and demos.

    Accuracy rises with conv ops and falls with 'none' ops; a tiny
    index-derived jitter breaks ties. No randomness, no data file.
    """
    OP_PARAMS = (0.0, 0.0, 0.10, 0.30, 0.0)  # per-op params in millions

    def __init__(self):
        self._cache = {}

    def query(self, g):
        if g in self._cache:
            return self._cache[g]
        cc = conv_count(g)
        n_none = sum(1 for op in g if op == 0)
        params = round(sum(self.OP_PARAMS[op] for op in g), 6)
        flops = round(params * 12.0, 6)
        jitter = (genome_to_index(g) % 97) / 10000.0  # 0..0.0096
        val = 0.50 + 0.06 * cc - 0.015 * n_none + jitter
        val = round(max(0.10, min(0.95, val)), 6)
        rec = {
            "val_accuracy": val,
            "test_accuracy": round(val - 0.008, 6),
            "params": params,
            "flops": flops,
            "conv_count": cc,
        }
        self._cache[g] = rec
        return rec
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_benchmark.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add evonas/benchmark.py tests/test_benchmark.py
git commit -m "feat: benchmark protocol and deterministic FakeBenchmark"
```

---

## Task 4: Archive (the map)

**Files:**
- Create: `evonas/archive.py`, `tests/test_archive.py`

**Interfaces:**
- Consumes: record dicts from Task 3
- Produces:
  - `param_bin_edges(all_params, x_bins) -> np.ndarray` (length `x_bins+1`)
  - `CONV_Y_EDGES: np.ndarray` = edges for conv_count 0..6 (7 bins)
  - `Archive(x_edges, y_edges)` with:
    - `.cells: dict[tuple[int,int], dict]` — each value is `{"genome", ...record}`
    - `cell_index(params, conv_count) -> tuple[int,int]`
    - `insert(genome, record) -> bool` (True if it became/replaced the elite)
    - `random_elite(rng) -> tuple` (genome)
    - `coverage(n_reachable) -> float`
    - `qd_score() -> float` (sum of elite `val_accuracy`)
    - `elites() -> list[dict]` (each includes `"genome"`)

**Note on resolution:** conv_count has only 7 possible values (0..6), so the Y axis uses **7 discrete bins**, not the illustrative "10" in the spec. Default map = `x_bins=20` × 7 = **140 cells**. This is the intended reconciliation.

- [ ] **Step 1: Write the failing tests**

`tests/test_archive.py`:
```python
import numpy as np
from evonas.archive import Archive, param_bin_edges, CONV_Y_EDGES

def make_archive():
    x = param_bin_edges([0.0, 1.8], x_bins=4)  # params range 0..1.8
    return Archive(x, CONV_Y_EDGES)

def rec(val, params, cc):
    return {"val_accuracy": val, "test_accuracy": val - 0.01,
            "params": params, "flops": params * 12, "conv_count": cc}

def test_conv_y_edges_give_seven_bins():
    assert len(CONV_Y_EDGES) == 8  # 7 bins => 8 edges

def test_insert_into_empty_cell_returns_true():
    a = make_archive()
    assert a.insert((0,0,0,0,0,0), rec(0.8, 0.0, 0)) is True
    assert len(a.cells) == 1

def test_better_val_replaces_incumbent():
    a = make_archive()
    a.insert((0,0,0,0,0,0), rec(0.80, 0.0, 0))
    assert a.insert((1,0,0,0,0,0), rec(0.90, 0.0, 0)) is True  # same cell, better
    assert len(a.cells) == 1
    assert a.elites()[0]["genome"] == (1,0,0,0,0,0)

def test_worse_or_equal_val_is_rejected():
    a = make_archive()
    a.insert((0,0,0,0,0,0), rec(0.90, 0.0, 0))
    assert a.insert((1,0,0,0,0,0), rec(0.90, 0.0, 0)) is False  # equal
    assert a.insert((1,0,0,0,0,0), rec(0.80, 0.0, 0)) is False  # worse

def test_qd_score_sums_elite_val_accuracy():
    a = make_archive()
    a.insert((0,0,0,0,0,0), rec(0.80, 0.0, 0))
    a.insert((3,3,0,0,0,0), rec(0.70, 0.6, 2))  # different conv_count => different cell
    assert abs(a.qd_score() - 1.50) < 1e-9

def test_coverage_is_filled_over_reachable():
    a = make_archive()
    a.insert((0,0,0,0,0,0), rec(0.80, 0.0, 0))
    assert a.coverage(4) == 0.25

def test_random_elite_returns_a_stored_genome():
    a = make_archive()
    a.insert((0,0,0,0,0,0), rec(0.80, 0.0, 0))
    assert a.random_elite(np.random.default_rng(0)) == (0,0,0,0,0,0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_archive.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evonas.archive'`

- [ ] **Step 3: Implement `evonas/archive.py`**

```python
import numpy as np

CONV_Y_EDGES = np.arange(-0.5, 7.5, 1.0)  # 7 bins for conv_count 0..6

def param_bin_edges(all_params, x_bins):
    lo, hi = float(min(all_params)), float(max(all_params))
    if hi <= lo:
        hi = lo + 1e-9
    return np.linspace(lo, hi, x_bins + 1)

def _bin(value, edges):
    i = int(np.digitize([value], edges)[0]) - 1
    return int(np.clip(i, 0, len(edges) - 2))

class Archive:
    def __init__(self, x_edges, y_edges):
        self.x_edges = np.asarray(x_edges, dtype=float)
        self.y_edges = np.asarray(y_edges, dtype=float)
        self.cells = {}

    def cell_index(self, params, conv_count):
        return (_bin(params, self.x_edges), _bin(conv_count, self.y_edges))

    def insert(self, genome, record):
        key = self.cell_index(record["params"], record["conv_count"])
        cur = self.cells.get(key)
        if cur is None or record["val_accuracy"] > cur["val_accuracy"]:
            self.cells[key] = {"genome": genome, **record}
            return True
        return False

    def random_elite(self, rng):
        keys = list(self.cells.keys())
        return self.cells[keys[int(rng.integers(len(keys)))]]["genome"]

    def coverage(self, n_reachable):
        return len(self.cells) / n_reachable

    def qd_score(self):
        return sum(c["val_accuracy"] for c in self.cells.values())

    def elites(self):
        return list(self.cells.values())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_archive.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add evonas/archive.py tests/test_archive.py
git commit -m "feat: MAP-Elites archive with insert-if-better and metrics"
```

---

## Task 5: Metrics (snapshot, evaluate, pareto front)

**Files:**
- Create: `evonas/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Consumes: `Archive` from Task 4
- Produces:
  - `metrics_snapshot(archive, eval_index, best_val, n_reachable) -> dict` with keys `eval, coverage, qd_score, best_val`
  - `evaluate(archive, gt_archive, n_reachable) -> dict` with keys `coverage_ratio, qd_ratio`
  - `pareto_front(records) -> list[dict]` — minimize `params`, maximize `test_accuracy`

- [ ] **Step 1: Write the failing tests**

`tests/test_metrics.py`:
```python
from evonas.archive import Archive, param_bin_edges, CONV_Y_EDGES
from evonas.metrics import metrics_snapshot, evaluate, pareto_front

def make_archive():
    return Archive(param_bin_edges([0.0, 1.8], 4), CONV_Y_EDGES)

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evonas.metrics'`

- [ ] **Step 3: Implement `evonas/metrics.py`**

```python
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
        "coverage_ratio": len(archive.cells) / n_reachable,
        "qd_ratio": (archive.qd_score() / gt_qd) if gt_qd else 0.0,
    }

def pareto_front(records):
    ordered = sorted(records, key=lambda r: (r["params"], -r["test_accuracy"]))
    front, best = [], float("-inf")
    for r in ordered:
        if r["test_accuracy"] > best:
            front.append(r)
            best = r["test_accuracy"]
    return front
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add evonas/metrics.py tests/test_metrics.py
git commit -m "feat: snapshot, ground-truth evaluation, and pareto front metrics"
```

---

## Task 6: MAP-Elites loop (with shared `_search`)

**Files:**
- Create: `evonas/map_elites.py`, `tests/test_map_elites.py`

**Interfaces:**
- Consumes: `random_genome`, `mutate`, `metrics_snapshot`, a `benchmark`, a `make_archive` zero-arg factory, `n_reachable: int`
- Produces:
  - `_search(benchmark, make_archive, budget, rng, n_reachable, candidate) -> tuple[Archive, list[dict]]`
    where `candidate(archive, eval_index, rng) -> genome`
  - `map_elites(benchmark, make_archive, budget, init_random, rng, n_reachable) -> tuple[Archive, list[dict]]`

- [ ] **Step 1: Write the failing tests**

`tests/test_map_elites.py`:
```python
import numpy as np
from evonas.benchmark import FakeBenchmark
from evonas.archive import Archive, param_bin_edges, CONV_Y_EDGES
from evonas.genome import index_to_genome
from evonas.map_elites import map_elites

def make_factory():
    all_params = [FakeBenchmark().query(index_to_genome(i))["params"] for i in range(5**6)]
    edges = param_bin_edges(all_params, x_bins=20)
    return lambda: Archive(edges, CONV_Y_EDGES)

def test_history_length_equals_budget():
    _, history = map_elites(FakeBenchmark(), make_factory(), budget=200,
                            init_random=20, rng=np.random.default_rng(0), n_reachable=140)
    assert len(history) == 200

def test_best_val_is_monotonic_nondecreasing():
    _, history = map_elites(FakeBenchmark(), make_factory(), budget=300,
                            init_random=20, rng=np.random.default_rng(1), n_reachable=140)
    bests = [h["best_val"] for h in history]
    assert all(b2 >= b1 for b1, b2 in zip(bests, bests[1:]))

def test_run_is_reproducible_under_same_seed():
    f = make_factory()
    a1, _ = map_elites(FakeBenchmark(), f, 200, 20, np.random.default_rng(7), 140)
    a2, _ = map_elites(FakeBenchmark(), f, 200, 20, np.random.default_rng(7), 140)
    assert sorted(a1.cells) == sorted(a2.cells)
    assert a1.qd_score() == a2.qd_score()

def test_coverage_grows_beyond_initial_seed():
    archive, history = map_elites(FakeBenchmark(), make_factory(), budget=500,
                                  init_random=20, rng=np.random.default_rng(2), n_reachable=140)
    assert len(archive.cells) > 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_map_elites.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evonas.map_elites'`

- [ ] **Step 3: Implement `evonas/map_elites.py`**

```python
from evonas.genome import random_genome
from evonas.operators import mutate
from evonas.metrics import metrics_snapshot

def _search(benchmark, make_archive, budget, rng, n_reachable, candidate):
    archive = make_archive()
    history = []
    best = 0.0
    for evals in range(1, budget + 1):
        g = candidate(archive, evals, rng)
        rec = benchmark.query(g)
        archive.insert(g, rec)
        best = max(best, rec["val_accuracy"])
        history.append(metrics_snapshot(archive, evals, best, n_reachable))
    return archive, history

def map_elites(benchmark, make_archive, budget, init_random, rng, n_reachable):
    def candidate(archive, evals, rng):
        if evals <= init_random or not archive.cells:
            return random_genome(rng)
        return mutate(archive.random_elite(rng), rng)
    return _search(benchmark, make_archive, budget, rng, n_reachable, candidate)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_map_elites.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add evonas/map_elites.py tests/test_map_elites.py
git commit -m "feat: MAP-Elites search loop over the archive"
```

---

## Task 7: Random-search baseline

**Files:**
- Create: `evonas/random_search.py`, `tests/test_random_search.py`

**Interfaces:**
- Consumes: `_search` from Task 6, `random_genome`
- Produces: `random_search(benchmark, make_archive, budget, rng, n_reachable) -> tuple[Archive, list[dict]]`

- [ ] **Step 1: Write the failing tests**

`tests/test_random_search.py`:
```python
import numpy as np
from evonas.benchmark import FakeBenchmark
from evonas.archive import Archive, param_bin_edges, CONV_Y_EDGES
from evonas.genome import index_to_genome
from evonas.random_search import random_search

def make_factory():
    all_params = [FakeBenchmark().query(index_to_genome(i))["params"] for i in range(5**6)]
    edges = param_bin_edges(all_params, x_bins=20)
    return lambda: Archive(edges, CONV_Y_EDGES)

def test_history_length_equals_budget():
    _, history = random_search(FakeBenchmark(), make_factory(), budget=200,
                               rng=np.random.default_rng(0), n_reachable=140)
    assert len(history) == 200

def test_reproducible_under_same_seed():
    f = make_factory()
    a1, _ = random_search(FakeBenchmark(), f, 200, np.random.default_rng(3), 140)
    a2, _ = random_search(FakeBenchmark(), f, 200, np.random.default_rng(3), 140)
    assert a1.qd_score() == a2.qd_score()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_random_search.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evonas.random_search'`

- [ ] **Step 3: Implement `evonas/random_search.py`**

```python
from evonas.genome import random_genome
from evonas.map_elites import _search

def random_search(benchmark, make_archive, budget, rng, n_reachable):
    return _search(benchmark, make_archive, budget, rng, n_reachable,
                   candidate=lambda archive, evals, rng: random_genome(rng))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_random_search.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add evonas/random_search.py tests/test_random_search.py
git commit -m "feat: random-search baseline reusing the shared search loop"
```

---

## Task 8: Ground truth (full enumeration)

**Files:**
- Create: `evonas/ground_truth.py`, `tests/test_ground_truth.py`

**Interfaces:**
- Consumes: `index_to_genome`, a `benchmark`, a `make_archive` factory
- Produces:
  - `enumerate_all(benchmark) -> Iterator[tuple[genome, record]]` over all 15,625
  - `ground_truth_archive(benchmark, make_archive) -> Archive`
  - `reachable_cells(benchmark, make_archive) -> int`

- [ ] **Step 1: Write the failing tests**

`tests/test_ground_truth.py`:
```python
from evonas.benchmark import FakeBenchmark
from evonas.archive import Archive, param_bin_edges, CONV_Y_EDGES
from evonas.genome import index_to_genome
from evonas.ground_truth import enumerate_all, ground_truth_archive, reachable_cells

def make_factory():
    all_params = [FakeBenchmark().query(index_to_genome(i))["params"] for i in range(5**6)]
    edges = param_bin_edges(all_params, x_bins=20)
    return lambda: Archive(edges, CONV_Y_EDGES)

def test_enumerate_visits_whole_space():
    assert sum(1 for _ in enumerate_all(FakeBenchmark())) == 15625

def test_ground_truth_archive_holds_best_per_cell():
    gt = ground_truth_archive(FakeBenchmark(), make_factory())
    # every filled cell's elite is the max val over genomes mapping to that cell
    assert len(gt.cells) == reachable_cells(FakeBenchmark(), make_factory())

def test_reachable_cells_is_positive_and_bounded():
    n = reachable_cells(FakeBenchmark(), make_factory())
    assert 0 < n <= 140
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_ground_truth.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evonas.ground_truth'`

- [ ] **Step 3: Implement `evonas/ground_truth.py`**

```python
from evonas.genome import index_to_genome

def enumerate_all(benchmark):
    for idx in range(5 ** 6):
        g = index_to_genome(idx)
        yield g, benchmark.query(g)

def ground_truth_archive(benchmark, make_archive):
    archive = make_archive()
    for g, rec in enumerate_all(benchmark):
        archive.insert(g, rec)
    return archive

def reachable_cells(benchmark, make_archive):
    return len(ground_truth_archive(benchmark, make_archive).cells)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ground_truth.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add evonas/ground_truth.py tests/test_ground_truth.py
git commit -m "feat: ground-truth enumeration and reachable-cell count"
```

---

## Task 9: Design selection (query a map)

**Files:**
- Create: `evonas/select.py`, `tests/test_select.py`

**Interfaces:**
- Consumes: a list of elite records (each has `params`, `test_accuracy`, `genome`, ...)
- Produces: `select_design(elites, max_params=None, min_accuracy=None, smallest=False, best=False) -> dict | None`

- [ ] **Step 1: Write the failing tests**

`tests/test_select.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_select.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evonas.select'`

- [ ] **Step 3: Implement `evonas/select.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_select.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add evonas/select.py tests/test_select.py
git commit -m "feat: query a map for a design under size/accuracy constraints"
```

---

## Task 10: Experiment orchestration + JSON serialization

**Files:**
- Create: `evonas/experiment.py`, `tests/test_experiment.py`

**Interfaces:**
- Consumes: everything above; a config `dict`
- Produces:
  - `load_config(path) -> dict`
  - `build_factory(benchmark, x_bins) -> callable` (computes param edges over the whole space)
  - `run_experiment(config, benchmark) -> dict` (results object; see shape below)
  - `to_json(results) -> str` / `from_json(text) -> dict` (genomes ↔ lists)

**Results shape (exact keys):**
```python
{
  "config": {...},                       # echoed config
  "n_reachable": int,
  "seeds": [
     {"seed": int,
      "map_elites": {"history": [...], "elites": [ {genome:list, ...record}, ... ]},
      "random":     {"history": [...], "elites": [ ... ]}}
  ],
  "ground_truth": {"qd_score": float, "n_cells": int,
                   "pareto": [ {params, test_accuracy, genome:list}, ... ]}
}
```

- [ ] **Step 1: Write the failing tests**

`tests/test_experiment.py`:
```python
from evonas.benchmark import FakeBenchmark
from evonas.experiment import build_factory, run_experiment, to_json, from_json

CONFIG = {
    "dataset": "fake",
    "budget": 150,
    "map": {"x_bins": 20},
    "init_random": 20,
    "seeds": [0, 1],
}

def test_run_produces_results_for_each_seed():
    res = run_experiment(CONFIG, FakeBenchmark())
    assert len(res["seeds"]) == 2
    assert res["n_reachable"] > 0
    assert res["ground_truth"]["n_cells"] == res["n_reachable"]

def test_each_seed_has_both_methods_with_full_history():
    res = run_experiment(CONFIG, FakeBenchmark())
    s0 = res["seeds"][0]
    assert len(s0["map_elites"]["history"]) == 150
    assert len(s0["random"]["history"]) == 150
    assert len(s0["map_elites"]["elites"]) > 0

def test_json_round_trip_preserves_values():
    res = run_experiment(CONFIG, FakeBenchmark())
    back = from_json(to_json(res))
    assert back["n_reachable"] == res["n_reachable"]
    # genomes come back as lists
    assert isinstance(back["seeds"][0]["map_elites"]["elites"][0]["genome"], list)

def test_build_factory_uses_whole_space_for_edges():
    factory = build_factory(FakeBenchmark(), x_bins=20)
    archive = factory()
    assert len(archive.x_edges) == 21
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_experiment.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evonas.experiment'`

- [ ] **Step 3: Implement `evonas/experiment.py`**

```python
import json
import numpy as np
import yaml

from evonas.genome import index_to_genome
from evonas.archive import Archive, param_bin_edges, CONV_Y_EDGES
from evonas.map_elites import map_elites
from evonas.random_search import random_search
from evonas.ground_truth import ground_truth_archive
from evonas.metrics import pareto_front

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def build_factory(benchmark, x_bins):
    all_params = [benchmark.query(index_to_genome(i))["params"] for i in range(5 ** 6)]
    edges = param_bin_edges(all_params, x_bins)
    return lambda: Archive(edges, CONV_Y_EDGES)

def _elite_list(archive):
    return [dict(c) for c in archive.elites()]

def run_experiment(config, benchmark):
    x_bins = config["map"]["x_bins"]
    budget = config["budget"]
    init_random = config["init_random"]
    factory = build_factory(benchmark, x_bins)

    gt = ground_truth_archive(benchmark, factory)
    n_reachable = len(gt.cells)
    pareto = pareto_front(_elite_list(gt))  # gt elites carry genome/params/test_accuracy

    seeds = []
    for seed in config["seeds"]:
        me_archive, me_hist = map_elites(
            benchmark, factory, budget, init_random,
            np.random.default_rng(seed), n_reachable)
        rs_archive, rs_hist = random_search(
            benchmark, factory, budget,
            np.random.default_rng(seed), n_reachable)
        seeds.append({
            "seed": seed,
            "map_elites": {"history": me_hist, "elites": _elite_list(me_archive)},
            "random": {"history": rs_hist, "elites": _elite_list(rs_archive)},
        })

    return {
        "config": config,
        "n_reachable": n_reachable,
        "seeds": seeds,
        "ground_truth": {
            "qd_score": gt.qd_score(),
            "n_cells": len(gt.cells),
            "pareto": [{"params": r["params"], "test_accuracy": r["test_accuracy"],
                        "genome": list(r["genome"])} for r in pareto],
        },
    }

def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj

def to_json(results):
    return json.dumps(_jsonable(results))

def from_json(text):
    return json.loads(text)
```

Note: `Archive.insert` already stores `"genome"` on each cell, so `_elite_list(gt)` yields records with `params`, `test_accuracy`, and `genome` — exactly what `pareto_front` and the ground-truth output need.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_experiment.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add evonas/experiment.py tests/test_experiment.py
git commit -m "feat: experiment orchestration with multi-seed runs and JSON I/O"
```

---

## Task 11: `run_search.py` CLI

**Files:**
- Create: `run_search.py`, `configs/fake.yaml`, `tests/test_run_search_cli.py`

**Interfaces:**
- Consumes: `load_config`, `run_experiment`, `to_json`, `FakeBenchmark`, `Nb201Benchmark` (Task 12; guard import)
- Produces: CLI `python run_search.py --config <path> --out <path>` writing a results JSON.

- [ ] **Step 1: Write the failing test**

`tests/test_run_search_cli.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_run_search_cli.py -v`
Expected: FAIL — `run_search.py` not found / non-zero exit

- [ ] **Step 3: Implement `run_search.py` and `configs/fake.yaml`**

`configs/fake.yaml`:
```yaml
dataset: fake
budget: 3000
map:
  x_bins: 20
init_random: 50
seeds: [0, 1, 2, 3, 4]
```

`run_search.py`:
```python
import argparse
from evonas.experiment import load_config, run_experiment, to_json
from evonas.benchmark import FakeBenchmark

def make_benchmark(config):
    if config["dataset"] == "fake":
        return FakeBenchmark()
    from evonas.nb201 import Nb201Benchmark  # imported lazily
    return Nb201Benchmark(config["data_csv"], dataset=config["dataset"])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    config = load_config(args.config)
    results = run_experiment(config, make_benchmark(config))
    with open(args.out, "w") as f:
        f.write(to_json(results))
    print(f"wrote {args.out}: {results['n_reachable']} reachable cells")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_run_search_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add run_search.py configs/fake.yaml tests/test_run_search_cli.py
git commit -m "feat: run_search CLI to build and save a map"
```

---

## Task 12: NAS-Bench-201 CSV benchmark + data export

**Files:**
- Create: `evonas/nb201.py`, `tests/test_nb201.py`, `scripts/export_nb201_csv.py`

**Interfaces:**
- Consumes: `genome_to_index`, `conv_count`
- Produces: `Nb201Benchmark(csv_path, dataset="cifar10").query(g) -> dict` (same record keys as `FakeBenchmark`).
- CSV schema (columns): `index, val_accuracy, test_accuracy, params, flops` — one row per architecture index `0..15624`, accuracies as fractions in `[0,1]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_nb201.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_nb201.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evonas.nb201'`

- [ ] **Step 3: Implement `evonas/nb201.py` and `scripts/export_nb201_csv.py`**

`evonas/nb201.py`:
```python
import pandas as pd
from evonas.genome import genome_to_index, conv_count

class Nb201Benchmark:
    def __init__(self, csv_path, dataset="cifar10"):
        self.dataset = dataset
        df = pd.read_csv(csv_path)
        self._rows = {int(r["index"]): r for _, r in df.iterrows()}

    def query(self, g):
        r = self._rows[genome_to_index(g)]
        return {
            "val_accuracy": float(r["val_accuracy"]),
            "test_accuracy": float(r["test_accuracy"]),
            "params": float(r["params"]),
            "flops": float(r["flops"]),
            "conv_count": conv_count(g),
        }
```

`scripts/export_nb201_csv.py` (run once, needs the NB201 data file + `nats_bench`; NOT imported by the package/tests):
```python
"""Export NAS-Bench-201 to the CSV schema used by Nb201Benchmark.

Usage:
    pip install nats_bench
    # download NATS-tss-v1_0-3ffb9.pickle.pbz2 per the NATS-Bench README
    python scripts/export_nb201_csv.py --nats <path-to-pickle> --out data/nb201_cifar10.csv

Accuracy fields for cifar10: 'x-valid' (val) and 'ori-test' (test), scaled to [0,1].
Architecture index i maps to our genome via index_to_genome(i); both use the
same edge ordering (node1<-0, node2<-0, node2<-1, node3<-0, node3<-1, node3<-2).
"""
import argparse, csv
from nats_bench import create
from evonas.genome import index_to_genome, genome_to_arch_str

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nats", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--dataset", default="cifar10")
    args = p.parse_args()
    api = create(args.nats, "tss", fast_mode=True, verbose=False)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "val_accuracy", "test_accuracy", "params", "flops"])
        for idx in range(len(api)):
            # sanity: our genome's arch string must match the API's
            assert api.arch(idx) == genome_to_arch_str(index_to_genome(idx)), idx
            info = api.get_more_info(idx, args.dataset, hp="200")
            cost = api.get_cost_info(idx, args.dataset)
            w.writerow([idx,
                        info["valid-accuracy"] / 100.0,
                        info["test-accuracy"] / 100.0,
                        cost["params"], cost["flops"]])
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
```

**If `api.arch(idx)` does not match `genome_to_arch_str(index_to_genome(idx))`**, adjust `index_to_genome`'s edge ordering in Task 1 (not the export) until the assertion passes for all indices — the export script is the source of truth for ordering.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_nb201.py -v`
Expected: PASS (2 tests). (The export script is not unit-tested; it needs the real data file.)

- [ ] **Step 5: Commit**

```bash
git add evonas/nb201.py tests/test_nb201.py scripts/export_nb201_csv.py
git commit -m "feat: NAS-Bench-201 CSV benchmark and one-off export script"
```

---

## Task 13: Plots

**Files:**
- Create: `evonas/plots.py`, `tests/test_plots.py`

**Interfaces:**
- Consumes: an `Archive`, per-seed histories, pareto lists
- Produces (each returns a `matplotlib.figure.Figure`):
  - `heatmap_figure(archive) -> Figure`
  - `convergence_figure(me_histories, rs_histories, metric="qd_score") -> Figure`
  - `frontier_figure(discovered_elites, true_pareto) -> Figure`

- [ ] **Step 1: Write the failing tests**

`tests/test_plots.py`:
```python
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import numpy as np
from evonas.benchmark import FakeBenchmark
from evonas.experiment import build_factory
from evonas.map_elites import map_elites
from evonas.plots import heatmap_figure, convergence_figure, frontier_figure

def _run():
    f = build_factory(FakeBenchmark(), 20)
    return map_elites(FakeBenchmark(), f, 200, 20, np.random.default_rng(0), 140)

def test_heatmap_returns_figure():
    archive, _ = _run()
    assert isinstance(heatmap_figure(archive), Figure)

def test_convergence_returns_figure():
    _, h = _run()
    assert isinstance(convergence_figure([h], [h], metric="qd_score"), Figure)

def test_frontier_returns_figure():
    archive, _ = _run()
    pareto = [{"params": e["params"], "test_accuracy": e["test_accuracy"]} for e in archive.elites()]
    assert isinstance(frontier_figure(archive.elites(), pareto), Figure)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_plots.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evonas.plots'`

- [ ] **Step 3: Implement `evonas/plots.py`**

```python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def heatmap_figure(archive):
    nx = len(archive.x_edges) - 1
    ny = len(archive.y_edges) - 1
    grid = np.full((ny, nx), np.nan)
    for (i, j), cell in archive.cells.items():
        grid[j, i] = cell["val_accuracy"]
    fig, ax = plt.subplots()
    im = ax.imshow(grid, origin="lower", aspect="auto")
    ax.set_xlabel("model size bin (params)")
    ax.set_ylabel("conv count")
    fig.colorbar(im, ax=ax, label="val accuracy")
    return fig

def convergence_figure(me_histories, rs_histories, metric="qd_score"):
    fig, ax = plt.subplots()
    for label, hists in (("MAP-Elites", me_histories), ("random", rs_histories)):
        arr = np.array([[h[metric] for h in hist] for hist in hists])
        mean = arr.mean(axis=0)
        x = np.arange(1, len(mean) + 1)
        ax.plot(x, mean, label=label)
        if arr.shape[0] > 1:
            sd = arr.std(axis=0)
            ax.fill_between(x, mean - sd, mean + sd, alpha=0.2)
    ax.set_xlabel("evaluations")
    ax.set_ylabel(metric)
    ax.legend()
    return fig

def frontier_figure(discovered_elites, true_pareto):
    fig, ax = plt.subplots()
    d = sorted(discovered_elites, key=lambda e: e["params"])
    ax.scatter([e["params"] for e in d], [e["test_accuracy"] for e in d],
               s=12, label="discovered elites")
    t = sorted(true_pareto, key=lambda e: e["params"])
    ax.plot([e["params"] for e in t], [e["test_accuracy"] for e in t],
            color="black", label="true pareto front")
    ax.set_xlabel("params")
    ax.set_ylabel("test accuracy")
    ax.legend()
    return fig
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_plots.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add evonas/plots.py tests/test_plots.py
git commit -m "feat: heatmap, convergence, and frontier figures"
```

---

## Task 14: `get_design.py` CLI

**Files:**
- Create: `get_design.py`, `tests/test_get_design_cli.py`

**Interfaces:**
- Consumes: `from_json`, `select_design`
- Produces: CLI `python get_design.py --map <json> [--seed N] [--max-params X] [--min-accuracy Y] [--smallest] [--best]` printing the chosen design.

- [ ] **Step 1: Write the failing test**

`tests/test_get_design_cli.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_get_design_cli.py -v`
Expected: FAIL — `get_design.py` not found

- [ ] **Step 3: Implement `get_design.py`**

```python
import argparse
from evonas.experiment import from_json
from evonas.select import select_design

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--map", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-params", type=float, default=None)
    p.add_argument("--min-accuracy", type=float, default=None)
    p.add_argument("--smallest", action="store_true")
    p.add_argument("--best", action="store_true")
    args = p.parse_args()

    with open(args.map) as f:
        results = from_json(f.read())
    seeds = results["seeds"]
    entry = next((s for s in seeds if s["seed"] == args.seed), seeds[0])
    elites = entry["map_elites"]["elites"]

    d = select_design(elites, max_params=args.max_params,
                      min_accuracy=args.min_accuracy,
                      smallest=args.smallest, best=args.best)
    if d is None:
        print("No design matches those constraints.")
        return
    print(f"genome:   {d['genome']}")
    print(f"test acc: {d['test_accuracy']:.4f}")
    print(f"params:   {d['params']}")
    print(f"conv:     {d['conv_count']}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_get_design_cli.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add get_design.py tests/test_get_design_cli.py
git commit -m "feat: get_design CLI to query a saved map"
```

---

## Task 15: Streamlit UI

**Files:**
- Create: `app.py`, `evonas/uidata.py`, `tests/test_uidata.py`

**Interfaces:**
- Consumes: `from_json`, `select_design`, `plots`
- Produces:
  - `evonas/uidata.py`: `load_results(path) -> dict`; `replay_archive(history_elites, up_to)` helper for the replay slider (pure, testable). To support replay, the experiment must log **insertions**; add that below.
  - `app.py`: Streamlit page (manual run; not unit-tested beyond import).

**Add insertion log for replay:** In `evonas/map_elites.py` `_search`, record inserts. Change the snapshot append to also capture inserted cells:

```python
inserted = archive.insert(g, rec)
...
snap = metrics_snapshot(archive, evals, best, n_reachable)
snap["insert"] = {"cell": list(archive.cell_index(rec["params"], rec["conv_count"])),
                  "val_accuracy": rec["val_accuracy"], "genome": list(g)} if inserted else None
history.append(snap)
```
Update `tests/test_map_elites.py` `test_history_length_equals_budget` still holds (length unchanged). Add a test that at least one snapshot has a non-None `insert`.

- [ ] **Step 1: Write the failing tests**

`tests/test_uidata.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_uidata.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'evonas.uidata'`

- [ ] **Step 3: Implement `evonas/uidata.py` and `app.py`**

`evonas/uidata.py`:
```python
from evonas.experiment import from_json

def load_results(path):
    with open(path) as f:
        return from_json(f.read())

def replay_archive(history, up_to):
    cells = {}
    for snap in history[:up_to]:
        ins = snap.get("insert")
        if ins is None:
            continue
        key = tuple(ins["cell"])
        cur = cells.get(key)
        if cur is None or ins["val_accuracy"] > cur["val_accuracy"]:
            cells[key] = {"val_accuracy": ins["val_accuracy"], "genome": ins["genome"]}
    return cells
```

`app.py`:
```python
import numpy as np
import streamlit as st
from evonas.uidata import load_results, replay_archive
from evonas.select import select_design

st.title("QD-NAS: MAP-Elites over NAS-Bench-201")

path = st.text_input("Results JSON path", "results/fake.json")
try:
    results = load_results(path)
except FileNotFoundError:
    st.warning("Run `python run_search.py --config configs/fake.yaml --out results/fake.json` first.")
    st.stop()

seed_entry = results["seeds"][0]
history = seed_entry["map_elites"]["history"]
elites = seed_entry["map_elites"]["elites"]

st.subheader("Query a design")
max_p = st.slider("max params", 0.0, 2.0, 2.0, 0.05)
min_a = st.slider("min test accuracy", 0.0, 1.0, 0.0, 0.01)
choice = select_design(elites, max_params=max_p, min_accuracy=min_a)
st.write(choice if choice else "No design matches those constraints.")

st.subheader("Replay the search")
step = st.slider("evaluations", 1, len(history), len(history))
cells = replay_archive(history, up_to=step)
grid = np.full((7, results["config"]["map"]["x_bins"]), np.nan)
for (i, j), c in cells.items():
    grid[j, i] = c["val_accuracy"]
st.write(f"cells filled: {len(cells)}")
st.image(np.nan_to_num(grid), caption="val accuracy heatmap", use_column_width=True, clamp=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_uidata.py -v`
Expected: PASS. Then manually: `python run_search.py --config configs/fake.yaml --out results/fake.json && streamlit run app.py` and confirm the page loads, sliders update the design, and the replay slider fills the heatmap.

- [ ] **Step 5: Commit**

```bash
git add app.py evonas/uidata.py evonas/map_elites.py tests/test_uidata.py tests/test_map_elites.py
git commit -m "feat: Streamlit UI with design query and search replay"
```

---

## Task 16: Full suite + demo run + README

**Files:**
- Create: `README.md`
- Modify: none (verification task)

- [ ] **Step 1: Run the entire test suite**

Run: `python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 2: Do a real fake-data experiment run**

Run: `python run_search.py --config configs/fake.yaml --out results/fake.json`
Expected: prints reachable-cell count; `results/fake.json` exists.

- [ ] **Step 3: Sanity-check MAP-Elites beats random on the fake benchmark**

Run:
```bash
python -c "import json; d=json.load(open('results/fake.json')); s=d['seeds'][0]; print('ME qd', s['map_elites']['history'][-1]['qd_score']); print('RS qd', s['random']['history'][-1]['qd_score'])"
```
Expected: MAP-Elites final `qd_score` ≥ random-search final `qd_score` (they should be close on the easy fake benchmark; the real gap shows on NB201).

- [ ] **Step 4: Write `README.md`**

```markdown
# evo-nas — Quality-Diversity NAS on NAS-Bench-201

MAP-Elites evolutionary search that maps the best network at every size, verified
against the enumerable ground truth, with a Streamlit UI. CPU-only, no training.

## Quickstart (fake benchmark, no data needed)
    pip install -r requirements-dev.txt
    pytest -q
    python run_search.py --config configs/fake.yaml --out results/fake.json
    streamlit run app.py

## Real NAS-Bench-201 data
    pip install nats_bench   # + download the NATS-tss pickle (see scripts/export_nb201_csv.py)
    python scripts/export_nb201_csv.py --nats <pickle> --out data/nb201_cifar10.csv
    # then set `dataset: cifar10` and `data_csv: data/nb201_cifar10.csv` in a config

See docs/specs/qd-nas-design.md for the design.
```

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: README with quickstart and full-suite verification"
```

---

## Self-Review notes (addressed)

- **Spec coverage:** genome/ops (T1–2), benchmark + fake + real (T3, T12), archive/map (T4), MAP-Elites (T6) + random baseline (T7), ground truth & reachable cells (T8), metrics/qd/coverage/frontier (T5, T13), val-vs-test protocol (records carry both; search uses `val_accuracy`, reporting/frontier uses `test_accuracy`), multi-seed experiment + JSON (T10–11), design query (T9, T14), Streamlit UI with query + replay + compare (T13, T15), milestones map to task order.
- **Descriptor axes:** default params × conv_count implemented; `x_bins` configurable; conv_count fixed at 7 discrete bins (documented deviation from the illustrative "10"). Alternative axes are a stretch, not in these tasks.
- **Type consistency:** record keys (`val_accuracy, test_accuracy, params, flops, conv_count`) identical across `FakeBenchmark`, `Nb201Benchmark`, and `Archive`; `map_elites`/`random_search` share `_search` with the same `(archive, history)` return; `select_design` signature identical in `get_design.py` and `app.py`.
- **No placeholders:** every step has runnable code/tests.
