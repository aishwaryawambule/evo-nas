# evo-nas

**Quality-diversity neural architecture search on NAS-Bench-201.**

Most NAS returns a single "best" network. `evo-nas` runs **MAP-Elites** instead: it
keeps an *archive* of the best architecture at **every** model size, giving you an
accuracy-vs-cost menu rather than one winner — then grades itself against the true
optimum, which is knowable here because the whole 15,625-architecture space can be
enumerated.

Runs on a laptop CPU. No training, no GPU: every architecture's accuracy is a table
lookup into the benchmark.

## Results (real NAS-Bench-201, CIFAR-10)

Identical evaluation budget (3,000 lookups ≈ 20% of the space), 5 seeds:

| | QD-score (of 25.21 max) | Niche coverage | Best found |
|---|---|---|---|
| **MAP-Elites** | **25.20** ± 0.00 | **100%** | **94.37%** — the true optimum |
| Random search | 22.11 ± 0.45 | 88% | — |

The archive's real trade-off curve: **93.54%** at 0.43M params → **94.31%** at 0.64M →
**94.37%** at 1.07M. That last 0.06% of accuracy costs 67% more parameters — the kind of
call a single-best-architecture result can't help you make.

## Install

```bash
pip install evonas            # library + evonas-search / evonas-design
pip install "evonas[ui]"      # + the Streamlit explorer
```

From a clone:

```bash
pip install ".[ui,dev]"
pytest -q
```

## Quickstart — no data download

A built-in synthetic benchmark (`FakeBenchmark`) makes the whole pipeline runnable
before you download anything:

```bash
evonas-search --config configs/fake.yaml --out results/fake.json
evonas-design --map results/fake.json --max-params 1.0
streamlit run app.py
```

## Real NAS-Bench-201 data (CIFAR-10)

```bash
pip install "evonas[data]"     # nats_bench + gdown; both numpy-only, no PyTorch
gdown "https://drive.google.com/uc?id=17_saCsj_krKjlCBLOJEpNtzPXArMCqxU" \
    -O data/NATS-tss-simple.tar          # ~1.1 GB, once
tar xf data/NATS-tss-simple.tar -C data/
python scripts/export_nb201_csv.py \
    --nats data/NATS-tss-v1_0-3ffb9-simple --out data/nb201_cifar10.csv
evonas-search --config configs/cifar10.yaml --out results/cifar10.json
streamlit run app.py                     # auto-prefers results/cifar10.json
```

`data/` and `results/` are git-ignored — regenerate them with the commands above.

## Use it as a library

```python
import numpy as np
from evonas.benchmark import FakeBenchmark          # or Nb201Benchmark for real data
from evonas.experiment import build_factory
from evonas.map_elites import map_elites
from evonas.ground_truth import reachable_cells
from evonas.select import select_design

bench = FakeBenchmark()
factory = build_factory(bench, x_bins=20)
n_reachable = reachable_cells(bench, factory)

archive, history = map_elites(bench, factory, budget=3000, init_random=50,
                              rng=np.random.default_rng(0), n_reachable=n_reachable)

# best architecture that fits in 1M parameters
print(select_design(archive.elites(), max_params=1.0))
```

## How it works

An architecture is the NAS-Bench-201 cell: a 4-node DAG whose **6 edges** each take one
of **5 operations** — so a design is six integers, and the space is 5⁶ = 15,625.

MAP-Elites keeps a grid ("archive") of niches indexed by **model size × conv-op count**,
storing the best architecture found in each. Each iteration it mutates one edge of a
random elite, looks the child up, and files it if it beats that niche's incumbent.

Because the space is enumerable, `ground_truth.py` computes the *exact* best-per-niche
and true Pareto front, so results are measured against the real optimum rather than
another heuristic.

**Honest protocol:** search selects on **validation** accuracy (models trained on the
train split); every reported number is **test** accuracy (models trained on train+valid).
Nothing is ever selected on the test set.

## Layout

| Module | Responsibility |
|---|---|
| `genome.py` | architecture ↔ 6 ints ↔ index ↔ NAS-Bench arch string |
| `operators.py` | single-edge mutation |
| `benchmark.py` / `nb201.py` | synthetic scores / real NAS-Bench-201 CSV |
| `archive.py` | the MAP-Elites grid (insert-if-better, coverage, QD-score) |
| `map_elites.py` / `random_search.py` | the search loop and its baseline |
| `ground_truth.py` | full enumeration → true optimum |
| `metrics.py` | QD-score, coverage, Pareto front |
| `select.py` | query an archive under size/accuracy constraints |
| `plots.py` / `uidata.py` / `app.py` | figures and the Streamlit explorer |

Design notes: [`docs/superpowers/specs/`](docs/superpowers/specs/).

## License

MIT — see [LICENSE](LICENSE).
