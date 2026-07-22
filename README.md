# evo-nas

**Quality-diversity neural architecture search on NAS-Bench-201.**

Most NAS returns a single "best" network. `evo-nas` runs **MAP-Elites** instead: it
keeps an *archive* of the best architecture in **every** size-and-depth niche, giving you an
accuracy-vs-cost menu rather than one winner — then grades itself against the true
optimum, which is knowable here because the whole 15,625-architecture space can be
enumerated.

Runs on a laptop CPU in seconds. No training, no GPU, no PyTorch: every architecture's
accuracy is a table lookup into the benchmark.

> **The archive's second axis started as conv-op count** — until using the tool showed that
> was a function of model size, so the map was effectively 1-D and capped at a degenerate 28
> niches no binning could exceed. It is now **cell depth**, a topological measure independent
> of size: 40 genuine niches, verified to still recover the true optimum. The full
> before/after is in [Known limitations](#known-limitations).

## Why this exists

A search algorithm is easy to claim and hard to verify. Point one at a real design space
and you can report what it *found* — but never what it *missed*, because nobody knows the
right answer.

NAS-Bench-201 is small enough to brute-force. That makes it possible to compute the true
best architecture in every niche and grade the search against the **actual optimum**
rather than against another heuristic. That verification is what this repo is for.

The price of that choice is honest and worth stating up front: because the answers are
pre-computed, the architectures this finds have no downstream use. The verified behaviour
of the search is the deliverable, not the designs.

## Results (real NAS-Bench-201, CIFAR-10)

The archive is illuminated over **model size × cell depth** — 40 reachable niches.
Identical evaluation budget (3,000 lookups ≈ 19% of the space), 5 seeds:

| | QD-score (of 30.32 max) | Niche coverage | Best found |
|---|---|---|---|
| **MAP-Elites** | **30.05** ± 0.52 | **98%** (100% on 4/5 seeds) | **94.37%** — the true optimum |
| Random search | 26.28 ± 0.45 | 88% | — |

MAP-Elites recovers the true global optimum in **all five seeds** and reaches 99.1% of the
theoretically achievable QD-score, against random search's 87%. These 40 niches are harder
to fill than a degenerate 28 (see [Known limitations](#known-limitations)) — one seed fell
short of full coverage — but MAP-Elites still beats random search on every seed.

Trade-off from the archive: **93.50%** at 0.40M params → **93.88%** at 0.62M → **94.37%**
at 1.07M. And the largest network in the space (1.53M, all 3×3) scores only **93.76%** —
past a point smaller is better, a call a single-best-architecture result can't help you make.

**The honest floor:** an architecture whose cell is empty — every edge `none` or `skip`,
zero learnable parameters inside the searched component (cell depth 0) — still scores
**86.63%**, because NAS-Bench-201's fixed stem, reduction blocks, and classifier do that
much on their own. So the search's real dynamic range is 86.63% → 94.37%, or **7.74 points**.

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

Full flag reference: [COMMANDS.md](COMMANDS.md).

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

The export runs once and reduces 2.2 GB of benchmark pickles to an 880 KB CSV; after that
`nats_bench` and `gdown` are never needed again. `data/` and `results/` are git-ignored —
regenerate them with the commands above.

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
of **5 operations** — so a design is six integers, and the space is 5⁶ = 15,625. The node
count, wiring, operation set, and surrounding macro-architecture are all fixed by the
benchmark; the search chooses only which operation sits on each edge.

MAP-Elites keeps a grid ("archive") of niches indexed by **model size × cell depth** (the
longest input→output path through the cell), storing the best architecture found in each.
Each iteration it mutates one edge of a random elite, looks the child up, and files it if it
beats that niche's incumbent. Parent selection is uniform over the archive, not biased
toward the best — that is what spreads coverage.

Because the space is enumerable, `ground_truth.py` computes the *exact* best-per-niche
and true Pareto front, so results are measured against the real optimum rather than
another heuristic.

**Honest protocol:** search selects on **validation** accuracy (models trained on the
train split); every reported number is **test** accuracy (models trained on train+valid).
Nothing is ever selected on the test set.

## Known limitations

**Why depth, and why coverage isn't perfect.** An earlier version binned the y-axis on
conv-op count. But model size is exactly `0.073306 + 0.028·n₁ₓ₁ + 0.243040·n₃ₓ₃` — a
function of operation *counts* only, because NAS-Bench-201 cells are channel-uniform — and
conv-count is `n₁ₓ₁ + n₃ₓ₃`, so both axes read off the same two numbers and the archive
capped at a **degenerate 28 niches** that no binning could exceed (FLOPs fails the same
way). The y-axis is now **cell depth**, which depends on *where* the ops sit, not how many,
so it varies independently of size — 40 genuinely-2-D niches. Those niches are harder to
fill: at the current budget MAP-Elites reaches full coverage on 4 of 5 seeds and 88% on the
fifth. Restoring perfect coverage would take a larger budget, not a better descriptor.

**The budget is generous.** 3,000 evaluations is 19% of the space. The regime that matters
for real NAS is 1–2%, and it has not been swept.

**Scope.** CIFAR-10 only (the exporter rejects other datasets); mutation-only, no
crossover; depth counts a skip connection as a hop (graph path length, not learnable-layer
depth); no CMA-ME or CVT-MAP-Elites variants.

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

## References

- Mouret & Clune (2015), *Illuminating search spaces by mapping elites* — the MAP-Elites algorithm.
- Dong & Yang (2020), *NAS-Bench-201* — the benchmark.
- Dong et al. (2021), *NATS-Bench* — the distribution this repo exports from.

## License

MIT — see [LICENSE](LICENSE).
