# Quality-Diversity Neural Architecture Search on NAS-Bench-201

**Design spec — 2026-07-14**

## 1. Summary

A Quality-Diversity search (MAP-Elites) over the NAS-Bench-201 architecture
space that, instead of returning a single "best" network, **illuminates an
archive**: the best-accuracy architecture found for each region of a 2-D
behaviour space (model size × structural complexity). The entire search runs by
**table lookup on a laptop CPU** — no network training. An interactive Streamlit
UI lets a user explore the resulting map, query it for a design under their own
constraints, and watch the search evolve.

The research contribution is a rigorous, self-verifying study: because the whole
NAS-Bench-201 space is only 15,625 architectures, we enumerate it once to obtain
the **ground-truth archive** and **true accuracy-vs-size Pareto front**, then
measure exactly how close MAP-Elites gets — and how far it beats random search —
using only a fraction of the evaluation budget.

## 2. Goals & context

- **Type:** research / novel-angle student & portfolio project.
- **Constraint:** laptop CPU only, no GPU, no network training.
- **Author profile:** strong ML / PyTorch background, newer to evolutionary
  algorithms — so the EA machinery is where the learning and novelty live.
- **Timeframe:** ~2–4 weeks. One clean, novel idea with a solid experiment,
  an interactive demo, and a short paper-style writeup.

### Novelty framing (honest)

This is a *fresh, under-explored combination*, not an unprecedented algorithm:
Quality-Diversity applied to NAS-Bench-201, evaluated against the enumerable
ground truth. That combination — QD illumination + exact ground-truth
verification on a small, reproducible benchmark — is a defensible and tidy
contribution appropriate for the scope.

## 3. Success criteria

1. A working MAP-Elites loop over NAS-Bench-201 (CIFAR-10 primary dataset).
2. A random-search baseline run over many seeds (20–50; runs are instant) for an
   honest, statistically framed comparison.
3. Ground-truth archive + true Pareto front computed by full enumeration.
4. Core figures: archive heatmap filling in over evaluations; QD-score & coverage
   curves vs random search (with confidence bands); discovered-vs-true frontier.
5. An interactive **Streamlit** app: view the map, query a design under
   constraints, replay the search, compare vs random search.
6. A short paper-style writeup (~4–6 pages) framing QD-NAS, method, and results.

### Out of scope (YAGNI)

- Training any real networks; any GPU work.
- CIFAR-100 / ImageNet16-120 (optional stretch only; CIFAR-10 is primary).
- Advanced QD variants (CMA-ME, CVT-MAP-Elites) — vanilla MAP-Elites first.
- Crossover is optional and deferred; mutation-only is the baseline operator.

## 4. Background primitives

### NAS-Bench-201 search space

- A network is built by stacking copies of one **cell**. The cell is a small DAG
  with **6 edges**, each assigned one of **5 operations**:
  `{ none (zeroize), skip_connect, nor_conv_1x1, nor_conv_3x3, avg_pool_3x3 }`.
- Total space = `5^6 = 15,625` architectures — fully enumerable.
- The benchmark provides, per architecture and per dataset: **test/val accuracy**,
  **parameter count**, and **FLOPs**. On CIFAR-10 the best architecture scores
  ~94.4%; the accuracy values are real measured results (all 15,625 were trained
  once by the benchmark authors), so the search introduces **zero scoring error**.

### MAP-Elites

A Quality-Diversity evolutionary algorithm. Maintain a grid ("map") of cells
indexed by behaviour descriptors; each cell keeps the single best-accuracy
("elite") genome found for it. The loop:

1. Seed the map with a handful of random genomes.
2. Repeat for the evaluation budget:
   a. Select a random elite already in the map (the parent).
   b. Mutate it into a child.
   c. Look up the child's accuracy + descriptors from NAS-Bench-201.
   d. Insert: if the child's cell is empty **or** the child beats the incumbent,
      it becomes that cell's elite; otherwise discard.

Same evolutionary machinery as a genetic algorithm (genomes, selection,
mutation) — the QD twist is filing survivors into a grid of niches rather than a
single converging population.

## 5. Technical design

### 5.1 Genome

- Representation: a length-6 integer vector over `{0,1,2,3,4}` (one op per edge).
- `encode`/`decode` between this vector and NAS-Bench-201's arch string.
- `random_genome(rng)` samples each gene uniformly.
- All `5^6` genomes are valid architectures — no invalid/orphan case.

### 5.2 Behaviour descriptors (map axes)

**Default (recommended): model size × structural complexity.**

- **X-axis:** parameter count (model size).
- **Y-axis:** number of learnable (conv) operations in the cell (`conv_count`,
  0–6) — interpretable and gives good spread.

Rationale: params × FLOPs is highly correlated (both are "cost") and spreads
poorly; size × structural-complexity illuminates a more meaningful landscape.
The axes are **configurable** (params×flops and params×depth are alternative
presets) — this default is easy to change and is an open point for spec review.

- Bin edges are derived from the **known min/max over all 15,625 architectures**
  (computed up front), so the grid provably covers every possible design — every
  child maps to exactly one cell.
- Default resolution: 20 (x) × 10 (y) = 200 cells.

### 5.3 Operators

- `mutate(genome, rng)`: change exactly one edge to a different operation
  (single-edge mutation).
- `crossover(a, b, rng)` (optional, deferred): uniform crossover of the 6 genes.

### 5.4 Archive (the map)

- `descriptor_to_cell(params, conv_count) -> (i, j)` via precomputed bin edges.
- `insert(genome, accuracy, descriptors)`: empty-cell → fill; occupied → replace
  only if strictly better. Returns whether an insertion happened.
- Metrics: `coverage` (fraction of reachable cells filled) and `qd_score` (sum of
  elite accuracies across filled cells).
- Holds at most one genome per cell (≤200 total), regardless of how many designs
  were tried.

### 5.5 MAP-Elites runner & random-search baseline

- `map_elites(config) -> (archive, history)`: the loop of §4, logging
  best-so-far accuracy, coverage, and qd_score vs number of evaluations.
- `random_search(config)`: same evaluation budget, genomes drawn uniformly at
  random, filed into an identical archive — the baseline to beat.
- Budget deliberately capped well below 15,625 (default **3,000 ≈ 20%**) so the
  result demonstrates *efficiency*, not brute force. Lookups are cached; repeated
  genomes are free. Repeats count toward the budget (documented choice).

### 5.6 Ground-truth (enumeration)

- `ground_truth(config)`: enumerate all 15,625 architectures once → the true
  best genome per cell (perfect archive) and the true accuracy-vs-size Pareto
  front. Used as the yardstick for coverage, qd_score, and frontier gap.

### 5.7 Benchmark interface

- `Benchmark.query(genome, dataset) -> {accuracy, params, flops, conv_count}`,
  with an in-memory cache.
- Data source: prefer a **precomputed CSV** of accuracies/params/FLOPs (light,
  no PyTorch, no 2 GB download); fall back to the official NAS-Bench-201 /
  NATS-Bench API. Wrapped behind an interface so a tiny deterministic **fake**
  benchmark can back the tests without any data file.

## 6. Experiments & metrics

- **Comparison:** MAP-Elites vs random search, identical budget (3,000),
  20–50 seeds each, CIFAR-10.
- **Metrics over evaluations:** best-so-far accuracy; coverage; qd_score.
- **Against ground truth:** final coverage vs max reachable; qd_score vs perfect;
  discovered frontier gap vs true Pareto front.
- **Reporting:** mean ± std / confidence bands across seeds (MAP-Elites is
  stochastic — variation reported honestly, not hidden).
- **Headline results:** (a) evolution beats random search at illuminating the
  map; (b) MAP-Elites reaches near-ground-truth quality using ~20% of the space.

## 7. UI (Streamlit)

A local browser app (`streamlit run app.py`) reading a saved run:

- **Map view:** interactive archive heatmap (cell colour = elite accuracy); hover
  → genome, accuracy, params, conv_count.
- **Query panel:** sliders for `max params` / `min accuracy`; the matching
  champion cell highlights and its blueprint is shown (the visual `get_design`).
- **Replay:** a slider scrubbing through the 3,000 evaluations to watch the map
  fill in.
- **Compare:** MAP-Elites vs random search side by side; discovered-vs-true
  frontier overlay.

## 8. Code architecture

Small, single-purpose, independently testable modules (Python + numpy + pandas +
matplotlib + streamlit; pytest for tests):

| Module | Responsibility |
|---|---|
| `evonas/benchmark.py` | Load NAS-Bench-201; `query()`; cache; interface + fake |
| `evonas/genome.py` | 6-int representation; encode/decode; random genome |
| `evonas/operators.py` | `mutate()`; optional `crossover()` |
| `evonas/archive.py` | Binning, insert-if-better, coverage, qd_score |
| `evonas/map_elites.py` | MAP-Elites loop + history logging |
| `evonas/random_search.py` | Baseline over the same budget |
| `evonas/ground_truth.py` | Full enumeration → perfect archive & Pareto front |
| `evonas/metrics.py` | qd_score, coverage, best-so-far, frontier gap |
| `evonas/plots.py` | Heatmap, convergence curves, frontier |
| `run_search.py` | CLI: `--config config.yaml --out results/map_42.json` |
| `get_design.py` | CLI: query a saved map by constraint |
| `app.py` | Streamlit UI |
| `tests/` | Unit + smoke tests against the fake benchmark |

### Config (build the map)

```yaml
dataset:   cifar10
budget:    3000
map:
  x_axis:  params
  y_axis:  conv_count
  x_bins:  20
  y_bins:  10
init_random: 50
mutation:  single_edge
seed:      42
```

### Querying a saved map

```
python get_design.py --map results/map_42.json --max-params 1.0M
python get_design.py --map results/map_42.json --min-accuracy 0.90 --smallest
python get_design.py --map results/map_42.json --best
```

## 9. Testing

- `genome`: encode∘decode round-trips; random genomes are valid.
- `operators`: mutation changes exactly one edge; result stays valid.
- `archive`: insert replaces only when strictly better; coverage & qd_score match
  a hand-worked tiny example.
- `map_elites`: end-to-end smoke run on the fake benchmark; deterministic under a
  fixed seed.
- All tests use a tiny deterministic **fake benchmark** — no data file required,
  runs in a blink.

## 10. Risks & mitigations

- **Data acquisition** (API install / large file): mitigated by the CSV-first
  loader and the benchmark interface (dev + tests never need the real file).
- **Descriptor choice affects illumination:** axes are configurable; validated
  against the ground-truth archive; default flagged for review.
- **Reproducibility:** every run seeded; configs logged alongside results.
- **Scope creep:** CIFAR-10 primary; extra datasets and crossover are explicit
  stretch goals, not commitments.

## 11. Milestones (~2–4 weeks)

1. **Week 1 — core loop.** Benchmark loader (+fake), genome, operators, archive,
   MAP-Elites loop, random-search baseline. Unit + smoke tests green.
2. **Week 2 — experiment.** Ground-truth enumeration, metrics, multi-seed runs,
   core plots. First real MAP-Elites-vs-random result.
3. **Week 3 — UI + writeup.** Streamlit app (map / query / replay / compare);
   draft the paper-style report. Polish figures.
4. **Stretch.** Crossover operator; CIFAR-100; alternative descriptor axes;
   ablation on budget and map resolution.
