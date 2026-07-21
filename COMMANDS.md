# Commands

Every example below is a real invocation with its actual output.

Two console scripts are installed with the package:

| command | purpose |
|---|---|
| [`evonas-search`](#evonas-search) | run the search, write an archive to JSON |
| [`evonas-design`](#evonas-design) | query a saved archive under a constraint |

Plus a one-off [data export script](#scripts-export_nb201_csvpy) and the
[Streamlit explorer](#streamlit-run-apppy).

---

## `evonas-search`

Builds the archive. Reads a YAML config, runs MAP-Elites and random search at an
identical budget for every seed, writes one JSON with both histories, both sets of
elites, and the enumerated ground truth.

```
usage: evonas-search [-h] --config CONFIG --out OUT
```

| flag | required | description |
|---|---|---|
| `--config` | yes | path to a YAML config (see [Config](#config)) |
| `--out` | yes | where to write the results JSON |

```bash
evonas-search --config configs/fake.yaml --out results/fake.json
```
```
wrote results/fake.json: 54 reachable cells
```

The printed cell count is how many niches any architecture can actually occupy — **40**
for real NAS-Bench-201 (54 for the synthetic benchmark, whose param values differ), out of
a nominal 20×4 = 80 grid. The rest are geometrically impossible, not merely undiscovered.

Real data (requires the [CSV export](#scripts-export_nb201_csvpy) first):

```bash
evonas-search --config configs/cifar10.yaml --out results/cifar10.json
```

Runtime is a few seconds: 5 seeds × 3,000 evaluations × 2 methods = 30,000 CSV lookups.

### Config

```yaml
dataset: cifar10                    # "fake" selects the built-in synthetic benchmark
data_csv: data/nb201_cifar10.csv    # ignored when dataset is "fake"
budget: 3000                        # evaluations per method, per seed
map:
  x_bins: 20                        # model-size bins on the archive's x-axis
init_random: 50                     # random evaluations before mutation begins
seeds: [0, 1, 2, 3, 4]              # one full run per seed
```

Notes on the knobs:

- **`budget`** is the one that matters. 3,000 is 19% of the 15,625-architecture space.
- **`x_bins`** sets the model-size resolution of the archive's x-axis. Finer bins split
  more designs apart (up to the 28 distinct param values); coarser bins merge them.
- The y-axis is **cell depth** and is not configurable — it is fixed at 4 bins (depth 0–3).

---

## `evonas-design`

Queries a saved archive. This is the archive's payoff: search once, query as many
times as you have constraints.

```
usage: evonas-design [-h] --map MAP [--seed SEED] [--max-params MAX_PARAMS]
                     [--min-accuracy MIN_ACCURACY] [--smallest | --best]
```

| flag | default | description |
|---|---|---|
| `--map` | *required* | results JSON from `evonas-search` |
| `--seed` | first seed in the file | which seed's archive to query |
| `--max-params` | none | only consider designs at or below this size (millions) |
| `--min-accuracy` | none | only consider designs at or above this test accuracy |
| `--smallest` | | return the smallest design meeting the constraints |
| `--best` | (default) | return the most accurate design meeting the constraints |

`--smallest` and `--best` are mutually exclusive.

### Most accurate design overall

```bash
evonas-design --map results/cifar10.json --best
```
```
genome:   [3, 3, 3, 1, 3, 2]
test acc: 0.9437
params:   1.073466
conv:     5
```

This is the true global optimum of the space, confirmed by enumeration.

### Best design that fits a size budget

```bash
evonas-design --map results/cifar10.json --max-params 0.5
```
```
genome:   [3, 2, 2, 1, 2, 2]
test acc: 0.9354
params:   0.428346
conv:     5
```

### Cheapest design that clears an accuracy bar

```bash
evonas-design --map results/cifar10.json --smallest --min-accuracy 0.93
```
```
genome:   [1, 0, 3, 0, 1, 2]
test acc: 0.9329
params:   0.344346
conv:     2
```

`--min-accuracy` only changes the answer in `--smallest` mode. Under `--best` the
winner already clears any floor it possibly can, so the flag can only leave the
result unchanged or empty the archive.

### Query a different seed

```bash
evonas-design --map results/cifar10.json --seed 3 --best
```
```
genome:   [3, 3, 3, 1, 3, 2]
test acc: 0.9437
params:   1.073466
conv:     5
```

All five seeds recover the same optimum.

### Reading the output

`genome` is six integers, one per cell edge, in this order:

| position | edge | |
|---|---|---|
| 0 | node1 ← node0 | |
| 1, 2 | node2 ← node0, node1 | |
| 3, 4, 5 | node3 ← node0, node1, node2 | |

Each integer selects an operation: `0` none (edge deleted), `1` skip connection,
`2` conv 1×1, `3` conv 3×3, `4` avg-pool 3×3. So `[3, 2, 2, 1, 2, 2]` is a conv 3×3
from node0 to node1, conv 1×1 on both edges into node2, a skip connection straight
from input to output, and conv 1×1 on the remaining two.

`params` is in millions and follows exactly
`0.073306 + 0.028000·(#conv1×1) + 0.243040·(#conv3×3)` — only the two convolutions
carry weights.

### Error cases

No design satisfies the constraints:

```bash
evonas-design --map results/cifar10.json --min-accuracy 0.99
```
```
No design matches those constraints.
```

A seed that is not in the file — the available ones are named:

```bash
evonas-design --map results/cifar10.json --seed 99 --best
```
```
seed 99 is not in results/cifar10.json (available: 0, 1, 2, 3, 4)
```

Both mode flags at once:

```bash
evonas-design --map results/cifar10.json --best --smallest
```
```
evonas-design: error: argument --smallest: not allowed with argument --best
```

---

## `scripts/export_nb201_csv.py`

One-off. Converts the NATS-Bench distribution into the 880 KB CSV the benchmark
reads. Needs `pip install "evonas[data]"` (`nats_bench` + `gdown`, both numpy-only).

```
usage: export_nb201_csv.py [-h] --nats NATS --out OUT [--dataset DATASET]
```

| flag | default | description |
|---|---|---|
| `--nats` | *required* | path to the untarred NATS-tss "simple" directory |
| `--out` | *required* | destination CSV |
| `--dataset` | `cifar10` | only `cifar10` is supported |

```bash
gdown "https://drive.google.com/uc?id=17_saCsj_krKjlCBLOJEpNtzPXArMCqxU" \
    -O data/NATS-tss-simple.tar          # ~1.1 GB
tar xf data/NATS-tss-simple.tar -C data/
python scripts/export_nb201_csv.py \
    --nats data/NATS-tss-v1_0-3ffb9-simple --out data/nb201_cifar10.csv
```

Run this once. Afterwards `nats_bench` and `gdown` are never needed again — the
search only reads the CSV.

The CSV is keyed by this project's base-5 genome index, **not** NATS-Bench's internal
index; the two differ, so each genome is resolved via its architecture string.

---

## `streamlit run app.py`

The explorer. Prefers `results/cifar10.json`, falls back to `results/fake.json`, and
labels which one is on screen.

```bash
streamlit run app.py                     # http://localhost:8501
```

Four sections: query the archive (with the full 40-design table), replay the search
filling niches in, MAP-Elites vs random search convergence, and the archive against
the exactly-known Pareto front.

Requires the UI extra: `pip install "evonas[ui]"`.

---

## Tests

```bash
pip install ".[ui,dev]"
pytest -q
```
```
68 passed
```

CI runs the suite on Python 3.11, 3.12, and 3.13, then does an end-to-end smoke run of
both console scripts.

### A note on `pip install -e` and Python 3.14

On Python 3.14, setuptools' editable install writes a `.pth` finder that is not
executed at interpreter startup, so the console scripts fail with
`ModuleNotFoundError: No module named 'evonas'` even though the package is listed as
installed. A regular (non-editable) install works correctly on 3.14. If you want an
editable checkout, use Python 3.11–3.13.
