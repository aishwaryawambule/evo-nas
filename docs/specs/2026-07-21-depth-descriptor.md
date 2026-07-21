# Replace the conv-count descriptor with cell depth

**Design spec — 2026-07-21**

## 1. Problem

The archive's two behaviour descriptors are not independent. Model size is exactly

```
params = 0.073306 + 0.028000·n₁ₓ₁ + 0.243040·n₃ₓ₃
```

a function of operation *counts* only (NAS-Bench-201 cells are channel-uniform, so an
operation costs the same on any edge). Conv-count is `n₁ₓ₁ + n₃ₓ₃`. Both axes therefore
read off the same two numbers, so the y-axis carries no information the x-axis doesn't
already have. Consequences:

- The archive caps at **28 reachable niches** regardless of `x_bins` — no binning choice
  can exceed it, because the coordinate is fully determined by `(n₁ₓ₁, n₃ₓ₃)` and there
  are only `C(8,2) = 28` such pairs.
- Two cells of identical size but different structure collide in one niche; one evicts
  the other. The archive cannot hold structurally distinct designs of the same cost — the
  exact thing quality-diversity is supposed to preserve.

The map is nominally 2-D but effectively 1-D.

## 2. Goal

Replace the conv-count y-axis with **cell depth** — a topological descriptor that varies
at fixed size — so the archive becomes genuinely 2-D and holds structurally distinct
designs. Non-goals: hyperparameter tuning (no headroom left at 99.94% of ground truth),
new QD variants, a third descriptor.

## 3. Design

### Depth definition

`cell_depth(genome)` = the length of the longest path from the input node (node0) to the
output node (node3), over edges that are not `none`, counting each edge as one hop.

- Range 0–3. **depth 0** means node3 is unreachable from node0 (the cell is disconnected;
  341 architectures, the ~86.6% accuracy floor). **depth 3** is a full serial chain.
- **Decision, open to review:** a `skip_connect` is a present edge, so it counts as a hop.
  This makes depth measure *graph* path length (how many connections the signal traverses),
  not *computational* depth (how many learnable layers). This is the clean graph-theoretic
  definition and the one behind the niche counts below. The alternative — skip is "free,"
  only convolutions add depth — is defensible but fuzzier and is not chosen here.

### Where it is computed

`Archive.insert(genome, record)` already receives the genome, and depth is a pure function
of the genome. So the archive computes it there — the benchmarks (`FakeBenchmark`,
`Nb201Benchmark`) are **not** touched, and no CSV re-export is needed. Depth is stored on
the elite record so consumers can display it.

`conv_count` is retained everywhere it is *displayed* (the query result, the archive
table). It simply stops being the axis the archive bins on.

### Files

| file | change |
|---|---|
| `genome.py` | add `cell_depth(g)` and the edge adjacency it needs; keep `conv_count` |
| `archive.py` | `CONV_Y_EDGES` → `DEPTH_Y_EDGES = arange(-0.5, 4.5, 1.0)` (4 bins); `insert` computes depth from the genome, bins on it, and stores `depth` on the elite |
| `experiment.py` | import/pass `DEPTH_Y_EDGES` instead of `CONV_Y_EDGES` |
| `uidata.py` | replay grid rows = `len(DEPTH_Y_EDGES) - 1` (4); y-axis semantics = depth |
| `plots.py` | heatmap y-axis label → "cell depth (longest input→output path)" |
| `app.py` | intro bullet and section-2 caption: "conv ops" axis → "cell depth"; the per-design `conv ops` metric and archive-table column stay |
| tests | row-count assertions 7 → 4; add `cell_depth` unit tests; update any conv-count-as-axis assertions |
| `README.md` / `COMMANDS.md` | rewrite the degeneracy limitation; refresh result numbers from the regenerated run |

## 4. Expected outcome

With the archive's 20 param bins, `(param_bin, depth)` yields **40 reachable niches**
(up from 28), and depth splits **9 of the 16 occupied param bins** — i.e. the axis is
genuinely independent, not a restatement of size. The niche-count increase is the symptom;
the independence is the substance.

## 5. Verification gate (mandatory)

The change is only worth it if the search still works on the harder map. After
implementing, regenerate `results/cifar10.json` and `results/fake.json` and confirm:

1. **Ground truth** now spans 40 niches (`n_reachable == 40`).
2. **MAP-Elites still reaches near-full coverage and beats random search** across the 40
   niches at the current budget. 40 niches are harder to fill than 28, so this may require
   raising `budget` or `init_random`. If MAP-Elites no longer clearly beats random search,
   that is itself a finding, and the design is not "done" until it is understood.
3. All README/COMMANDS numbers are updated from the *actual* regenerated run, not predicted.

Full test suite green throughout.

## 6. Out of scope

- `skip_count` as the descriptor (46 niches) — considered, not chosen; depth is the more
  intuitive primary axis.
- A third descriptor / 3-D archive — would break the 2-D heatmap and require generalizing
  `cell_index`/`insert` to N descriptors. YAGNI for a benchmark demo.
- Re-exporting the NAS-Bench-201 CSV — depth is derived from the genome, so the CSV schema
  is unchanged.
