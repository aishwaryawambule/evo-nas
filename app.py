import os

import streamlit as st

from evonas.uidata import (load_results, comparison_figures, describe_genome,
                           replay_grid)
from evonas.select import select_design
from evonas.plots import grid_heatmap_figure

st.title("QD-NAS — quality-diversity architecture search on NAS-Bench-201")

st.markdown("""
Conventional NAS returns **one** architecture. This runs **MAP-Elites**, a *quality-diversity*
algorithm: it partitions the space into **niches** and keeps the best architecture found in
each, so the result is an **archive** — the best design at *every* size/complexity, an
accuracy-vs-cost menu instead of a single winner.

- **Search space** — the NAS-Bench-201 cell: 6 edges × 5 candidate ops = **15,625** architectures.
- **Evaluation is free** — every architecture's accuracy is pre-computed in the benchmark, so
  search is a table lookup. No training; runs on a laptop CPU.
- **Niches** — the archive's two axes: model size (parameters) × number of conv ops in the cell.
- **Protocol** — search selects on **validation** accuracy; every number reported here is
  **test** accuracy, so nothing is selected on the test set.
- **Why this space** — it's small enough to enumerate all 15,625, so the *true* best-per-niche
  and *true* Pareto front are known exactly. The search can be graded against the real optimum
  rather than against another heuristic.
""")

# prefer real NAS-Bench-201 results if present, else the no-download fake run
_default = next((p for p in ("results/cifar10.json", "results/fake.json")
                 if os.path.exists(p)), "results/cifar10.json")
path = st.text_input("Results JSON path", _default)
if os.path.basename(path) == "cifar10.json":
    st.caption("Showing **real NAS-Bench-201 / CIFAR-10** data — actual trained accuracies.")
else:
    st.caption("Showing the **synthetic FakeBenchmark** run (no data download needed).")
try:
    results = load_results(path)
except FileNotFoundError:
    st.warning("No results at that path. Generate one first:\n\n"
               "- real data: `python run_search.py --config configs/cifar10.yaml --out results/cifar10.json`\n"
               "- no download: `python run_search.py --config configs/fake.yaml --out results/fake.json`")
    st.stop()

seed_entry = results["seeds"][0]
history = seed_entry["map_elites"]["history"]
elites = seed_entry["map_elites"]["elites"]

st.header("1. Query the archive")
st.caption(
    "The archive doubles as a lookup table over deployment constraints: state a budget, "
    "read off the best architecture that satisfies it. This is what a quality-diversity "
    "result buys you over a single-best-architecture result."
)
max_p = st.slider("max model size (millions of parameters)", 0.0, 2.0, 2.0, 0.05)
min_a = st.slider("min test accuracy", 0.0, 1.0, 0.0, 0.01)
choice = select_design(elites, max_params=max_p, min_accuracy=min_a)

if choice is None:
    st.info("No architecture in the archive satisfies those constraints — loosen a slider.")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("test accuracy", f"{choice['test_accuracy']:.2%}")
    c2.metric("model size", f"{choice['params']:.2f}M params")
    c3.metric("conv ops in cell", choice["conv_count"])
    st.markdown("**The winning cell** — which operation sits on each edge of the cell's DAG:")
    st.table(describe_genome(choice["genome"]))

st.header("2. Watch the archive fill in")
st.caption(
    "One tile = one niche. Colour = the best validation accuracy found in that niche so far; "
    "**blank = not yet discovered**. Drag the slider to replay the search: MAP-Elites mutates "
    "an existing elite, and the child only earns a tile if it beats that niche's incumbent."
)
step = st.slider("architectures evaluated", 1, len(history), len(history))
grid, n_filled = replay_grid(results, history, step)
st.pyplot(grid_heatmap_figure(grid, results.get("x_edges")))
st.caption(f"**{n_filled} of {results['n_reachable']} reachable niches filled** after "
           f"{step:,} evaluations "
           f"({step / len(history):.0%} of this run's budget; the full space is 15,625).")

figures = comparison_figures(results)

st.header("3. Does evolution actually beat random search?")
st.caption(
    "Both methods get the identical evaluation budget; bands are ±1 SD across seeds. "
    "**QD-score** = the sum of best accuracies over all filled niches — it rewards filling "
    "*more* niches *and* filling them *better*, so it is the headline quality-diversity metric. "
    "**Coverage** = the fraction of reachable niches filled. MAP-Elites sitting above random "
    "means evolution maps the trade-off space both faster and more completely."
)
st.pyplot(figures["qd"])
st.pyplot(figures["coverage"])

st.header("4. How close is it to the true optimum?")
st.caption(
    "Because all 15,625 architectures are enumerable, the true Pareto front is known exactly "
    "(black line) — not estimated. Dots are the archive's elites. The vertical gap between the "
    "dots and the line is the search's actual suboptimality, measured rather than assumed."
)
st.pyplot(figures["frontier"])
