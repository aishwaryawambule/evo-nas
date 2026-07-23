import os

import pandas as pd
import streamlit as st

from evonas.uidata import (load_results, comparison_figures, describe_genome,
                           replay_grid, archive_table)
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
- **Niches** — the archive's two axes: model size (parameters) × cell depth (the longest
  input→output path through the cell). Depth comes from *where* the ops sit, not how many,
  so it varies independently of size — two same-size cells wired differently are kept apart.
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
# slider ranges track the archive's actual span, so no part of their travel is dead
_p = [e["params"] for e in elites]
_a = [e["test_accuracy"] for e in elites]
p_lo, p_hi = float(min(_p)), float(max(_p))
a_lo, a_hi = float(min(_a)), float(max(_a))

mode = st.radio(
    "optimize for", ["best accuracy", "smallest model"], horizontal=True,
    help="Each mode reads the archive from one end: cap the size and take the most "
         "accurate design, or set an accuracy floor and take the cheapest one.",
)
smallest = mode == "smallest model"

# Each mode has exactly one constraint that can bind. Maximising accuracy under an
# accuracy floor is vacuous — the winner clears any floor it possibly can — so that
# slider is disabled rather than left looking live but inert.
max_p = st.slider("max model size (millions of parameters)", p_lo, p_hi, p_hi, 0.01,
                  disabled=smallest)
min_a = st.slider("min test accuracy", a_lo, a_hi, a_lo, 0.001, format="%.3f",
                  disabled=not smallest)
st.caption(
    "Set an **accuracy floor**; the archive returns the cheapest design that clears it."
    if smallest else
    "Set a **size cap**; the archive returns the most accurate design that fits."
)
choice = select_design(elites, max_params=max_p, min_accuracy=min_a, smallest=smallest)

if choice is None:
    st.info("No architecture in the archive satisfies those constraints — loosen a slider.")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("test accuracy", f"{choice['test_accuracy']:.2%}")
    c2.metric("model size", f"{choice['params']:.2f}M params")
    c3.metric("conv ops in cell", choice["conv_count"])
    st.markdown("**The winning cell** — which operation sits on each edge of the cell's DAG:")
    st.table(describe_genome(choice["genome"]))

st.markdown(
    f"**The whole archive — all {len(elites)} designs.** The query above returns one row "
    "(highlighted); the search returned *every* row in a single run. This table is the "
    "quality-diversity result itself — the accuracy-vs-cost menu the sliders read from."
)
_rows, _sel = archive_table(elites, choice["genome"] if choice else None)
_df = pd.DataFrame(_rows)
# format for display (params to 3 dp, accuracy to 2) while the underlying
# values stay numeric so column sorting still works
_styled = _df.style.format({"params (M)": "{:.3f}", "test acc %": "{:.2f}"})
if _sel is not None:
    _styled = _styled.apply(
        lambda row: ["background-color: rgba(255, 75, 75, 0.18)"
                     if row.name == _sel else "" for _ in row],
        axis=1,
    )
st.dataframe(_styled, width="stretch", hide_index=True)
st.caption(
    "Sort by any column. Note that the largest design is **not** the best: filling all six "
    "edges with convs leaves no room for the free `skip` to the output that every top design keeps."
)

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
with st.expander("What exactly differs between the two — and how to read the curves"):
    st.markdown(
        "**The comparison is rigged to be fair.** Both methods share the *same* archive, the "
        "same budget, the same keep-the-better-of-each-niche rule, and the same 5 seeds. The "
        "**only** difference is how each picks its next candidate:\n\n"
        "- **Random search** draws a fresh random genome (six random ops) every single evaluation.\n"
        "- **MAP-Elites** bootstraps with 50 random genomes, then mutates *one edge* of a design "
        "that already survived in the archive.\n\n"
        "So the gap between the curves measures exactly one thing: **directed mutation vs blind "
        "sampling** — nothing else.\n\n"
        "**Reading the curves:** they overlap at the start (empty niches are easy — even random "
        "guessing fills them), then separate once the easy niches are gone. MAP-Elites keeps "
        "climbing because a one-edge mutation of a nearby design often lands in a rare unfilled "
        "niche; random search stalls because dicing your way into a specific hard niche almost "
        "never happens. The bands barely overlapping after the split is what makes the win "
        "*reliable*, not luck."
    )
st.pyplot(figures["qd"])
st.pyplot(figures["coverage"])

st.header("4. How close is it to the true optimum?")
st.caption(
    "Because all 15,625 architectures are enumerable, the **true Pareto front** — the best "
    "accuracy achievable at each model size — is known exactly (black line), not estimated. "
    "Blue dots are MAP-Elites' archive. Where a dot sits *on* the line, the search recovered "
    "the provably optimal design for that size."
)
with st.expander("How to read this plot — dots on, below, and above the line"):
    st.markdown(
        "The front is computed over **all 15,625** designs, independently of the search, so it "
        "is an honest answer key rather than a comparison to another heuristic.\n\n"
        "- **On the line** — the archive holds the true best design at that size (*quality*).\n"
        "- **Below the line** — best in its own (size, depth) niche, but another design is smaller "
        "*and* more accurate, so it is dominated. These are the diversity the archive keeps on "
        "purpose, not search failures — at each size the vertical stack is several depths, and "
        "only the top one is Pareto-optimal.\n"
        "- **The bottom row** (far below, on real CIFAR-10 near 10%) — **depth-0 niches**, cells "
        "whose output node is disconnected. The network is broken; 10% is a random guess over 10 "
        "classes. Quality-diversity maps the dead corner too.\n"
        "- **The line ends at the global optimum** — nothing larger is Pareto-optimal, so dots to "
        "the right of the line's end are *bigger but not better* (dominated by a smaller design).\n"
        "- **A few dots edge just above the line** — the front is chosen on *validation* accuracy "
        "and drawn on *test*; a design with slightly lower val but higher test pokes above. That "
        "overshoot is the fingerprint of an honest split — nothing was ever selected on the test set."
    )
st.pyplot(figures["frontier"])
