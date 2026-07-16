import numpy as np
import streamlit as st
from evonas.uidata import load_results, replay_archive, comparison_figures
from evonas.select import select_design
from evonas.archive import CONV_Y_EDGES

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
grid = np.full((len(CONV_Y_EDGES) - 1, results["config"]["map"]["x_bins"]), np.nan)
for (i, j), c in cells.items():
    grid[j, i] = c["val_accuracy"]
st.write(f"cells filled: {len(cells)}")
st.image(np.nan_to_num(grid), caption="val accuracy heatmap", width="stretch", clamp=True)

figures = comparison_figures(results)

st.subheader("MAP-Elites vs random search")
st.pyplot(figures["qd"])
st.pyplot(figures["coverage"])

st.subheader("Discovered vs true frontier")
st.pyplot(figures["frontier"])
