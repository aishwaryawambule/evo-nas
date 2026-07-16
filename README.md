# evo-nas — Quality-Diversity NAS on NAS-Bench-201

MAP-Elites evolutionary search that maps the best network at every size, verified
against the enumerable ground truth, with a Streamlit UI. CPU-only, no training.

## Quickstart (fake benchmark, no data needed)
    pip install -r requirements-dev.txt
    pytest -q
    python run_search.py --config configs/fake.yaml --out results/fake.json
    streamlit run app.py

## Real NAS-Bench-201 data (CIFAR-10)
    pip install nats_bench gdown          # both are numpy-only; no PyTorch, no GPU
    gdown "https://drive.google.com/uc?id=17_saCsj_krKjlCBLOJEpNtzPXArMCqxU" \
        -O data/NATS-tss-simple.tar       # ~1.1 GB
    tar xf data/NATS-tss-simple.tar -C data/
    python scripts/export_nb201_csv.py \
        --nats data/NATS-tss-v1_0-3ffb9-simple --out data/nb201_cifar10.csv
    python run_search.py --config configs/cifar10.yaml --out results/cifar10.json
    streamlit run app.py                  # auto-prefers results/cifar10.json when present

On real CIFAR-10 data, MAP-Elites reaches QD-score 25.20 / 25.21 (100% niche
coverage) vs random search's 22.11 (88%), and recovers the true global optimum
(94.37% test accuracy) — using ~20% of the 15,625-design space. `data/` and
`results/` are git-ignored; regenerate them with the commands above.

See docs/superpowers/specs/2026-07-14-qd-nas-design.md for the design.
