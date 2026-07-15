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

See docs/superpowers/specs/2026-07-14-qd-nas-design.md for the design.
