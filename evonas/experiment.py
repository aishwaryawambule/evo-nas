import json
import numpy as np
import yaml

from evonas.genome import index_to_genome
from evonas.archive import Archive, param_bin_edges, CONV_Y_EDGES
from evonas.map_elites import map_elites
from evonas.random_search import random_search
from evonas.ground_truth import ground_truth_archive
from evonas.metrics import evaluate, pareto_front

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def build_factory(benchmark, x_bins):
    all_params = [benchmark.query(index_to_genome(i))["params"] for i in range(5 ** 6)]
    edges = param_bin_edges(all_params, x_bins)
    return lambda: Archive(edges, CONV_Y_EDGES)

def _elite_list(archive):
    return [dict(c) for c in archive.elites()]

def run_experiment(config, benchmark):
    x_bins = config["map"]["x_bins"]
    budget = config["budget"]
    init_random = config["init_random"]
    factory = build_factory(benchmark, x_bins)
    # persisted so consumers can label the size axis with real param values
    x_edges = [float(v) for v in factory().x_edges]

    gt = ground_truth_archive(benchmark, factory)
    n_reachable = len(gt.cells)
    pareto = pareto_front(_elite_list(gt))  # gt elites carry genome/params/test_accuracy

    seeds = []
    for seed in config["seeds"]:
        me_archive, me_hist = map_elites(
            benchmark, factory, budget, init_random,
            np.random.default_rng(seed), n_reachable)
        rs_archive, rs_hist = random_search(
            benchmark, factory, budget,
            np.random.default_rng(seed), n_reachable)
        seeds.append({
            "seed": seed,
            "map_elites": {"history": me_hist, "elites": _elite_list(me_archive),
                           "vs_ground_truth": evaluate(me_archive, gt, n_reachable)},
            "random": {"history": rs_hist, "elites": _elite_list(rs_archive),
                       "vs_ground_truth": evaluate(rs_archive, gt, n_reachable)},
        })

    return {
        "config": config,
        "n_reachable": n_reachable,
        "x_edges": x_edges,
        "seeds": seeds,
        "ground_truth": {
            "qd_score": gt.qd_score(),
            "n_cells": len(gt.cells),
            "pareto": [{"params": r["params"], "test_accuracy": r["test_accuracy"],
                        "genome": list(r["genome"])} for r in pareto],
        },
    }

def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj

def to_json(results):
    return json.dumps(_jsonable(results))

def from_json(text):
    return json.loads(text)
