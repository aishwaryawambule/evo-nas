from evonas.genome import random_genome
from evonas.operators import mutate
from evonas.metrics import metrics_snapshot

def _search(benchmark, make_archive, budget, rng, n_reachable, candidate):
    archive = make_archive()
    history = []
    best = 0.0
    for evals in range(1, budget + 1):
        g = candidate(archive, evals, rng)
        rec = benchmark.query(g)
        archive.insert(g, rec)
        best = max(best, rec["val_accuracy"])
        history.append(metrics_snapshot(archive, evals, best, n_reachable))
    return archive, history

def map_elites(benchmark, make_archive, budget, init_random, rng, n_reachable):
    def candidate(archive, evals, rng):
        if evals <= init_random or not archive.cells:
            return random_genome(rng)
        return mutate(archive.random_elite(rng), rng)
    return _search(benchmark, make_archive, budget, rng, n_reachable, candidate)
