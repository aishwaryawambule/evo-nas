from evonas.genome import random_genome
from evonas.map_elites import _search

def random_search(benchmark, make_archive, budget, rng, n_reachable):
    return _search(benchmark, make_archive, budget, rng, n_reachable,
                   candidate=lambda archive, evals, rng: random_genome(rng))
