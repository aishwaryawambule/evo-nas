from evonas.genome import NUM_OPS, NUM_EDGES

def mutate(g, rng):
    edge = int(rng.integers(0, NUM_EDGES))
    # pick a different op in 0..NUM_OPS-1 without rejection sampling
    new_op = int(rng.integers(0, NUM_OPS - 1))
    if new_op >= g[edge]:
        new_op += 1
    child = list(g)
    child[edge] = new_op
    return tuple(child)
