OPS = ("none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3")
CONV_OPS = (2, 3)
NUM_OPS = 5
NUM_EDGES = 6

def random_genome(rng):
    return tuple(int(x) for x in rng.integers(0, NUM_OPS, size=NUM_EDGES))

def genome_to_index(g):
    return sum(op * (NUM_OPS ** i) for i, op in enumerate(g))

def index_to_genome(idx):
    g = []
    for _ in range(NUM_EDGES):
        g.append(idx % NUM_OPS)
        idx //= NUM_OPS
    return tuple(g)

def conv_count(g):
    return sum(1 for op in g if op in CONV_OPS)

def genome_to_arch_str(g):
    node1 = f"|{OPS[g[0]]}~0|"
    node2 = f"|{OPS[g[1]]}~0|{OPS[g[2]]}~1|"
    node3 = f"|{OPS[g[3]]}~0|{OPS[g[4]]}~1|{OPS[g[5]]}~2|"
    return f"{node1}+{node2}+{node3}"
