import numpy as np

OPS = ("none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3")
CONV_OPS = (2, 3)
NUM_OPS = 5
NUM_EDGES = 6
NUM_ARCHS = NUM_OPS ** NUM_EDGES  # 15625 — the whole search space

def validate_genome(g):
    """Raise ValueError unless g is a length-6 sequence of ops in 0..4."""
    if len(g) != NUM_EDGES:
        raise ValueError(f"genome must have {NUM_EDGES} edges, got {len(g)}: {g!r}")
    for i, op in enumerate(g):
        if not isinstance(op, (int, np.integer)) or not 0 <= int(op) < NUM_OPS:
            raise ValueError(
                f"edge {i} must be an int in 0..{NUM_OPS - 1}, got {op!r}")
    return g

def random_genome(rng):
    return tuple(int(x) for x in rng.integers(0, NUM_OPS, size=NUM_EDGES))

def genome_to_index(g):
    validate_genome(g)
    # int() keeps the return a plain Python int even if g holds numpy integers
    return sum(int(op) * (NUM_OPS ** i) for i, op in enumerate(g))

def index_to_genome(idx):
    if not 0 <= idx < NUM_ARCHS:
        raise ValueError(f"index must be in 0..{NUM_ARCHS - 1}, got {idx}")
    g = []
    for _ in range(NUM_EDGES):
        g.append(idx % NUM_OPS)
        idx //= NUM_OPS
    return tuple(g)

def conv_count(g):
    return sum(1 for op in g if op in CONV_OPS)

# cell edges in genome order: gene k wires EDGES[k] = (src_node, dst_node).
# node i receives from every node j < i, so the DAG is fixed — only the ops vary.
EDGES = ((0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3))

def cell_depth(g):
    """Longest path from input (node0) to output (node3) over present (non-'none')
    edges, counting each edge as one hop. 0 means node3 is unreachable — the cell
    is disconnected and contributes nothing but the surrounding macro-architecture."""
    reach = {0: 0}
    for node in (1, 2, 3):  # topological order: a node's inputs are all earlier
        hops = [reach[s] + 1 for k, (s, d) in enumerate(EDGES)
                if d == node and g[k] != 0 and s in reach]
        if hops:
            reach[node] = max(hops)
    return reach.get(3, 0)

def genome_to_arch_str(g):
    node1 = f"|{OPS[g[0]]}~0|"
    node2 = f"|{OPS[g[1]]}~0|{OPS[g[2]]}~1|"
    node3 = f"|{OPS[g[3]]}~0|{OPS[g[4]]}~1|{OPS[g[5]]}~2|"
    return f"{node1}+{node2}+{node3}"
