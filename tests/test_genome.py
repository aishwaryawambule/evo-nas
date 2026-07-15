import numpy as np
from evonas.genome import (
    OPS, NUM_EDGES, random_genome, genome_to_index,
    index_to_genome, genome_to_arch_str, conv_count,
)

def test_ops_are_the_five_nb201_operations():
    assert OPS == ("none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3")

def test_random_genome_has_six_valid_edges():
    rng = np.random.default_rng(0)
    g = random_genome(rng)
    assert len(g) == NUM_EDGES
    assert all(0 <= op <= 4 for op in g)

def test_index_round_trips_over_whole_space():
    for idx in (0, 1, 4, 5, 3125, 15624):
        assert genome_to_index(index_to_genome(idx)) == idx

def test_index_covers_exactly_15625():
    seen = {genome_to_index(index_to_genome(i)) for i in range(5**6)}
    assert seen == set(range(5**6))

def test_conv_count_counts_only_conv_ops():
    assert conv_count((2, 3, 0, 1, 4, 2)) == 3  # ops 2,3,2 are conv
    assert conv_count((0, 1, 4, 0, 1, 4)) == 0

def test_arch_str_matches_nb201_format():
    g = (0, 0, 0, 0, 0, 0)
    assert genome_to_arch_str(g) == "|none~0|+|none~0|none~1|+|none~0|none~1|none~2|"
