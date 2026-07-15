from evonas.genome import index_to_genome

def enumerate_all(benchmark):
    for idx in range(5 ** 6):
        g = index_to_genome(idx)
        yield g, benchmark.query(g)

def ground_truth_archive(benchmark, make_archive):
    archive = make_archive()
    for g, rec in enumerate_all(benchmark):
        archive.insert(g, rec)
    return archive

def reachable_cells(benchmark, make_archive):
    return len(ground_truth_archive(benchmark, make_archive).cells)
