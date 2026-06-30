from artificial_society.systems import _lineage_frontier as lf

def test_frontier_marginal_and_monotone():
    lf.reset_frontiers()
    assert lf.frontier_value(1) == 0.0
    assert lf.update_frontier(1, 0.8) == 0.8       # first advance = full value
    assert lf.update_frontier(1, 0.5) == 0.0       # below frontier = no marginal
    assert round(lf.update_frontier(1, 1.1), 6) == 0.3  # only the increment counts
    assert lf.frontier_value(1) == 1.1
    assert lf.frontier_value(2) == 0.0             # tribes are independent
