import numpy as np
from artificial_society.environment.materials import DiscoveryRegistry


def test_peek_does_not_increment_but_record_use_does():
    reg = DiscoveryRegistry()
    mid = reg.register(np.ones(12, dtype=np.float32), discoverer_id=7, tick=3, recipe=("strike", "a", "b"))
    e = next(x for x in reg.entries if x["id"] == mid)
    assert e["uses"] == 0
    reg.peek_vector(mid); reg.peek_vector(mid)
    assert e["uses"] == 0          # peek is side-effect free
    reg.record_use(mid)
    assert e["uses"] == 1          # explicit use is counted
    assert e["discovered_by"] == 7 and e["tick"] == 3
