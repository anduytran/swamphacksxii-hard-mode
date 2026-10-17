from memory_engine.detector import CascadeDetector, Detection


class Fake:
    """Stand-in detector that returns a scripted confidence per call."""

    def __init__(self, confs):
        self.confs, self.calls = list(confs), 0

    def detect(self, image):
        conf = self.confs[self.calls]
        self.calls += 1
        return [Detection("keys", conf, (0, 0, 10, 10))] if conf else []


def test_confident_light_result_never_wakes_the_heavy_model():
    light, heavy = Fake([0.6, 0.5]), Fake([])
    cascade = CascadeDetector(light, heavy)
    assert cascade.detect(None)[0].conf == 0.6
    cascade.detect(None)
    assert heavy.calls == 0 and cascade.heavy_fraction == 0.0


def test_weak_candidate_escalates_to_heavy():
    light, heavy = Fake([0.05]), Fake([0.4])
    cascade = CascadeDetector(light, heavy)
    assert cascade.detect(None)[0].conf == 0.4
    assert heavy.calls == 1


def test_nothing_seen_stays_cheap_unless_periodic_check_is_due():
    light, heavy = Fake([0, 0, 0, 0]), Fake([0.3])
    cascade = CascadeDetector(light, heavy, periodic_every=4)
    results = [cascade.detect(None) for _ in range(4)]
    assert results[:3] == [[], [], []]
    assert results[3][0].conf == 0.3 and heavy.calls == 1


def test_forced_search_always_uses_heavy():
    light, heavy = Fake([]), Fake([0.2])
    cascade = CascadeDetector(light, heavy)
    assert cascade.detect(None, force_heavy=True)[0].conf == 0.2
    assert light.calls == 0
