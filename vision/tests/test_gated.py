import numpy as np

from lastscene_pipeline.gated import GatedFrames


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def scene(level):
    return np.full((48, 64, 3), level, np.uint8)


def run(images, until, paused=lambda t: False, **kw):
    """Grab images[i] on the i-th grab (the last one repeats); returns (passed grab indexes, stats)."""
    clock, grabs = Clock(), []

    def grab():
        grabs.append(clock.t)
        return images[min(len(grabs) - 1, len(images) - 1)]

    gate = GatedFrames(grab, paused=lambda: paused(clock.t), stop=lambda: clock.t >= until,
                       clock=clock, sleep=clock.sleep, wall=lambda: 1_000_000 + clock.t, **kw)
    passed = [len(grabs) - 1 for _ in gate]
    return passed, gate.stats, grabs


def test_still_scene_is_skipped_after_settling_except_the_heartbeat():
    passed, stats, _ = run([scene(100)], until=130, every=2, settle=6, heartbeat=60)
    # first frame counts as a change: it and the frames in the next 6 s (+2 s, +4 s) pass, then only heartbeats
    assert passed[:3] == [0, 1, 2]
    assert len(passed) == 3 + 2  # heartbeats at about 60 s and 120 s
    assert stats["grabbed"] == 65 and stats["skipped"] == stats["grabbed"] - len(passed)


def test_a_change_passes_with_its_settle_window():
    images = [scene(100)] * 10 + [scene(160)] + [scene(160)] * 30
    passed, _, _ = run(images, until=60, every=2, settle=6, heartbeat=600)
    assert passed == [0, 1, 2, 10, 11, 12]  # start-up, then the change and the item at rest 2 s and 4 s later


def test_small_flicker_is_not_motion():
    images = [scene(100), scene(101), scene(99), scene(102)] * 20
    passed, _, _ = run(images, until=80, every=2, settle=0, heartbeat=600)
    assert passed == [0]


def test_paused_means_no_grabs_and_a_full_interval_after():
    passed, stats, grabs = run([scene(100)], until=20, every=2, settle=100, heartbeat=600, paused=lambda t: 5 <= t < 12)
    assert not [t for t in grabs if 5 <= t < 12]
    assert min(t for t in grabs if t >= 12) >= 13.9  # resumes `every` seconds after the pause ends
    assert stats["paused_s"] > 6


def test_frames_carry_wall_clock_capture_times():
    clock = Clock()
    gate = GatedFrames(lambda: scene(1), every=1, stop=lambda: clock.t >= 3, clock=clock, sleep=clock.sleep,
                       wall=lambda: 1_700_000_000 + clock.t)
    frames = list(gate)
    assert frames[0].observed_ms == frames[0].wall_ms == 1_700_000_000_000
    assert [f.frame_id for f in frames] == list(range(len(frames)))
