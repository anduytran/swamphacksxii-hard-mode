"""Scene matcher tests on two real photos (Ultralytics sample images) and synthetic edits.

Synthetic warps are flat (planar), so a homography fits them exactly. Passing these
tests shows the code is correct; it does not show real-world accuracy, where 3D
parallax, blur and lighting make things harder.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from memory_engine.scene import DIFFERENT, SAME, UNKNOWN, SceneConfig, SceneFeatures, SceneMatcher, \
    homography_is_plausible, visible_fraction

FIXTURES = Path(__file__).parent / "fixtures"
BOX = (223, 406, 344, 860)  # a person in bus.jpg, standing in for "the keys"


@pytest.fixture(scope="module")
def matcher():
    return SceneMatcher()


@pytest.fixture(scope="module")
def bus():
    return cv2.imread(str(FIXTURES / "bus.jpg"))


@pytest.fixture(scope="module")
def zidane():
    return cv2.imread(str(FIXTURES / "zidane.jpg"))


def perspective(img, d):
    """Tilt the view: corners pulled in by fraction d (like looking from the side)."""
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[d * w, 0], [w - d * w, d * h], [w, h], [0, h - d * h]])
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, H, (w, h)), H


def true_box(H, bbox):
    x1, y1, x2, y2 = bbox
    c = cv2.perspectiveTransform(np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2), H)
    c = c.reshape(-1, 2)
    return (*c.min(axis=0), *c.max(axis=0))


def box_error(a, b):
    return max(abs(p - q) for p, q in zip(a, b))


def compare(matcher, img_a, img_b):
    return matcher.match(matcher.features(img_a), matcher.features(img_b))


# --- the same place, seen differently ---------------------------------------

def test_identical_image_is_same_scene(matcher, bus):
    r = compare(matcher, bus, bus)
    assert r.verdict == SAME and r.projection_trusted
    assert matcher.project_box(r, BOX) == BOX


@pytest.mark.parametrize("d", [0.05, 0.1, 0.2])
def test_moderate_viewpoint_change_maps_box_correctly(matcher, bus, d):
    warped, H = perspective(bus, d)
    r = compare(matcher, bus, warped)
    assert r.verdict == SAME and r.projection_trusted
    assert box_error(matcher.project_box(r, BOX), true_box(H, BOX)) < 5


def test_head_turn_and_step_back(matcher, bus):
    # camera turned (content shifts) and moved back (content shrinks)
    h, w = bus.shape[:2]
    H = np.float64([[0.9, 0, 120], [0, 0.9, 40], [0, 0, 1]])
    r = compare(matcher, bus, cv2.warpPerspective(bus, H, (w, h)))
    assert r.verdict == SAME
    assert box_error(matcher.project_box(r, BOX), true_box(H, BOX)) < 5


def test_looking_back_with_partial_overlap(matcher, bus):
    # turned far enough that only part of the old view is still in frame
    h, w = bus.shape[:2]
    H = np.float64([[1, 0, 350], [0, 1, 0], [0, 0, 1]])
    r = compare(matcher, bus, cv2.warpPerspective(bus, H, (w, h)))
    assert r.verdict == SAME
    projected = matcher.project_box(r, BOX)
    assert box_error(projected, true_box(H, BOX)) < 5
    assert visible_fraction(projected, bus.shape) == pytest.approx(1.0)


def test_partial_occlusion(matcher, bus):
    occluded = bus.copy()
    occluded[300:800, 0:500] = 128  # someone's arm covers a big chunk, including the box
    r = compare(matcher, bus, occluded)
    assert r.verdict == SAME and r.projection_trusted


def test_darker_lighting(matcher, bus):
    assert compare(matcher, bus, cv2.convertScaleAbs(bus, alpha=0.5, beta=-10)).verdict == SAME


def test_motion_blur_never_gives_a_wrong_trusted_box(matcher, bus):
    kernel = np.full((1, 21), 1 / 21)  # 21 px horizontal smear, like a quick head turn
    r = compare(matcher, bus, cv2.filter2D(bus, -1, kernel))
    if r.projection_trusted:
        assert box_error(matcher.project_box(r, BOX), BOX) < 5


def test_extreme_tilt_is_unknown_not_different(matcher, bus):
    warped, _ = perspective(bus, 0.3)
    assert compare(matcher, bus, warped).verdict != DIFFERENT, "strong tilt misread as a new place"


def test_random_viewpoints_never_produce_a_wrong_trusted_box(matcher, bus):
    """Safety property: a projection we trust must be right, whatever the verdict rate."""
    rng = np.random.default_rng(0)
    h, w = bus.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    a = matcher.features(bus)
    for _ in range(40):
        jitter = rng.uniform(-0.35, 0.35, size=(4, 2)) * [w, h]
        H = cv2.getPerspectiveTransform(src, (src + jitter).astype(np.float32))
        r = matcher.match(a, matcher.features(cv2.warpPerspective(bus, H, (w, h))))
        if r.projection_trusted:
            assert box_error(matcher.project_box(r, BOX), true_box(H, BOX)) < 8


# --- different places, and places we can't judge ----------------------------

def test_different_scene_is_not_called_same(matcher, bus, zidane):
    r = compare(matcher, bus, zidane)
    assert r.verdict == DIFFERENT
    assert matcher.project_box(r, BOX) is None


def test_different_scene_after_warp_is_not_called_same(matcher, bus, zidane):
    warped, _ = perspective(zidane, 0.1)
    assert compare(matcher, bus, warped).verdict != SAME


def test_blank_wall_is_unknown(matcher, bus):
    r = compare(matcher, bus, np.full_like(bus, 200))
    assert (r.verdict, r.reason) == (UNKNOWN, "low_texture")


def test_heavily_blurred_view_is_not_called_different(matcher, bus):
    r = compare(matcher, bus, cv2.GaussianBlur(bus, (0, 0), 15))
    assert r.verdict in (SAME, UNKNOWN)


def test_low_texture_never_yields_different(matcher):
    gradient = np.tile(np.linspace(0, 255, 640, dtype=np.uint8), (480, 1))
    r = compare(matcher, cv2.merge([gradient] * 3), cv2.merge([gradient[::-1]] * 3))
    assert r.verdict == UNKNOWN


def test_excluded_object_box_contributes_no_features(matcher, bus):
    feats = matcher.features(bus, exclude_boxes=[BOX], pad=0.0)
    x1, y1, x2, y2 = (v * feats.scale for v in BOX)
    inside = (feats.points[:, 0] > x1 + 2) & (feats.points[:, 0] < x2 - 2) & \
             (feats.points[:, 1] > y1 + 2) & (feats.points[:, 1] < y2 - 2)
    assert len(feats) > 0 and not inside.any()


def test_surroundings_only_still_maps_the_box(matcher, bus):
    warped, H = perspective(bus, 0.1)
    r = matcher.match(matcher.features(bus, exclude_boxes=[BOX]), matcher.features(warped))
    assert r.verdict == SAME
    assert box_error(matcher.project_box(r, BOX), true_box(H, BOX)) < 5


# --- geometry helpers --------------------------------------------------------

def fake_features(shape=(480, 640)):
    return SceneFeatures(np.zeros((0, 2), np.float32), None, 1.0, shape)


def test_mirror_and_wild_zoom_are_rejected():
    cfg = SceneConfig()
    mirrored = np.diag([-1.0, 1.0, 1.0])
    assert homography_is_plausible(mirrored, fake_features(), cfg) == (False, "mirrored")
    zoomed = np.diag([5.0, 5.0, 1.0])
    assert homography_is_plausible(zoomed, fake_features(), cfg) == (False, "scale")
    assert homography_is_plausible(np.eye(3), fake_features(), cfg) == (True, "")


def test_visible_fraction():
    assert visible_fraction((0, 0, 10, 10), (100, 100)) == 1.0
    assert visible_fraction((90, 0, 110, 10), (100, 100)) == 0.5
    assert visible_fraction((200, 200, 210, 210), (100, 100)) == 0.0
