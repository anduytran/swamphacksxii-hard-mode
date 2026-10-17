"""Is this the same place as before? Compare two images with ORB features + RANSAC.

How it works, in plain terms:
1. ORB finds distinctive little patterns (corners, texture, print) in each image and
   gives each one a short binary "fingerprint" (descriptor).
2. We pair fingerprints between the two images and keep only clear-cut pairs
   (Lowe's ratio test: the best match must be clearly better than the second best).
3. RANSAC fits one homography (a perspective transform from image A to image B) that
   as many pairs as possible agree with. Pairs that agree are "inliers".
4. Many inliers, spread over the image, means the same scene. The homography then
   lets us map a box from the old image into the new one.

Three possible verdicts, on purpose:
- same_scene:      enough well-spread inliers.
- different_scene: both images have plenty of texture, yet almost nothing matches.
                   This is evidence, not proof: a very different viewpoint of the
                   same place can look exactly like this.
- unknown:         everything else (too little texture, few or clustered inliers).
                   A failed match is NOT treated as a different place.

Box projection is a separate, stricter decision (`projection_trusted`): a homography
is exact only for flat scenes or pure camera rotation. With parallax (3D clutter, the
camera moving sideways) it can be wrong even when the scene match is right, so we also
run geometric sanity checks before trusting it.

This module knows nothing about memories or YOLO. Usage:
    matcher = SceneMatcher()
    a = matcher.features(old_image)      # cache this per memory
    b = matcher.features(new_image)      # compute once per frame you want to check
    result = matcher.match(a, b)
    box_now = matcher.project_box(result, old_bbox)   # None unless projection_trusted
"""

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

SAME = "same_scene"
DIFFERENT = "different_scene"
UNKNOWN = "unknown"


@dataclass
class SceneConfig:
    max_width: int = 640              # images are shrunk to this width before matching (speed + consistent thresholds)
    n_features: int = 1500            # ORB keypoints per image
    ratio_test: float = 0.75          # Lowe's ratio: best match distance < 0.75 x second best
    min_features: int = 150           # fewer keypoints in either image -> too little texture to judge
    rich_features: int = 500          # both images need this many before we may say "different_scene"
    max_inliers_for_different: int = 8
    min_inliers: int = 25
    min_inlier_ratio: float = 0.25    # inliers / ratio-test matches
    min_spread: float = 0.05          # inliers' convex hull must cover this fraction of each image
    ransac_px: float = 5.0            # RANSAC reprojection tolerance, in working-scale pixels
    max_scale_change: float = 3.0     # implied zoom outside [1/3, 3] -> implausible for a glance
    max_perspective: float = 0.002    # |h20|, |h21| at working scale; larger = extreme tilt, distrust


@dataclass
class SceneFeatures:
    points: np.ndarray                # (N, 2) keypoint positions at working scale
    descriptors: np.ndarray | None    # (N, 32) ORB fingerprints
    scale: float                      # working size / original size
    shape: tuple                      # original image (height, width)

    def __len__(self):
        return len(self.points)


@dataclass
class MatchResult:
    verdict: str
    reason: str
    features_a: int
    features_b: int
    good_matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    spread_a: float = 0.0
    spread_b: float = 0.0
    projection_trusted: bool = False
    elapsed_ms: float = 0.0
    homography: np.ndarray | None = field(default=None, repr=False)  # A working coords -> B working coords
    scale_a: float = 1.0
    scale_b: float = 1.0
    inlier_points_a: np.ndarray | None = field(default=None, repr=False)  # original-image pixels, for drawing
    inlier_points_b: np.ndarray | None = field(default=None, repr=False)


class SceneMatcher:
    def __init__(self, config: SceneConfig | None = None):
        self.config = config or SceneConfig()
        self._orb = cv2.ORB_create(nfeatures=self.config.n_features)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def features(self, image, exclude_boxes=(), pad: float = 0.25) -> SceneFeatures:
        """ORB features for an image.

        exclude_boxes: (x1, y1, x2, y2) boxes, e.g. the object itself, whose features are
        ignored (grown by `pad` x box size). Without this, a textured object like a key ring
        dominates the match, and "same scene" quietly turns into "same object".
        """
        h, w = image.shape[:2]
        scale = min(1.0, self.config.max_width / w)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        if scale < 1.0:
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        mask = None
        if exclude_boxes:
            mask = np.full(gray.shape, 255, dtype=np.uint8)
            for x1, y1, x2, y2 in exclude_boxes:
                px, py = (x2 - x1) * pad, (y2 - y1) * pad
                cv2.rectangle(mask, (int((x1 - px) * scale), int((y1 - py) * scale)),
                              (int((x2 + px) * scale), int((y2 + py) * scale)), 0, -1)
        keypoints, descriptors = self._orb.detectAndCompute(gray, mask)
        points = np.float32([kp.pt for kp in keypoints]).reshape(-1, 2)
        return SceneFeatures(points, descriptors, scale, (h, w))

    def match(self, a: SceneFeatures, b: SceneFeatures) -> MatchResult:
        cfg = self.config
        start = time.perf_counter()
        result = MatchResult(UNKNOWN, "", len(a), len(b), scale_a=a.scale, scale_b=b.scale)

        def done(verdict, reason):
            result.verdict, result.reason = verdict, reason
            result.elapsed_ms = (time.perf_counter() - start) * 1000
            return result

        if len(a) < cfg.min_features or len(b) < cfg.min_features:
            return done(UNKNOWN, "low_texture")
        both_rich = len(a) >= cfg.rich_features and len(b) >= cfg.rich_features

        pairs = self._matcher.knnMatch(a.descriptors, b.descriptors, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < cfg.ratio_test * p[1].distance]
        result.good_matches = len(good)
        if len(good) < 4:  # a homography needs at least 4 pairs
            return done(DIFFERENT if both_rich else UNKNOWN, "no_consistent_matches")

        src = a.points[[m.queryIdx for m in good]]
        dst = b.points[[m.trainIdx for m in good]]
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, cfg.ransac_px)
        if H is None:
            return done(DIFFERENT if both_rich else UNKNOWN, "no_consistent_matches")

        inlier = mask.ravel().astype(bool)
        result.inliers = int(inlier.sum())
        result.inlier_ratio = round(result.inliers / len(good), 3)
        result.homography = H
        result.inlier_points_a = src[inlier] / a.scale
        result.inlier_points_b = dst[inlier] / b.scale

        if result.inliers <= cfg.max_inliers_for_different:
            return done(DIFFERENT if both_rich else UNKNOWN, "no_consistent_matches")

        result.spread_a = round(_hull_fraction(src[inlier], a), 3)
        result.spread_b = round(_hull_fraction(dst[inlier], b), 3)
        if result.inliers < cfg.min_inliers or result.inlier_ratio < cfg.min_inlier_ratio:
            return done(UNKNOWN, "too_few_inliers")
        if min(result.spread_a, result.spread_b) < cfg.min_spread:
            return done(UNKNOWN, "inliers_clustered")

        plausible, why = homography_is_plausible(H, a, cfg)
        result.projection_trusted = plausible
        return done(SAME, "matched" if plausible else f"matched_projection_untrusted:{why}")

    def project_box(self, result: MatchResult, bbox):
        """Map an (x1, y1, x2, y2) box from image A into image B's pixels, or None if untrusted.

        Returns the axis-aligned box around the projected corners. It may extend past
        image B's edges; use visible_fraction() to check how much is in view.
        """
        if not result.projection_trusted or result.homography is None:
            return None
        x1, y1, x2, y2 = (v * result.scale_a for v in bbox)
        corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, result.homography).reshape(-1, 2) / result.scale_b
        (px1, py1), (px2, py2) = projected.min(axis=0), projected.max(axis=0)
        return int(round(px1)), int(round(py1)), int(round(px2)), int(round(py2))


def visible_fraction(bbox, shape) -> float:
    """Share of a box's area that lies inside an image of shape (height, width)."""
    h, w = shape[:2]
    x1, y1, x2, y2 = bbox
    area = max(0, x2 - x1) * max(0, y2 - y1)
    if area == 0:
        return 0.0
    inside = max(0, min(x2, w) - max(x1, 0)) * max(0, min(y2, h) - max(y1, 0))
    return inside / area


def homography_is_plausible(H, a: SceneFeatures, cfg: SceneConfig) -> tuple[bool, str]:
    """Reject transforms a real head turn couldn't produce."""
    if not np.all(np.isfinite(H)) or abs(H[2, 2]) < 1e-9:
        return False, "degenerate"
    H = H / H[2, 2]
    det = np.linalg.det(H[:2, :2])
    if det <= 0:
        return False, "mirrored"
    zoom = np.sqrt(det)
    if not (1 / cfg.max_scale_change <= zoom <= cfg.max_scale_change):
        return False, "scale"
    if max(abs(H[2, 0]), abs(H[2, 1])) > cfg.max_perspective:
        return False, "perspective"
    h, w = (int(round(d * a.scale)) for d in a.shape)
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    if not cv2.isContourConvex(cv2.perspectiveTransform(corners, H)):
        return False, "folded"
    return True, ""


def _hull_fraction(points, feats: SceneFeatures) -> float:
    if len(points) < 3:
        return 0.0
    h, w = (d * feats.scale for d in feats.shape)
    return cv2.contourArea(cv2.convexHull(points.astype(np.float32))) / (w * h)
