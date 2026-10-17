"""Real clips -> real detector -> bridge -> uploader -> local mock backend (GPU, ~40 s).

Skipped unless LASTSEEN_E2E=1 (needs the phone clips and model weights):
    $env:LASTSEEN_E2E=1; pytest -q tests/test_bridge_e2e.py
"""

import os
from pathlib import Path

import pytest

from lastscene_bridge.bridge import LastSceneBridge
from lastscene_bridge.capture_time import recording_time_ms
from lastscene_bridge.observation import captured_at_iso
from lastscene_bridge.uploader import Uploader
from lastscene_bridge.auth import token_provider_from_env
from memory_engine.api import LastSeen
from memory_engine.frames import FrameSource
from tests import mock_lastscene as mock

CLIPS = Path(__file__).resolve().parents[1] / "data" / "clips"
pytestmark = pytest.mark.skipif(os.environ.get("LASTSEEN_E2E") != "1" or not (CLIPS / "01_normal_placement.MOV").exists(),
                                reason="set LASTSEEN_E2E=1 (needs the phone clips and the YOLO-World model)")


def run_clip(tmp_path, clip, url, preset="m960d"):
    env = {"LASTSCENE_API_BASE": url, "LASTSCENE_PAIRING_CODE": mock.PAIRING_CODE, "AUTH0_DOMAIN": url,
           "AUTH0_CLIENT_ID": mock.CLIENT_ID, "AUTH0_CLIENT_SECRET": mock.CLIENT_SECRET, "AUTH0_AUDIENCE": mock.AUDIENCE}
    up = Uploader(url, mock.PAIRING_CODE, token_provider_from_env(env), tmp_path / "queue")
    ls = LastSeen(out_dir=str(tmp_path / "mem"), preset=preset, source=str(CLIPS / clip), source_kind="file")
    bridge = LastSceneBridge(ls, uploader=up, preset_name=preset)
    rec = recording_time_ms(CLIPS / clip)
    for f in FrameSource(str(CLIPS / clip), recorded_at_ms=rec, max_fps=2):
        bridge.handle(ls.process_frame(f.image, f.video_ms, f.wall_ms, f.observed_ms))
    bridge.handle(ls.finish())
    results = up.flush()
    ls.close()
    return bridge, results, rec


def test_real_placement_reaches_the_backend(tmp_path):
    srv, state, url = mock.serve(0)
    try:
        bridge, results, rec = run_clip(tmp_path, "01_normal_placement.MOV", url)
        assert [s for _, s in results] == ["sent"]
        (stored,) = state.observations.values()
        obs = stored["obs"]
        assert obs["objects"][0]["label"] == "keys" and obs["objects"][0]["event"] == "seen"
        # captured_at = real recording time + where in the clip the keys were confirmed (placed ~4.9 s)
        assert captured_at_iso(rec + 5000) <= obs["captured_at"] <= captured_at_iso(rec + 8000)
        assert state.last_seen["keys"]["observation_id"] == obs["observation_id"]
    finally:
        srv.shutdown()


def test_no_keys_clip_uploads_nothing_with_m960d(tmp_path):
    srv, state, url = mock.serve(0)
    try:
        bridge, results, _ = run_clip(tmp_path, "IMG_7688.MOV", url)
        assert results == [] and state.observations == {}
    finally:
        srv.shutdown()
