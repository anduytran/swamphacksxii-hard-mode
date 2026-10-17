from argparse import Namespace

from memory_engine.run import memory_config


def args(**kw):
    base = dict(min_conf=0.05, min_hits=None, min_seen_ms=None, max_gap_ms=None)
    return Namespace(**{**base, **kw})


def test_two_fps_needs_fewer_hits_and_longer_gaps():
    cfg = memory_config(args(), inference_fps=2)
    assert cfg.min_hits == 2
    assert cfg.max_gap_ms == 1500  # 3 x 500 ms = 1500, same as the default


def test_one_fps_stretches_the_gap_tolerance():
    assert memory_config(args(), inference_fps=1).max_gap_ms == 3000


def test_unlimited_rate_keeps_original_defaults():
    cfg = memory_config(args(), inference_fps=0)
    assert (cfg.min_hits, cfg.max_gap_ms) == (3, 1500)


def test_explicit_flags_win():
    cfg = memory_config(args(min_hits=4, max_gap_ms=900), inference_fps=1)
    assert (cfg.min_hits, cfg.max_gap_ms) == (4, 900)


def test_bridge_cli_defaults_to_m960d_and_stays_overridable():
    from lastscene_bridge.run import build_parser

    args = build_parser().parse_args(["--source", "http://cam/stream"])
    assert args.preset == "m960d" and args.reconnect_seconds == 60 and not args.show
    assert build_parser().parse_args(["--source", "0", "--preset", "efficient"]).preset == "efficient"
