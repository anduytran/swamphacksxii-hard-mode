from scripts.detect_test import summarize


def test_counts_frames_not_boxes():
    # two "keys" boxes in one frame still count as one detected frame
    frames = [[("keys", 0.4), ("keys", 0.6)], [], [("keys", 0.2)]]
    row = summarize(frames, 3, ["keys"])[0]
    assert row["frames_detected"] == 2
    assert row["detect_rate"] == round(2 / 3, 3)
    assert row["mean_conf"] == 0.4  # mean of per-frame best: (0.6 + 0.2) / 2
    assert row["max_conf"] == 0.6


def test_never_detected_prompt_still_reported():
    rows = summarize([[("wallet", 0.5)]], 1, ["keys", "wallet"])
    by_label = {r["label"]: r for r in rows}
    assert by_label["keys"]["frames_detected"] == 0
    assert by_label["keys"]["detect_rate"] == 0.0
    assert by_label["wallet"]["detect_rate"] == 1.0


def test_zero_frames_does_not_divide_by_zero():
    assert summarize([], 0, ["keys"])[0]["detect_rate"] == 0.0
