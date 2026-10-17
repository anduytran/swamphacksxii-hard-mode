r"""Timestamped contact sheet of a clip, for setting the label windows in an eval manifest.

    python scripts\contact_sheet.py data\clips\esp32\01_placement.mp4                  # whole clip, every 0.5 s
    python scripts\contact_sheet.py data\clips\esp32\01_placement.mp4 --from 4 --to 6 --every 0.2   # zoom in on a moment

Saves <clip>_sheet.jpg next to the clip (or --out). Read off when the keys are in hand,
put down, visible, out of view, and write those windows into the manifest.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def contact_sheet(path, every=0.5, start=0.0, end=None, cols=6, tile_w=240, out=None) -> Path:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    end = min(end if end is not None else duration, duration)
    tiles, t = [], start
    while t <= end + 1e-6:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, img = cap.read()
        if not ok:
            break
        img = cv2.resize(img, (tile_w, int(tile_w * img.shape[0] / img.shape[1])))
        cv2.putText(img, f"{t:.1f}s", (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        tiles.append(img)
        t += every
    cap.release()
    if not tiles:
        raise SystemExit(f"no frames read from {path}")
    blank = np.zeros_like(tiles[0])
    tiles += [blank] * (-len(tiles) % cols)
    sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    out = Path(out) if out else Path(path).with_name(Path(path).stem + "_sheet.jpg")
    cv2.imwrite(str(out), sheet)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("clip")
    p.add_argument("--every", type=float, default=0.5)
    p.add_argument("--from", dest="start", type=float, default=0.0)
    p.add_argument("--to", dest="end", type=float)
    p.add_argument("--out")
    args = p.parse_args()
    print(contact_sheet(args.clip, args.every, args.start, args.end, out=args.out))


if __name__ == "__main__":
    main()
