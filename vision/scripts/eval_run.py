r"""Real-clip evaluation of LastSeen, driven by a manifest (clips + labels). One command:

    python scripts\eval_run.py --manifest eval_manifests\phone_round2.json --presets efficient,m960d,x960d
    python scripts\eval_run.py --manifest eval_manifests\esp32_round1.json --presets efficient,m480d,m640d,m960d

Stage 1 (detect): decodes each clip once, samples it at 10 fps, runs every requested preset
on the full-resolution frame at confidence 0.01, and caches the boxes. Thresholds (0.05 vs
0.03) are applied afterwards, so every comparison shares identical detections. Cached
presets are skipped unless --redo.
Stage 2 (analyze): per preset, the production memory engine and LastSeen API at
2 inferences/s over 5 sampling phases, scene matching on/off, both thresholds. Then a
one-table summary across presets.

Outputs under data/eval/<manifest name>/:
    frames/<clip>/<ms>.jpg            display frames (long side <= 960 px)
    detections/<preset>/<clip>.json   cached boxes in display-frame pixels
    reports/<preset>/report.txt       full report, timelines, false-alarm gallery, navigation preview
    summary.md                        one row per preset

Manifest labels are time windows set by eye. Category meaning:
    absent              keys not visible: any detection is a false alarm
    unscored*, pickup   ambiguous or transitional: reported, never scored
    anything else       keys visible: counts toward recall (e.g. hand, placed, clutter, near, return)
A clip with "no_keys": true is a false-memory test. "placed_at"/"first_visible" give the latency
reference; "walk_away" is when the camera leaves the keys for good; "useful" lists the
categories in which a memory's evidence frame counts as showing the keys at the place.
"""

import argparse
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory_engine.api import LastSeen  # noqa: E402
from memory_engine.detector import Detection, drop_oversized  # noqa: E402
from memory_engine.frames import FrameSource  # noqa: E402
from memory_engine.memory import MemoryMaker  # noqa: E402
from memory_engine.presets import PRESETS, make_detector  # noqa: E402
from memory_engine.run import memory_config  # noqa: E402
from memory_engine.scene import SceneMatcher  # noqa: E402

SAMPLE_FPS = 10
DISPLAY_MAX = 960
THRESHOLDS = (0.05, 0.03)
PHASES = (0, 100, 200, 300, 400)


# --- manifest ------------------------------------------------------------------------
def load_manifest(path):
    m = json.loads(Path(path).read_text(encoding="utf-8-sig"))  # tolerate a BOM (Notepad/PowerShell)
    m.setdefault("prompt", "bunch of keys")
    m.setdefault("clips_dir", "data/clips")
    m["root"] = Path("data/eval") / m["name"]
    usable = []
    for clip in m["clips"]:
        for key in [k for k, v in clip.items() if v is None]:  # template placeholders not filled in yet
            del clip[key]
        clip.setdefault("windows", [])
        clip.setdefault("useful", ["placed", "clutter"])
        clip.setdefault("display", f"{clip['id']} [{clip['file']}]")
        if (Path(m["clips_dir"]) / clip["file"]).exists():
            usable.append(clip)
        else:
            print(f"!! missing clip file, skipped: {Path(m['clips_dir']) / clip['file']}")
    m["clips"] = usable
    m["by_id"] = {c["id"]: c for c in usable}
    return m


def category(clip, t_ms):
    for start, end, cat in clip["windows"]:
        if start * 1000 <= t_ms <= end * 1000:
            return cat
    return None


def scored(cat):
    return cat is not None and cat != "absent" and not cat.startswith("unscored") and cat != "pickup"


# --- stage 1: detection cache ---------------------------------------------------------
def stage_detect(m, presets, redo):
    root = m["root"]
    detectors = {}
    for clip in m["clips"]:
        todo = [p for p in presets if redo or not (root / "detections" / p / f"{clip['id']}.json").exists()]
        frames_dir = root / "frames" / clip["id"]
        if not todo and frames_dir.exists():
            continue
        for p in todo:
            if p not in detectors:
                detectors[p] = make_detector(p, [m["prompt"]], conf=0.01)
        frames_dir.mkdir(parents=True, exist_ok=True)
        rows = {p: [] for p in todo}
        times = {p: [] for p in todo}
        for f in FrameSource(str(Path(m["clips_dir"]) / clip["file"]), max_fps=SAMPLE_FPS):
            scale = min(1.0, DISPLAY_MAX / max(f.image.shape[:2]))
            out = frames_dir / f"{f.video_ms:06d}.jpg"
            if not out.exists():
                small = f.image if scale == 1.0 else cv2.resize(f.image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(out), small, [cv2.IMWRITE_JPEG_QUALITY, 90])
            for p in todo:
                t0 = time.perf_counter()
                dets = detectors[p].detect(f.image)
                times[p].append((time.perf_counter() - t0) * 1000)
                rows[p].append({"t_ms": f.video_ms, "boxes": [[round(v * scale) for v in d.bbox] + [d.conf] for d in dets]})
        for p in todo:
            d = root / "detections" / p
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{clip['id']}.json").write_text(json.dumps({"ms_per_frame": statistics.median(times[p]) if times[p] else None,
                                                              "rows": rows[p]}))
            print(f"  detect {p:<10} {clip['id']:<14} {len(rows[p])} frames, median {statistics.median(times[p]):.1f} ms")


# --- shared helpers -------------------------------------------------------------------
class Ctx:
    def __init__(self, m, preset):
        self.m, self.preset = m, preset
        self.out = m["root"] / "reports" / preset
        self.out.mkdir(parents=True, exist_ok=True)
        self._rows, self._shape = {}, {}

    def rows(self, cid):
        if cid not in self._rows:
            data = json.loads((self.m["root"] / "detections" / self.preset / f"{cid}.json").read_text())
            label = self.m["prompt"]
            self._rows[cid] = [(r["t_ms"], [Detection(label, b[4], tuple(b[:4])) for b in r["boxes"]]) for r in data["rows"]]
            self._rows[cid + ":ms"] = data.get("ms_per_frame")
        return self._rows[cid]

    def ms(self, cid):
        self.rows(cid)
        return self._rows[cid + ":ms"]

    def frame(self, cid, t_ms):
        return cv2.imread(str(self.m["root"] / "frames" / cid / f"{t_ms:06d}.jpg"))

    def shape(self, cid):
        if cid not in self._shape:
            self._shape[cid] = self.frame(cid, self.rows(cid)[0][0]).shape
        return self._shape[cid]


def kept(dets, shape, th):
    return [d for d in drop_oversized(dets, shape, 0.5) if d.conf >= th]


def subsample(rows, fps, phase_ms=0):
    step = int(1000 / fps)
    return [(t, d) for t, d in rows if t >= phase_ms and (t - phase_ms) % step < 50]


class _Args:
    min_hits = min_seen_ms = max_gap_ms = None


def run_memory(ctx, cid, th, scene_on, fps=2, phase_ms=0):
    _Args.min_conf = th
    maker = MemoryMaker(memory_config(_Args, fps), scene_matcher=SceneMatcher() if scene_on else None)
    events = []
    for i, (t, dets) in enumerate(subsample(ctx.rows(cid), fps, phase_ms)):
        img = ctx.frame(cid, t)
        events += maker.process(i, t, t, kept(dets, img.shape, 0), image=img)
    return maker, events + maker.finish()


# --- stage 2 sections ------------------------------------------------------------------
def section_detection(ctx, out, metrics):
    out.append(f"\n## 1. Per-frame detection ({SAMPLE_FPS} fps samples)\n")
    out.append(f"{'clip':<16} {'category':<13} {'frames':>6}  hit@0.05      hit@0.03      max conf  first hit @0.05 / @0.03")
    fps_examples = []
    for clip in ctx.m["clips"]:
        cid, rows = clip["id"], ctx.rows(clip["id"])
        for cat in sorted({category(clip, t) for t, _ in rows} - {None}):
            sel = [(t, d) for t, d in rows if category(clip, t) == cat]
            best = [max((x.conf for x in kept(d, ctx.shape(cid), 0)), default=0) for _, d in sel]
            hits = [sum(b >= th for b in best) for th in THRESHOLDS]
            firsts = [next((t / 1000 for (t, _), b in zip(sel, best) if b >= th), None) for th in THRESHOLDS]
            out.append(f"{cid:<16} {cat:<13} {len(sel):>6}  " + "  ".join(f"{h:>4} ({100 * h // len(sel):>3}%)" for h in hits)
                       + f"   {max(best):.3f}    " + " / ".join("-" if f is None else f"{f:.1f}s" for f in firsts))
            metrics["recall"][(cid, cat)] = (hits[0], hits[1], len(sel))
            if cat == "absent":
                fps_examples += [(cid, t, b, max(kept(d, ctx.shape(cid), 0), key=lambda x: x.conf))
                                 for (t, d), b in zip(sel, best) if b >= min(THRESHOLDS)]
    return fps_examples


def section_memory(ctx, out, metrics):
    out.append(f"\n## 2. Memory creation at 2 inferences/s over {len(PHASES)} sampling phases\n")
    out.append("useful = a memory exists before walk-away AND its final evidence frame is in a 'useful' category "
               "(keys at the place, not in a hand). For no-keys clips, every memory is a false memory.\n")
    out.append(f"{'clip':<16} {'th':>4} {'scene':>5}  {'memories/phase':<17} {'first memory s/phase':<32} {'result':<16} evidence in")
    for clip in ctx.m["clips"]:
        cid = clip["id"]
        ref = clip.get("placed_at", clip.get("first_visible"))
        for th in THRESHOLDS:
            for scene_on in (True, False):
                counts, firsts, useful, cats, lat = [], [], 0, [], []
                for ph in PHASES:
                    maker, events = run_memory(ctx, cid, th, scene_on, 2, ph)
                    created = [e for e in events if e["event"] == "created"]
                    counts.append(len(maker.memories))
                    firsts.append(f"{created[0]['video_ms'] / 1000:.1f}" if created else "-")
                    if created and ref is not None:
                        lat.append(created[0]["video_ms"] / 1000 - ref)
                    cats += [category(clip, mem.video_ms) or "?" for mem in maker.memories.values()]
                    if "walk_away" in clip and any(e["video_ms"] / 1000 <= clip["walk_away"] for e in created) and any(
                            category(clip, mem.video_ms) in clip["useful"] for mem in maker.memories.values()):
                        useful += 1
                if clip.get("no_keys"):
                    result = f"false mem {sum(c > 0 for c in counts)}/{len(PHASES)}"
                elif "walk_away" in clip:
                    result = f"useful {useful}/{len(PHASES)}"
                else:
                    result = f"formed {sum(c > 0 for c in counts)}/{len(PHASES)}"
                if scene_on:
                    metrics["memory"][(cid, th)] = result
                    metrics["latency"][(cid, th)] = lat
                    metrics["dupes"][(cid, th)] = max(counts)
                cat_txt = ",".join(f"{c}x{cats.count(c)}" for c in sorted(set(cats))) or "-"
                out.append(f"{cid:<16} {th:>4} {'on' if scene_on else 'off':>5}  {str(counts):<17} {' '.join(firsts):<32} "
                           f"{result:<16} {cat_txt}")


def section_sampling(ctx, out):
    out.append("\n## 3. Inference rate vs first memory time (th=0.05, scene on)\n")
    for clip in ctx.m["clips"]:
        if clip.get("no_keys"):
            continue
        for fps, phases in ((2, PHASES), (5, (0, 100)), (10, (0,))):
            cells = []
            for ph in phases:
                maker, events = run_memory(ctx, clip["id"], 0.05, True, fps, ph)
                created = [e["video_ms"] / 1000 for e in events if e["event"] == "created"]
                cells.append(f"+{ph}:{created[0]:.1f}" if created else f"+{ph}:none")
            out.append(f"  {clip['id']:<16} {fps:>2}/s  {'  '.join(cells)}")


class CachedDetector:
    def __init__(self):
        self.current = []

    def detect(self, image):
        return self.current


def section_flow(ctx, out, metrics):
    flow = ctx.m.get("flow", {})
    mem_clip, nav_clip = ctx.m["by_id"].get(flow.get("memory_clip")), ctx.m["by_id"].get(flow.get("navigation_clip"))
    if not mem_clip:
        return
    out.append(f"\n## 4. Retrieval -> navigation -> found it (LastSeen API, scene on, 2/s): memory from "
               f"{mem_clip['id']}, navigation on {nav_clip['id'] if nav_clip else '-'}\n")
    store, det = ctx.out / "api_store", CachedDetector()
    ls, mem, used = None, None, None
    for th in THRESHOLDS:
        for ph in PHASES:
            if ls:
                ls.close()
            shutil.rmtree(store, ignore_errors=True)
            ls = LastSeen(out_dir=str(store), detector=det, search_detector=det, prompts=(ctx.m["prompt"],))
            for t, dets in subsample(ctx.rows(mem_clip["id"]), 2, ph):
                img = ctx.frame(mem_clip["id"], t)
                det.current = kept(dets, img.shape, th)
                ls.process_frame(img, t, wall_ms=t, detections=det.current)
            ls.finish()
            mem = ls.latest_memory(ctx.m["prompt"])
            if mem:
                used = (th, ph)
                break
        if mem:
            break
    if mem is None:
        out.append("  RETRIEVAL: no memory at any threshold/phase -> FAIL")
        metrics["retrieval"] = "FAIL (no memory)"
        return
    cat = category(mem_clip, mem["video_ms"])
    note = "" if used[0] == 0.05 else "  [FALLBACK: 0.05 produced no memory]"
    out.append(f"  using th={used[0]} phase +{used[1]}ms{note}")
    out.append(f"  RETRIEVAL: evidence frame {mem['video_ms'] / 1000:.1f}s (labeled '{cat}'), bbox {mem['bbox']}, "
               f"conf {mem['confidence']}, memories stored {len(ls.memories(ctx.m['prompt']))}, caveats {mem['caveats']}")
    if cat is None:
        metrics["retrieval"] = f"unlabeled (evidence at {mem['video_ms'] / 1000:.1f}s, th={used[0]})"
    else:
        metrics["retrieval"] = f"{'OK' if cat in mem_clip['useful'] else 'WRONG'} ({cat}, th={used[0]})"
    ev = cv2.imread(mem["image_file"])
    x1, y1, x2, y2 = mem["bbox"]
    cv2.rectangle(ev, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.imwrite(str(ctx.out / "retrieval_evidence.jpg"), ev)

    if nav_clip:
        nav, writer = [], None
        for t, dets in ctx.rows(nav_clip["id"]):
            img = ctx.frame(nav_clip["id"], t)
            det.current = kept(dets, img.shape, 0.05)
            g = ls.guide(img, mem["memory_id"])
            nav.append((t, g))
            vis = img.copy()
            if g["remembered_box_in_view"]:
                a, b, c, d = g["remembered_box_in_view"]
                cv2.rectangle(vis, (a, b), (c, d), (0, 140, 255), 3)
                cv2.putText(vis, "remembered spot (projected)", (a, max(20, b - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
            if g["detection"]["visible"]:
                a, b, c, d = g["detection"]["bbox"]
                cv2.rectangle(vis, (a, b), (c, d), (0, 200, 0), 3)
                cv2.putText(vis, f"keys detected now {g['detection']['confidence']:.2f}", (a, min(vis.shape[0] - 8, d + 22)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
            arrow = f"{g['direction']['horizontal']}/{g['direction']['vertical']}" if g["direction"] else "-"
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 34), (255, 255, 255), -1)
            cv2.putText(vis, f"t={t / 1000:4.1f}s {g['status']} scene={g['scene_verdict']}({g['scene_inliers']}) go:{arrow}",
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            if writer is None:
                writer = cv2.VideoWriter(str(ctx.out / "navigation_preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         SAMPLE_FPS, (vis.shape[1], vis.shape[0]))
            writer.write(vis)
        if writer:
            writer.release()
        out.append("  NAVIGATION per labeled window:")
        for cat in sorted({category(nav_clip, t) for t, _ in nav} - {None}):
            sel = [g for t, g in nav if category(nav_clip, t) == cat]
            statuses = {}
            for g in sel:
                statuses[g["status"]] = statuses.get(g["status"], 0) + 1
            same = sum(g["scene_verdict"] == "same_scene" for g in sel)
            out.append(f"    {cat:<13} {len(sel):>3} frames | scene recognized {same} | {statuses}")
            if scored(cat):
                metrics["nav_scene"] = metrics.get("nav_scene", 0) + same
                metrics["nav_scored"] = metrics.get("nav_scored", 0) + len(sel)
                metrics["nav_detected"] = metrics.get("nav_detected", 0) + sum(g["detection"]["visible"] for g in sel)
        lies = sum(g["status"] in ("object_at_remembered_spot", "object_seen_elsewhere", "object_detected")
                   and not g["detection"]["visible"] for _, g in nav)
        metrics["false_visible"] = lies
        out.append(f"    frames claiming the keys are visible without a live detection: {lies} (must be 0)")

    # "found my keys" clears every memory of that object (an older one must not resurface)
    before = ls.memories(ctx.m["prompt"])
    removed = ls.found(label=ctx.m["prompt"])
    files = [Path(ls.store.out) / r[k] for r in before for k in ("image_path", "crop_path") if r.get(k)]
    gone = not any(f.exists() for f in files)
    log = [json.loads(line) for line in (store / "events.jsonl").read_text().splitlines()]
    hist = {r["memory_id"]: [e["event"] for e in log if e["memory_id"] == r["memory_id"]] for r in before}
    ok = gone and all(h == ["deleted"] for h in hist.values()) and ls.latest_memory(ctx.m["prompt"]) is None
    metrics["deletion"] = f"OK ({len(removed)} removed)" if ok else "FAIL"
    out.append(f"  FOUND IT (label-wide): {len(removed)} memories removed, {len(files)} files, all gone={gone}, "
               f"log per id {hist}, latest_memory after -> {ls.latest_memory(ctx.m['prompt'])}")
    ls.close()


def section_false_alarms(ctx, out, examples):
    out.append(f"\n## 5. Detections while the keys were labeled not visible (conf >= {min(THRESHOLDS)}): {len(examples)}")
    tiles = []
    for cid, t, b, d in sorted(examples, key=lambda x: -x[2])[:12]:
        out.append(f"   {cid} {t / 1000:.1f}s conf {b:.3f} box {list(d.bbox)}")
        img = ctx.frame(cid, t)
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(img, f"{cid} {t / 1000:.1f}s {b:.3f}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        tiles.append(cv2.resize(img, (int(img.shape[1] * 300 / img.shape[0]), 300)))
    if tiles:
        width = max(t.shape[1] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, 0, 0, width - t.shape[1], cv2.BORDER_CONSTANT) for t in tiles]
        rows = [np.hstack(tiles[i:i + 4] + [np.zeros_like(tiles[0])] * (4 - len(tiles[i:i + 4]))) for i in range(0, len(tiles), 4)]
        cv2.imwrite(str(ctx.out / "false_alarms.jpg"), np.vstack(rows))


def timeline(ctx, clip):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = ctx.rows(clip["id"])
    best = [max((x.conf for x in kept(d, ctx.shape(clip["id"]), 0)), default=0) for _, d in rows]
    fig, ax = plt.subplots(figsize=(9, 2.6))
    for start, end, cat in clip["windows"]:
        color = "#e9a0a0" if cat == "absent" else "#dddddd" if not scored(cat) else "#f4c27a" if cat == "hand" else "#8fd18f"
        ax.axvspan(start, end, color=color, alpha=0.5, lw=0)
        ax.text((start + end) / 2, 0.97, cat, ha="center", va="top", fontsize=7, transform=ax.get_xaxis_transform())
    ax.plot([t / 1000 for t, _ in rows], best, ".-", color="#1f4e8c", ms=3, lw=1)
    for th, ls in zip(THRESHOLDS, ("--", ":")):
        ax.axhline(th, color="k", ls=ls, lw=0.8, label=f"threshold {th}")
    ax.set_xlabel("seconds"), ax.set_ylabel("best conf"), ax.set_title(f"{clip['display']} ({ctx.preset})", fontsize=9)
    ax.set_ylim(0, max(0.12, max(best, default=0) * 1.1)), ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(ctx.out / f"timeline_{clip['id']}.png", dpi=110)
    plt.close(fig)


def analyze(m, preset):
    ctx = Ctx(m, preset)
    metrics = {"recall": {}, "memory": {}, "latency": {}, "dupes": {}}
    out = [f"# LastSeen evaluation: manifest '{m['name']}', preset '{preset}' {PRESETS[preset]}",
           "Ground truth = time windows labeled by eye (manifest). Clips: " + "; ".join(c["display"] for c in m["clips"])]
    if m.get("to_confirm"):
        out.append("Labels awaiting confirmation: " + "; ".join(m["to_confirm"]))
    examples = section_detection(ctx, out, metrics)
    for clip in m["clips"]:
        timeline(ctx, clip)
    section_memory(ctx, out, metrics)
    section_sampling(ctx, out)
    section_flow(ctx, out, metrics)
    section_false_alarms(ctx, out, examples)
    metrics["ms"] = statistics.median([v for v in (ctx.ms(c["id"]) for c in m["clips"]) if v is not None])
    (ctx.out / "report.txt").write_text("\n".join(out), encoding="utf-8")
    return metrics


def summary(m, all_metrics):
    clips = m["clips"]
    head = ["preset", "ms/inf"]
    head += [f"{c['id']} @0.05 (@0.03)" for c in clips]
    head += ["recall @0.05 by category", "retrieval", "nav: scene recognized / keys detected (scored frames)",
             "false 'visible' claims", "deletion"]
    lines = [f"# Summary: {m['name']}", "",
             "Memory columns: useful = useful memory before walk-away; false mem = phases (of 5) where a no-keys clip "
             "produced a memory; formed = phases with any memory. All at 2 inferences/s, scene matching on.", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for preset, mt in all_metrics.items():
        cells = [preset, f"{mt['ms']:.0f}"]
        for c in clips:
            a, b = mt["memory"].get((c["id"], 0.05), "-"), mt["memory"].get((c["id"], 0.03), "-")
            cells.append(f"{a} ({b.split()[-1] if b != '-' else '-'})")
        rec = {}
        for (cid, cat), (h05, _, n) in mt["recall"].items():
            if scored(cat):
                r = rec.setdefault(cat, [0, 0])
                r[0] += h05
                r[1] += n
        cells.append(", ".join(f"{cat} {100 * h // n}%" for cat, (h, n) in sorted(rec.items())))
        cells.append(mt.get("retrieval", "-"))
        cells.append(f"{mt.get('nav_scene', 0)} / {mt.get('nav_detected', 0)} of {mt.get('nav_scored', 0)}")
        cells.append(str(mt.get("false_visible", "-")))
        cells.append(mt.get("deletion", "-"))
        lines.append("| " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    (m["root"] / "summary.md").write_text(text, encoding="utf-8")
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--presets", default="efficient", help="comma-separated names from memory_engine/presets.py")
    p.add_argument("--stage", choices=["all", "detect", "analyze"], default="all")
    p.add_argument("--redo", action="store_true", help="re-run detection even if cached")
    args = p.parse_args()
    m = load_manifest(args.manifest)
    presets = [s.strip() for s in args.presets.split(",") if s.strip()]
    unknown = [s for s in presets if s not in PRESETS]
    if unknown:
        raise SystemExit(f"unknown preset(s) {unknown}; choose from {list(PRESETS)}")
    if not m["clips"]:
        raise SystemExit("no clip files found")
    if args.stage in ("all", "detect"):
        print(f"stage 1: detection cache -> {m['root']}")
        stage_detect(m, presets, args.redo)
    if args.stage in ("all", "analyze"):
        print("stage 2: analysis")
        all_metrics = {p_: analyze(m, p_) for p_ in presets}
        print(summary(m, all_metrics))
        print(f"reports: {m['root'] / 'reports'}  summary: {m['root'] / 'summary.md'}")


if __name__ == "__main__":
    main()
