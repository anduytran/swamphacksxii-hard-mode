import { describe, expect, it } from "vitest";
import { orderedSteps, timelineSegments } from "../pages/Live";
import type { VoiceEvent } from "../lib/types";

const base: VoiceEvent = {
  id: 1, device_id: "d", created_at: "2026-10-17T23:00:00Z", kind: "wake", outcome: "answered", photos: 4,
  steps: { transcribe: 400, decide: 700, speak: 500 }, served: {}, total_ms: 1700,
};

describe("timelineSegments", () => {
  it("splits wake-to-reply into the wearable, network, server steps and playback", () => {
    const segs = timelineSegments({ ...base, timeline: { speech_end: 2.0, photos_ready: 2.3, sent: 2.4, reply: 4.6, played: 5.6 } });
    expect(segs.map((s) => s.label)).toEqual([
      "listening", "last photo", "preparing", "sending", "hearing", "understanding", "speaking", "processing", "playing reply",
    ]);
    const total = segs.reduce((a, s) => a + s.s, 0);
    expect(total).toBeCloseTo(5.6, 5); // wake to end of reply
    expect(segs.find((s) => s.label === "sending")!.s).toBeCloseTo(0.5, 5); // 2.2 s round trip - 1.7 s server
  });

  it("shows what's known before the device's timeline arrives", () => {
    const segs = timelineSegments({ ...base, device: { recorded_s: 2.0 } });
    expect(segs[0]).toEqual({ label: "listening", s: 2.0, group: "wearable" });
    expect(segs.some((s) => s.group === "network")).toBe(false);
  });
});

describe("orderedSteps", () => {
  it("puts steps in the order they run, whatever order the database returns", () => {
    expect(orderedSteps({ save: 7, speak: 1, photos: 4010, transcribe: 598, decide: 740 }).map(([k]) => k)).toEqual([
      "transcribe", "decide", "photos", "save", "speak",
    ]);
  });
});
