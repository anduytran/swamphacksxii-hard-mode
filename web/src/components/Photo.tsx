// A memory's photo, with the vision pipeline's detection boxes drawn over it.
import { useState, type ReactNode } from "react";
import type { Detection } from "../lib/types";

interface Props {
  url: string | null;
  alt: string;
  boxes?: Detection[];
  className?: string;
  children?: ReactNode; // overlays such as the "Latest scene" tag
}

export function Photo({ url, alt, boxes = [], className = "", children }: Props) {
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);
  if (!url) {
    return (
      <div className={`photo empty-photo ${className}`.trim()}>
        {children}
      </div>
    );
  }
  const detected = boxes.length ? ` Detected: ${boxes.map((b) => b.label).join(", ")}.` : "";
  return (
    <div className={`photo ${className}`.trim()}>
      {children}
      <img src={url} alt={alt + detected} loading="lazy" onLoad={(e) => setSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })} />
      {size && boxes.length > 0 && <Boxes boxes={boxes} w={size.w} h={size.h} />}
    </div>
  );
}

// The SVG uses the image's own pixel size with "slice", so it crops exactly like the photo's
// object-fit: cover.
function Boxes({ boxes, w, h }: { boxes: Detection[]; w: number; h: number }) {
  return (
    <svg className="boxes" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="xMidYMid slice" aria-hidden="true">
      {boxes.map((b, i) => {
        const [x, y, bw, bh] = b.bbox;
        return (
          <g key={i}>
            <rect
              className={b.maybe ? "maybe" : undefined}
              x={x * w}
              y={y * h}
              width={bw * w}
              height={bh * h}
              rx={Math.min(w, h) * 0.01}
              vectorEffect="non-scaling-stroke"
            />
            <text x={x * w} y={Math.max(y * h - h * 0.015, h * 0.05)} fontSize={Math.round(h * 0.045)}>
              {`${b.label}${b.maybe ? "?" : ""} ${Math.round(b.confidence * 100)}%`}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
