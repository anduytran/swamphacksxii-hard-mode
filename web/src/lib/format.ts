// Shared text formatting for times, names and counts.

export function timeAgo(iso: string, now = Date.now()): string {
  const minutes = Math.round((now - new Date(iso).getTime()) / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} hour${hours === 1 ? "" : "s"} ago`;
  const days = Math.round(hours / 24);
  return `${days} day${days === 1 ? "" : "s"} ago`;
}

export function shortAgo(iso: string, now = Date.now()): string {
  const minutes = Math.round((now - new Date(iso).getTime()) / 60000);
  if (minutes < 60) return `${Math.max(minutes, 1)}m ago`;
  if (minutes < 60 * 24) return `${Math.round(minutes / 60)}h ago`;
  return `${Math.round(minutes / 1440)}d ago`;
}

export function clockTime(iso: string): string {
  return new Date(iso).toLocaleString(undefined, { weekday: "short", hour: "numeric", minute: "2-digit" });
}

export const capitalize = (s: string | null | undefined): string => (s ? s[0].toUpperCase() + s.slice(1) : "");

/** "Riley Chen" → "RC", "sam@example.com" → "S" */
export function initials(name: string): string {
  const words = name.replace(/@.*/, "").split(/[\s._-]+/).filter(Boolean);
  return words.slice(0, 2).map((w) => w[0].toUpperCase()).join("") || "?";
}

export const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

/** Counts timestamped events into N equal buckets of the viewer's local day. */
export function bucketByHour(events: { timestamp: string }[], buckets = 8): number[] {
  const size = 24 / buckets;
  const counts = Array<number>(buckets).fill(0);
  for (const e of events) counts[Math.floor(new Date(e.timestamp).getHours() / size)] += 1;
  return counts;
}

/** Animate? Honours both the device setting and LastScene's own Display > Motion choice. */
export { reducedMotion as prefersReducedMotion } from "./display";
