// How many Teach photos need a look, per household, for the badge on the Teach nav item. The Teach page
// publishes it whenever its queue loads; the header fetches it on its own too, so the badge shows before
// anyone opens Teach.
import { useSyncExternalStore } from "react";

const counts = new Map<string, number>();
const listeners = new Set<() => void>();
let total = 0;

export function publishReviewCount(householdId: string, count: number) {
  counts.set(householdId, count);
  const next = [...counts.values()].reduce((a, b) => a + b, 0);
  if (next === total) return;
  total = next;
  listeners.forEach((l) => l());
}

export function useReviewCount(): number {
  return useSyncExternalStore(
    (l) => (listeners.add(l), () => listeners.delete(l)),
    () => total,
  );
}
