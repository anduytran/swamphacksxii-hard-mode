// Display settings: text size, contrast, theme and motion, chosen per browser (kept in
// localStorage) and applied as data attributes on <html> that styles.css reads. index.html
// applies the saved choice before the first paint, so a large-text page never flashes small.

export type TextSize = "standard" | "large" | "largest";
export type Contrast = "standard" | "high";
export type Theme = "system" | "light" | "dark";
export type Motion = "system" | "reduce";

export interface Display {
  text: TextSize;
  contrast: Contrast;
  theme: Theme;
  motion: Motion;
}

export const DEFAULT_DISPLAY: Display = { text: "standard", contrast: "standard", theme: "system", motion: "system" };
export const STORAGE_KEY = "lastscene-display";

const CHOICES: { [K in keyof Display]: readonly Display[K][] } = {
  text: ["standard", "large", "largest"],
  contrast: ["standard", "high"],
  theme: ["system", "light", "dark"],
  motion: ["system", "reduce"],
};

/** The saved settings, ignoring anything unknown (an old or hand-edited value). */
export function loadDisplay(): Display {
  let saved: Partial<Record<keyof Display, unknown>> = {};
  try {
    saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}") ?? {};
  } catch {
    /* blocked or corrupt storage: defaults */
  }
  const pick = <K extends keyof Display>(k: K): Display[K] =>
    (CHOICES[k] as readonly unknown[]).includes(saved[k]) ? (saved[k] as Display[K]) : DEFAULT_DISPLAY[k];
  return { text: pick("text"), contrast: pick("contrast"), theme: pick("theme"), motion: pick("motion") };
}

export function saveDisplay(d: Display) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(d));
  } catch {
    /* still applied for this visit */
  }
}

const darkQuery = () => (typeof matchMedia === "function" ? matchMedia("(prefers-color-scheme: dark)") : null);
const motionQuery = () => (typeof matchMedia === "function" ? matchMedia("(prefers-reduced-motion: reduce)") : null);

/** Light or dark, after "system" is resolved against the device. */
export const resolvedTheme = (d: Display): "light" | "dark" =>
  d.theme === "system" ? (darkQuery()?.matches ? "dark" : "light") : d.theme;

export function applyDisplay(d: Display, root: HTMLElement = document.documentElement) {
  root.dataset.text = d.text;
  root.dataset.contrast = d.contrast;
  root.dataset.theme = resolvedTheme(d);
  root.dataset.motion = d.motion === "reduce" || motionQuery()?.matches ? "reduce" : "system";
  root.style.colorScheme = resolvedTheme(d);
}

/** Re-apply when the device switches light/dark or motion while "system" is chosen. */
export function followDevice(get: () => Display): () => void {
  const queries = [darkQuery(), motionQuery()].filter((q): q is MediaQueryList => q !== null);
  const update = () => applyDisplay(get());
  queries.forEach((q) => q.addEventListener("change", update));
  return () => queries.forEach((q) => q.removeEventListener("change", update));
}

/** For code that animates (map fly-tos, smooth scrolling): honours the setting and the device. */
export const reducedMotion = () => document.documentElement.dataset.motion === "reduce" || !!motionQuery()?.matches;
