// In-page links for the signed-out pages. The URL hash drives page routing, so "#features"
// can't be a plain anchor: these scroll to the section (and move focus there) without
// touching the hash, coming back to the landing page first when needed.
import { prefersReducedMotion } from "../../lib/format";

export type SectionId = "features" | "privacy" | "get-started";

export const SECTION_LINKS: { id: SectionId; label: string }[] = [
  { id: "features", label: "How it works" },
  { id: "privacy", label: "Privacy" },
];

function reveal(id: SectionId): boolean {
  const target = document.getElementById(id);
  if (!target) return false;
  target.scrollIntoView({ behavior: prefersReducedMotion() ? "auto" : "smooth", block: "start" });
  target.focus({ preventScroll: true }); // sections carry tabIndex={-1}
  return true;
}

export function goToSection(id: SectionId) {
  if (reveal(id)) return;
  window.location.hash = "#/"; // another page (the public map): back to the landing page
  let tries = 0;
  const retry = () => {
    if (!reveal(id) && ++tries < 20) requestAnimationFrame(retry);
  };
  requestAnimationFrame(retry);
}

/** onClick for an <a href="#id">: keeps the link semantics, skips the hash change. */
export const sectionLink = (id: SectionId, after?: () => void) => (e: { preventDefault(): void }) => {
  e.preventDefault();
  after?.();
  goToSection(id);
};
