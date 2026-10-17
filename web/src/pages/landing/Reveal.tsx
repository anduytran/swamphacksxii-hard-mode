// A section that fades up as it scrolls into view. Content is never hidden without JavaScript,
// without IntersectionObserver, or with reduced motion: only a section the observer is waiting
// on starts faded, and it shows the moment any of it is on screen.
import { useEffect, useRef, useState, type ReactNode } from "react";
import { prefersReducedMotion } from "../../lib/format";

interface RevealProps {
  id?: string;
  className?: string;
  labelledBy?: string;
  children: ReactNode;
}

export function RevealSection({ id, className, labelledBy, children }: RevealProps) {
  const ref = useRef<HTMLElement>(null);
  const [state, setState] = useState<"pending" | "shown" | undefined>(undefined);

  useEffect(() => {
    const el = ref.current;
    if (!el || typeof IntersectionObserver !== "function" || prefersReducedMotion()) return;
    const box = el.getBoundingClientRect();
    if (box.top < window.innerHeight && box.bottom > 0) return; // already on screen: no fade
    setState("pending");
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setState("shown");
          observer.disconnect();
        }
      },
      { rootMargin: "0px 0px -8% 0px" },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <section ref={ref} id={id} className={className} aria-labelledby={labelledBy} tabIndex={id ? -1 : undefined} data-reveal={state}>
      {children}
    </section>
  );
}
