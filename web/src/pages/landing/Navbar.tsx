// The signed-out header: logo on the left; on desktop, section links, Display settings, Log in
// and Get started; on phones, a menu button that opens the same list.
import { useEffect, useId, useRef, useState } from "react";
import { DisplayButton } from "../../components/DisplaySettings";
import { BrandMark, Icon } from "../../components/Icons";
import { useAuth } from "../../lib/auth";
import { SECTION_LINKS, sectionLink } from "./sections";

export function Navbar() {
  const { login, signup } = useAuth();
  const [open, setOpen] = useState(false);
  const menuId = useId();
  const toggleRef = useRef<HTMLButtonElement>(null);
  const close = () => setOpen(false);

  // Escape closes the menu and gives focus back to its button; so does a wider window.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setOpen(false);
        toggleRef.current?.focus();
      }
    };
    const wide = matchMedia("(min-width: 768px)");
    const onWide = () => wide.matches && setOpen(false);
    window.addEventListener("keydown", onKey);
    wide.addEventListener("change", onWide);
    return () => {
      window.removeEventListener("keydown", onKey);
      wide.removeEventListener("change", onWide);
    };
  }, [open]);

  return (
    <header className="site-header landing-nav" data-open={open || undefined}>
      <a className="brand" href="#/" aria-label="LastScene home" onClick={close}>
        <BrandMark />
      </a>

      <nav className="landing-links" aria-label="Main">
        {SECTION_LINKS.map((l) => (
          <a key={l.id} className="nav-link" href={`#${l.id}`} onClick={sectionLink(l.id)}>
            {l.label}
          </a>
        ))}
        <a className="nav-link" href="#/public">
          Public map
        </a>
      </nav>

      <div className="landing-actions">
        <DisplayButton />
        <button id="login-header" type="button" className="nav-link nav-button" onClick={login}>
          Log in
        </button>
        <button type="button" className="primary nav-cta" onClick={signup}>
          Get started
        </button>
      </div>

      <button
        ref={toggleRef}
        type="button"
        className="icon-btn menu-toggle"
        aria-expanded={open}
        aria-controls={menuId}
        aria-label={open ? "Close menu" : "Open menu"}
        onClick={() => setOpen((o) => !o)}
      >
        <Icon name={open ? "close" : "menu"} />
      </button>

      <div id={menuId} className="mobile-menu" hidden={!open}>
        <nav aria-label="Main (mobile)">
          {SECTION_LINKS.map((l) => (
            <a key={l.id} href={`#${l.id}`} onClick={sectionLink(l.id, close)}>
              {l.label}
            </a>
          ))}
          <a href="#/public" onClick={close}>
            Public map
          </a>
        </nav>
        <div className="mobile-menu-actions">
          <button type="button" className="primary" onClick={signup}>
            Get started
          </button>
          <button type="button" className="pill" onClick={login}>
            Log in
          </button>
          <DisplayButton className="pill" withLabel />
        </div>
      </div>
    </header>
  );
}
