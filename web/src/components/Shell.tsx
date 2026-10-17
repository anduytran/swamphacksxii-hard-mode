// The frame around every page: header (a sidebar on wide screens, a top bar + bottom tab bar
// on phones), the status line, and the phone "More" sheet.
import { useState } from "react";
import { useApp, type Page } from "../lib/app";
import { useAuth } from "../lib/auth";
import { initials } from "../lib/format";
import { Dialog } from "./Dialog";
import { BrandMark, Icon, type IconName } from "./Icons";

interface NavItem {
  page: Page;
  label: string;
  icon: IconName;
  secondary?: boolean; // behind "More" on phones
  blurb?: string; // shown in the More sheet
}

const NAV: { group: string; items: NavItem[] }[] = [
  {
    group: "Today",
    items: [
      { page: "items", label: "Items", icon: "items" },
      { page: "timeline", label: "Timeline", icon: "timeline" },
      { page: "map", label: "Map", icon: "map" },
    ],
  },
  {
    group: "Understand",
    items: [
      { page: "insights", label: "Insights", icon: "insights" },
      { page: "search", label: "Search", icon: "search", secondary: true, blurb: "Ask what the wearer would" },
      { page: "public", label: "Public map", icon: "public", secondary: true, blurb: "Where people lose things" },
    ],
  },
  {
    group: "Setup",
    items: [{ page: "household", label: "Household", icon: "household", secondary: true, blurb: "Wearable, family and places" }],
  },
];

export function Header({ page }: { page: Page }) {
  const { signedIn, user, login, logout } = useAuth();
  const { refresh } = useApp();
  const [moreOpen, setMoreOpen] = useState(false);
  const name = user?.name || user?.email || "";
  const current = (p: Page) => (p === page ? ("page" as const) : undefined);

  return (
    <header className="site-header">
      <a className="brand" href="#/items" aria-label="LastScene home">
        <BrandMark />
        <span>LastScene</span>
      </a>

      {!signedIn && (
        <div className="guest-actions">
          <a className="nav-link" href="#/public">
            Public map
          </a>
          <button id="login-header" type="button" className="pill" onClick={login}>
            Log in
          </button>
        </div>
      )}

      {signedIn && (
        <>
          <nav id="tabs" className="tabs" aria-label="Dashboard pages">
            {NAV.map(({ group, items }) => (
              <div className="nav-group" key={group}>
                <span className="nav-label">{group}</span>
                {items.map((n) => (
                  <a key={n.page} href={`#/${n.page}`} aria-current={current(n.page)} className={n.secondary ? "nav-secondary" : undefined}>
                    <Icon name={n.icon} />
                    {n.label}
                  </a>
                ))}
              </div>
            ))}
            <button type="button" className="more-tab" aria-haspopup="dialog" aria-controls="more-sheet" onClick={() => setMoreOpen(true)}>
              <Icon name="more" />
              More
            </button>
          </nav>

          <div className="user-bar">
            <span className="avatar" aria-hidden="true">
              {initials(name)}
            </span>
            <span className="user-name">{name}</span>
            <button id="refresh" type="button" className="icon-btn" aria-label="Refresh this page" title="Refresh" onClick={refresh}>
              <Icon name="refresh" />
            </button>
            <button id="logout" type="button" className="icon-btn" aria-label="Log out" title="Log out" onClick={logout}>
              <Icon name="logout" />
            </button>
          </div>

          <MoreSheet open={moreOpen} onClose={() => setMoreOpen(false)} page={page} />
        </>
      )}
    </header>
  );
}

function MoreSheet({ open, onClose, page }: { open: boolean; onClose(): void; page: Page }) {
  const { refresh } = useApp();
  const { logout } = useAuth();
  const secondary = NAV.flatMap((g) => g.items).filter((n) => n.secondary);
  return (
    <Dialog open={open} onClose={onClose} className="sheet" labelledBy="more-title">
      <div className="sheet-head">
        <h2 id="more-title">More</h2>
        <button type="button" className="icon-btn" data-close aria-label="Close menu">
          <Icon name="close" />
        </button>
      </div>
      <nav className="sheet-links" aria-label="More pages">
        {secondary.map((n) => (
          <a key={n.page} href={`#/${n.page}`} aria-current={n.page === page ? "page" : undefined} onClick={onClose}>
            <Icon name={n.icon} />
            <span>
              <strong>{n.label}</strong>
              <small>{n.blurb}</small>
            </span>
          </a>
        ))}
      </nav>
      <div className="sheet-actions">
        <button
          type="button"
          className="pill"
          onClick={() => {
            onClose();
            refresh();
          }}
        >
          <Icon name="refresh" />
          Refresh
        </button>
        <button type="button" className="pill ghost" onClick={logout}>
          <Icon name="logout" />
          Log out
        </button>
      </div>
    </Dialog>
  );
}

/** The status line under the header: a spinner pill while loading, a red notice with Try again on errors. */
export function StatusLine() {
  const { status } = useApp();
  return (
    <p id="status" role="status" data-state={status.state || undefined}>
      {status.text && <span>{status.text}</span>}
      {status.retry && (
        <button type="button" className="pill" onClick={status.retry}>
          Try again
        </button>
      )}
    </p>
  );
}
