// The frame around every page: header (a sidebar on wide screens, a top bar + bottom tab bar
// on phones), the status line, and the phone "More" sheet.
import { useEffect, useState } from "react";
import { useApi } from "../lib/api";
import { useApp, type Page } from "../lib/app";
import { useAuth } from "../lib/auth";
import { publishReviewCount, useReviewCount } from "../lib/reviewCount";
import type { Household, ReviewQueue } from "../lib/types";
import { initials } from "../lib/format";
import { Dialog } from "./Dialog";
import { DisplayButton } from "./DisplaySettings";
import { Navbar } from "../pages/landing/Navbar";
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
    items: [
      { page: "household", label: "Household", icon: "household", secondary: true, blurb: "Wearable, people and places" },
      { page: "teach", label: "Teach items", icon: "sparkle", secondary: true, blurb: "Help the camera spot your things" },
      { page: "live", label: "Live", icon: "mic", secondary: true, blurb: "Each voice command as it happens" },
    ],
  },
];

/** Keeps the "needs a look" count fresh for the Teach badge: on sign-in, on Refresh, and on each page change. */
function useReviewBadge(signedIn: boolean, page: Page): number {
  const api = useApi();
  const { refreshNonce } = useApp();
  useEffect(() => {
    if (!signedIn || page === "teach") return; // Teach publishes its own count
    let gone = false;
    api<{ households: Household[] }>("/api/v1/households/me")
      .then(({ households }) =>
        Promise.all(
          households.map((h) =>
            api<ReviewQueue>(`/api/v1/households/${h.id}/training/review?limit=0`).then((q) => !gone && publishReviewCount(h.id, q.count)),
          ),
        ),
      )
      .catch(() => {}); // a badge is a nicety: never an error on the page
    return () => {
      gone = true;
    };
  }, [api, signedIn, page, refreshNonce]);
  return useReviewCount();
}

function Badge({ count }: { count: number }) {
  if (!count) return null;
  return (
    <span className="nav-badge" aria-label={`, ${count} photo${count === 1 ? " needs" : "s need"} a look`}>
      {count > 99 ? "99+" : count}
    </span>
  );
}

export function Header({ page }: { page: Page }) {
  const { signedIn, user, logout } = useAuth();
  const { refresh } = useApp();
  const [moreOpen, setMoreOpen] = useState(false);
  const toReview = useReviewBadge(signedIn, page);
  const badges: Partial<Record<Page, number>> = { teach: toReview };
  const name = user?.name || user?.email || "";
  const current = (p: Page) => (p === page ? ("page" as const) : undefined);

  if (!signedIn) return <Navbar />;

  return (
    <header className="site-header">
      <a className="brand" href="#/items" aria-label="LastScene home">
        <BrandMark />
        <span>LastScene</span>
      </a>

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
                    <Badge count={badges[n.page] ?? 0} />
                  </a>
                ))}
              </div>
            ))}
            <button type="button" className="more-tab" aria-haspopup="dialog" aria-controls="more-sheet" onClick={() => setMoreOpen(true)}>
              <Icon name="more" />
              More
              {toReview > 0 && <span className="nav-dot" aria-hidden="true" />}
            </button>
          </nav>

          <div className="user-bar">
            <span className="avatar" aria-hidden="true">
              {initials(name)}
            </span>
            <span className="user-name">{name}</span>
            <DisplayButton />
            <button id="refresh" type="button" className="icon-btn" aria-label="Refresh this page" title="Refresh" onClick={refresh}>
              <Icon name="refresh" />
            </button>
            <button id="logout" type="button" className="icon-btn" aria-label="Log out" title="Log out" onClick={logout}>
              <Icon name="logout" />
            </button>
          </div>

          <MoreSheet open={moreOpen} onClose={() => setMoreOpen(false)} page={page} badges={badges} />
        </>
      )}
    </header>
  );
}

function MoreSheet({ open, onClose, page, badges }: { open: boolean; onClose(): void; page: Page; badges: Partial<Record<Page, number>> }) {
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
              <strong>
                {n.label}
                <Badge count={badges[n.page] ?? 0} />
              </strong>
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
        <DisplayButton className="pill" withLabel />
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
