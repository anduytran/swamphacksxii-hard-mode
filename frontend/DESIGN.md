# LastScene dashboard: UI specification

High-fidelity, responsive spec for the caregiver dashboard as built in `index.html`, `styles.css`, `app.js` and `charts.js`. Change the code and this file together.

## 1. Four layers

| Layer | Definition |
|---|---|
| **Context** | Web dashboard (also used on phones) for family members who look after someone with memory loss. They check where things were left, what keeps going missing, and set up the wearable. Often older, often on a phone, often in a hurry. |
| **Structure** | Desktop ≥ 961px: 248px sidebar + content column, max 1320px, 48px side padding, 16px gutters. Tablet 641–960px and phone ≤ 640px: sticky top bar + fixed bottom tab bar, single column, 16–20px side padding. Grids are `auto-fill` / `auto-fit` (e.g. item cards `minmax(250px, 1fr)`: 3–4 columns desktop, 2 tablet, 1 phone). |
| **Constraints** | Dark only ("Darkroom"). No box shadows: depth comes from layered surfaces and 1px borders. No stock photography: real camera frames, plus line-art and geometric illustration. No placeholder-only labels. Every interactive element ≥ 44×44px. Text ≥ 4.5:1 contrast. Chart colours come only from the validated series pair, never from status colours. |
| **Output** | Responsive, production code. This file is the reference. |

## 2. Screens and flows

| Screen | Route | Purpose | Empty state | Loading | Error |
|---|---|---|---|---|---|
| Landing | (signed out) | What it is, Log in | – | – | Setup error line under the hero |
| Items | `#/items` | Latest scene + one card per item | Setup checklist + "Check again" | Featured + 6 card skeletons | Status notice + Try again |
| Timeline | `#/timeline` | Every memory in 24 h, newest first | Setup checklist + "Check again" | 4 scene skeletons | Status notice + Try again |
| Map | `#/map` | House plan (deck.gl) / Our places (OSM) | Setup checklist in the map frame | Shimmering map frame | Status notice + Try again |
| Insights | `#/insights` | KPIs, summary, rankings, activity, questions | Charts say what's missing | Tile, bar, chart and list skeletons | Status notice + Try again |
| Search | `#/search` | Ask what the wearer would | "No memories matched" + "Try “keys”" | 2 result skeletons + button spinner | Status notice + Try again (resubmits) |
| Public map | `#/public` | Anonymous lost-item map (no login): 3D OpenStreetMap buildings (OpenFreeMap, no key), slim cylinders (24 m across, 30–120 m tall) per ~100 m spot; opens on the busiest spot, each "Busiest spots" row flies there | "Nothing public yet" + share/log-in CTA | Map shimmer + row skeletons | Status notice + Try again |
| Household | `#/household` | Create/join, add wearable, invite, places, sharing | Create / Join forms | Two form-card skeletons | Inline field errors; form message |
| Item detail | drawer | Drill-down for one item | "Never logged" + tip | Photo + line skeletons | Message + Try again |

**Journeys**
1. *First visit:* Landing → Log in (Auth0) → no household → Household: create → Add a wearable (Client ID + pairing code) → Items.
2. *"Where are the keys?":* Items → filter "keys" or tap the card → detail drawer (photo + box, where, when, where it's been, questions) → "Show on the home map".
3. *Spotting a pattern:* Insights → range 24 h / 2 d / 3 d → tap a bar → detail drawer.
4. *Family member:* Household → Join with an invite code.

**Navigation.** Desktop: sidebar grouped *Today* (Items, Timeline, Map), *Understand* (Insights, Search, Public map), *Setup* (Household), with the user, Refresh and Log out at the bottom. Phone/tablet: bottom tab bar with Items · Timeline · Map · Insights · More. "More" opens a bottom sheet with Search, Public map, Household, Refresh and Log out, and shows as active when one of those pages is open. Signed out on a phone, Log in is a full-width bar at the bottom.

**Responsive rules.** Featured item: photo | text (1.2 : 1) → stacked ≤ 960px. Timeline: time | rail | card → stacked without the rail ≤ 640px. Search results: photo | text → stacked ≤ 640px. Landing hero: copy | scene card → stacked ≤ 960px. Item drawer: right panel (460px) → bottom sheet (≤ 88% height) ≤ 640px.

## 3. Visual style

Mood: calm, warm and quiet: a darkroom with a safelight. Amber means "this matters / where it is now"; everything else stays neutral.

| Token | Value | Use |
|---|---|---|
| `--ground` | `#0e0d0c` | Page |
| `--surface` / `-2` / `-3` | `#171614` / `#1f1d1a` / `#292723` | Cards → raised controls → selected/pressed |
| `--line` / `-strong` / `-hi` | warm white at 7% / 12% / 20% | Card borders → control borders → hover borders |
| `--text` / `--text-2` / `--muted` / `--faint` | `#f3efe8` / `#d2cbbf` / `#a8a196` / `#918a7f` | ≥ 4.9:1 on every text surface |
| `--amber` | `#f4a73b` | Accent, primary action, focus ring |
| `--series-1` / `-2` | `#3987e5` / `#c98500` | Chart series (validated pair, CVD-safe) |
| good / warn / danger | `#8fd6a5` / `#f1c46b` / `#ff8a7a` on tinted grounds | Status, always with an icon or label |

**Type**

| Role | Font | Size / weight / line height |
|---|---|---|
| H1 page title | Bricolage Grotesque | 32–48px fluid, 700, 1.02 |
| Landing title | Bricolage Grotesque | 46–84px fluid, 700, 0.95 |
| H2 card title | Geist | 20px, 600, 1.3 |
| H3 | Geist / Bricolage | 16–21px, 600–700 |
| Body | Geist | 16px, 400, 1.55 |
| Small / hint | Geist | 14px, 400 |
| Eyebrow / time / count | Geist Mono | 12px, 500, uppercase eyebrows |

Illustration: the landing scene card is an inline SVG (line-art counter, keys, mug, plant) with a detection box. Icons are 1.8px stroke line icons.

## 4. Dashboard data → weight

| Data (`/dashboard/insights?hours=`) | Visual | Weight |
|---|---|---|
| memories_logged, questions_asked, unanswered_count, items_tracked | KPI tiles; "never logged" turns amber when > 0 | Full width, top |
| Gemini summary | Bordered summary card | Full width |
| Highlights | "Worth a conversation" notice | Full width |
| most_asked / most_logged | Ranked bar lists, rows open the item drawer | Two 50% columns (stack < 680px) |
| activity | Grouped columns by 3-hour block, legend + sortable table | Full width |
| recent_questions | List with answerable badges | Full width |

**Interaction.** Hover or focus on any bar or column shows a tooltip with the exact value. Bar rows and item cards are buttons that open the drill-down drawer. The Insights range (24 h / 2 d / 3 d) is a segmented control that re-queries the API, capped at the 72 h retention. Every chart has a "Show as table" view with sortable columns (`aria-sort`).

## 5. Mobile rules

- Targets ≥ 44×44 (tab bar items 56px tall; segmented buttons, pills, chips, icon buttons and summaries 44px).
- Thumb zone: navigation and the landing Log in sit in the bottom bar. Secondary actions (Refresh, Log out, secondary pages) sit behind More.
- Native patterns: bottom tab bar, large titles, modal bottom sheets (`<dialog>`, Esc and backdrop tap close them).
- Keyboard: `scroll-padding-bottom` keeps a focused field above the tab bar. Submit stays disabled until the form is valid.

## 6. Accessibility

- Skip-to-content link is the first focusable element; `main` takes focus.
- Focus ring: 2px solid amber, 2px offset, on every control. Cards show the ring around the whole card.
- Persistent visible labels, `label for`, inline errors via `aria-describedby` + `aria-invalid`, an "(optional)" marker.
- `role="status"` for loading, results count and errors; icon-only buttons carry `aria-label`; dialogs are labelled by their title; skeletons are `aria-hidden`.
- Alt text describes the item and place, adds detected objects, and quotes what was said for search results.
- Reduced motion turns off every animation and shimmer.

## 7. Components

| Component | Variants | States |
|---|---|---|
| Button | `.primary` (amber), `.pill` (secondary), `.ghost`, `.danger`, `.icon-btn` | default · hover (lighter surface/border) · active (pressed surface) · focus-visible (ring) · disabled (45% opacity, not-allowed) · loading (`aria-busy`: inline spinner, disabled) |
| Segmented control | – | default · hover · pressed (`aria-pressed`, raised surface + border) |
| Chip | neutral, `.warn`, `.auto` (info), `.place` | static |
| Card | `.card`, `.item-card`, `.featured`, `.tile`, `.tile.alert`, `.summary` | default · hover (border, raised surface) · focus-within (ring) |
| Field | text, search, select, checkbox toggle | default · hover · focus · invalid (danger border + message) · disabled |
| Empty state | plain line, rich (title, text, setup checklist, CTA) | – |
| Skeleton | line, block, card shapes per page | shimmer (static with reduced motion) |
| Dialog | bottom sheet (`.sheet`), drawer (`.drawer`) | open (slide in) · closed |
| Status | loading (spinner pill), error (notice + Try again), info | – |
