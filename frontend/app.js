let auth0Client;
let audience;

const $ = (id) => document.getElementById(id);
const PAGES = ["items", "map", "timeline", "insights", "search", "public", "household"];
// Pages anyone can open without logging in.
const PUBLIC_PAGES = ["public"];
let signedIn = false;
const DEFAULT_PAGE = "items";

async function init() {
  const config = await fetch("/config.json").then((r) => r.json());
  audience = config.audience;
  auth0Client = await auth0.createAuth0Client({
    domain: config.domain,
    clientId: config.clientId,
    authorizationParams: { audience, redirect_uri: window.location.origin },
    // Keep the session across page reloads. Silent re-auth via iframe is blocked by most
    // browsers' third-party cookie rules, so without this every reload meant a new login.
    cacheLocation: "localstorage",
    // Don't leave a blank page for the SDK's 60s default if the silent session check stalls.
    authorizeTimeoutInSeconds: 5,
  });

  // Returning from Universal Login.
  const params = new URLSearchParams(window.location.search);
  if (params.has("error")) {
    window.history.replaceState({}, document.title, window.location.pathname);
    throw new Error(params.get("error_description") || params.get("error"));
  }
  if (params.has("code") && params.has("state")) {
    const { appState } = await auth0Client.handleRedirectCallback();
    window.history.replaceState({}, document.title, window.location.pathname + (appState?.hash || ""));
  }

  const login = () => auth0Client.loginWithRedirect({ appState: { hash: window.location.hash } });
  const logout = () => auth0Client.logout({ logoutParams: { returnTo: window.location.origin } });
  const refresh = () => showPage(currentPage(), { force: true });
  for (const id of ["login", "login-header", "login-sticky"]) $(id).onclick = login;
  $("logout").onclick = logout;
  $("refresh").onclick = refresh;
  $("item-filter").oninput = filterItems;
  $("search-form").onsubmit = runSearch;
  // The search button stays disabled until there is a question to ask.
  const syncSearch = () => ($("search-submit").disabled = !$("search-q").value.trim());
  $("search-q").addEventListener("input", syncSearch);
  for (const btn of document.querySelectorAll(".suggestion")) {
    btn.onclick = () => {
      $("search-q").value = btn.textContent;
      syncSearch();
      $("search-form").requestSubmit();
    };
  }
  for (const btn of document.querySelectorAll("#insights-range [data-hours]")) {
    btn.onclick = () => setInsightsRange(Number(btn.dataset.hours));
  }

  // Phones: secondary pages and account actions live in the "More" bottom sheet.
  $("more-btn").onclick = () => $("more-sheet").showModal();
  $("sheet-refresh").onclick = () => ($("more-sheet").close(), refresh());
  $("sheet-logout").onclick = logout;
  for (const a of document.querySelectorAll("#more-sheet a")) a.addEventListener("click", () => $("more-sheet").close());
  // Dialogs close from their close button or a tap on the backdrop.
  for (const dialog of document.querySelectorAll("dialog")) {
    dialog.addEventListener("click", (e) => {
      if (e.target === dialog || e.target.closest("[data-close]")) dialog.close();
    });
  }

  for (const btn of document.querySelectorAll(".segmented [data-mode]")) btn.onclick = () => setMapMode(btn.dataset.mode);
  window.addEventListener("hashchange", () => route());

  signedIn = await auth0Client.isAuthenticated();
  document.body.classList.toggle("signed-in", signedIn);
  $("guest-actions").hidden = signedIn;
  if (signedIn) {
    const user = await auth0Client.getUser();
    currentUser = user;
    const name = user?.name || user?.email || "";
    $("user-name").textContent = name;
    $("user-avatar").textContent = initials(name);
    $("user-bar").hidden = false;
    $("tabs").hidden = false;
    // Someone with no household yet starts at setup.
    const { households } = await api("/api/v1/households/me");
    noteSetup(households);
    if (!households.length && !["household", ...PUBLIC_PAGES].includes(currentPage())) {
      window.location.hash = "#/household";
      return;
    }
  }
  await route();
}

let currentUser = null;

let routedPage = null;

async function route() {
  const page = currentPage();
  // A different page starts at the top; Refresh and same-page hash changes keep the scroll.
  if (routedPage && routedPage !== page) window.scrollTo(0, 0);
  routedPage = page;
  if (signedIn || PUBLIC_PAGES.includes(page)) {
    $("logged-out").hidden = true;
    await showPage(page);
  } else {
    for (const p of PAGES) $(`page-${p}`).hidden = true;
    $("logged-out").hidden = false;
  }
}

// ---------- routing ----------

function currentPage() {
  const name = window.location.hash.replace(/^#\/?/, "");
  return PAGES.includes(name) ? name : DEFAULT_PAGE;
}

// When each page last loaded; switching back after this long fetches fresh data.
const loaded = {};
const STALE_AFTER_MS = 60_000;

async function showPage(name, { force = false } = {}) {
  for (const p of PAGES) $(`page-${p}`).hidden = p !== name;
  for (const a of document.querySelectorAll("a[data-page]")) {
    if (a.dataset.page === name) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  }
  if ($("status").dataset.state !== "loading") setStatus(""); // messages belong to the page they came from
  if (!force && loaded[name] && Date.now() - loaded[name] < STALE_AFTER_MS) return;
  // Search has no loader: it only runs when the caregiver submits a question.
  const loaders = {
    items: loadItems, map: loadMap, timeline: loadTimeline, insights: loadInsights,
    search: async () => {}, public: loadPublic, household: loadHousehold,
  };
  // First visit: placeholder shapes; later visits keep the old content until the new arrives.
  if (!loaded[name]) paintSkeleton(name);
  setStatus("Loading…", "loading");
  try {
    await loaders[name]();
    loaded[name] = Date.now();
    if ($("status").dataset.state === "loading") setStatus("");
  } catch (err) {
    setStatus(`Couldn't load this page: ${err.message}`, "error", () => showPage(name, { force: true }));
  } finally {
    for (const map of document.querySelectorAll(".home-map.is-loading")) map.classList.remove("is-loading");
  }
}

// The status line under the header: "loading" shows a spinner, "error" a red notice
// with a Try again button when the failed action can be retried.
function setStatus(text, state = "", retry = null) {
  const status = $("status");
  status.replaceChildren();
  status.dataset.state = text ? state : "";
  if (!text) return;
  status.append(el("span", {}, text));
  if (retry) status.append(el("button", { type: "button", class: "pill", onclick: retry }, "Try again"));
}

// A button's loading state: an inline spinner (CSS), disabled until done.
function setBusy(button, busy) {
  button.setAttribute("aria-busy", String(busy));
  button.disabled = busy;
}

// ---------- loading skeletons (shapes match each page's components) ----------

const sk = (cls) => el("span", { class: `sk ${cls}`, "aria-hidden": "true" });
const SK_ITEM = () =>
  el("li", { class: "item-card is-skeleton", "aria-hidden": "true" },
    el("div", { class: "photo sk" }),
    el("div", { class: "item-body" }, sk("line w60 h20"), sk("line w40"), sk("line w70")));
const SK_SCENE = () =>
  el("li", { class: "scene is-skeleton", "aria-hidden": "true" },
    el("div", { class: "scene-time" }, sk("line w70"), sk("line w50")),
    el("div", { class: "scene-body" }, el("div", { class: "photo sk" }),
      el("div", { class: "scene-text" }, sk("line w80 h24"), sk("line w60"), sk("line w30"))));
const SK_BARS = () => Array.from({ length: 5 }, () => el("div", { class: "bar-row", "aria-hidden": "true" }, sk("line w70"), sk("line h12"), sk("line w100")));
const SK_RESULT = () =>
  el("li", { class: "result is-skeleton", "aria-hidden": "true" },
    el("div", { class: "photo sk" }),
    el("div", { class: "result-body" }, sk("line w30"), sk("line w80"), sk("line w60"), sk("line w70")));
const repeat = (n, make) => Array.from({ length: n }, make);

function paintSkeleton(name) {
  if (name === "items") {
    $("items-featured").replaceChildren(
      el("article", { class: "featured is-skeleton", "aria-hidden": "true" },
        el("div", { class: "photo sk" }),
        el("div", { class: "featured-body" }, sk("line w30"), sk("line w50 h40"), sk("line w40 h24"), sk("line w90"), sk("line w70"))));
    $("items-grid").replaceChildren(...repeat(6, SK_ITEM));
  } else if (name === "timeline") {
    $("feed").replaceChildren(...repeat(4, SK_SCENE));
  } else if (name === "insights") {
    for (const id of ["stat-memories", "stat-questions", "stat-unanswered", "stat-items"]) $(id).replaceChildren(sk("line w40 h40"));
    for (const id of ["chart-asked", "chart-logged"]) $(id).replaceChildren(...SK_BARS());
    $("chart-activity").replaceChildren(sk("block h200"));
    $("questions").replaceChildren(...repeat(3, () => el("li", { "aria-hidden": "true" }, sk("line w60 h20"), sk("line w80"))));
  } else if (name === "household") {
    $("household-body").replaceChildren(el("div", { class: "two-col" }, ...repeat(2, () =>
      el("div", { class: "card setup-card is-skeleton", "aria-hidden": "true" }, sk("line w40 h24"), sk("line w80"), sk("block h44"), sk("block h44"), sk("line w30 h44")))));
  } else if (name === "map") {
    for (const id of ["home-map", "places-map"]) $(id).classList.add("is-loading");
  } else if (name === "public") {
    $("public-map").classList.add("is-loading");
    $("public-list").replaceChildren(...repeat(3, () => el("li", { class: "public-row", "aria-hidden": "true" }, sk("line w60"), sk("line w40"))));
  }
}

async function api(path, { method = "GET", body } = {}) {
  let token;
  try {
    token = await auth0Client.getTokenSilently({ authorizationParams: { audience } });
  } catch (err) {
    // Cached session expired: go back through Universal Login (usually instant with SSO).
    if (["login_required", "consent_required", "missing_refresh_token"].includes(err.error)) {
      await auth0Client.loginWithRedirect({ appState: { hash: window.location.hash } });
    }
    throw err;
  }
  const res = await fetch(path, {
    method,
    headers: { Authorization: `Bearer ${token}`, ...(body ? { "Content-Type": "application/json" } : {}) },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(`${res.status} ${detail.detail || res.statusText}`);
  }
  return res.json();
}

// ---------- shared formatting ----------

function timeAgo(iso) {
  const minutes = Math.round((Date.now() - new Date(iso)) / 60000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} hour${hours === 1 ? "" : "s"} ago`;
  const days = Math.round(hours / 24);
  return `${days} day${days === 1 ? "" : "s"} ago`;
}

function shortAgo(iso) {
  const minutes = Math.round((Date.now() - new Date(iso)) / 60000);
  if (minutes < 60) return `${Math.max(minutes, 1)}m ago`;
  if (minutes < 60 * 24) return `${Math.round(minutes / 60)}h ago`;
  return `${Math.round(minutes / 1440)}d ago`;
}

function clockTime(iso) {
  return new Date(iso).toLocaleString(undefined, { weekday: "short", hour: "numeric", minute: "2-digit" });
}

const capitalize = (s) => (s ? s[0].toUpperCase() + s.slice(1) : s);

// "Riley Chen" → "RC", "sam@example.com" → "S"
function initials(name) {
  const words = name.replace(/@.*/, "").split(/[\s._-]+/).filter(Boolean);
  return words.slice(0, 2).map((w) => w[0].toUpperCase()).join("") || "?";
}

function setWindowHours(hours) {
  for (const el of document.querySelectorAll(".window-hours")) el.textContent = hours;
}

// Setup progress, from /households/me: drives the empty-state checklists.
let setup = { household: false, wearable: false };
function noteSetup(households) {
  setup = { household: households.length > 0, wearable: households.some((h) => h.devices.length > 0) };
}

/**
 * An empty state. Plain: one line of text. With opts.title it becomes a card with the
 * setup checklist (opts.checklist = the last step's label) and an action (opts.cta =
 * {label, href} or {label, onclick}).
 */
function emptyState(container, text, opts = {}) {
  const tag = ["OL", "UL"].includes(container.tagName) ? "li" : "div";
  if (!opts.title) return container.replaceChildren(el(tag, { class: "empty" }, text));
  const steps = opts.checklist && [
    ["Create a household for the person who wears it", setup.household],
    ["Add the wearable with the Client ID and pairing code on its label", setup.wearable],
    [opts.checklist, false],
  ];
  // Until setup is done, the action is finishing it.
  const cta = opts.checklist && !(setup.household && setup.wearable)
    ? { label: setup.household ? "Add the wearable" : "Set up a household", href: "#/household" }
    : opts.cta;
  container.replaceChildren(
    el(tag, { class: "empty empty-rich" },
      el("h3", {}, opts.title),
      el("p", {}, text),
      steps && el("ol", { class: "checklist", "aria-label": "Setup steps" },
        steps.map(([label, done]) => el("li", { class: done ? "done" : "" },
          el("span", { class: "check", "aria-hidden": "true" }),
          el("span", {}, label, done ? el("span", { class: "sr-only" }, " (done)") : null)))),
      cta && (cta.href
        ? el("a", { class: "primary", href: cta.href }, cta.label)
        : el("button", { type: "button", class: "primary", onclick: cta.onclick }, cta.label))));
}

// ---------- Items page ----------

function chip(text, cls = "") {
  const span = document.createElement("span");
  span.className = `chip ${cls}`.trim();
  span.textContent = text;
  return span;
}

function placeChip(name, location) {
  const span = document.createElement("span");
  span.className = "chip place";
  const where = document.createElement("b");
  where.textContent = location || "somewhere";
  span.append(`${capitalize(name)} → `, where);
  return span;
}

function fillPhoto(photo, url, alt) {
  const img = photo.querySelector("img");
  if (url) {
    img.src = url;
    img.alt = alt;
  } else {
    img.remove();
    photo.classList.add("empty-photo");
  }
}

// Bounding boxes from the vision pipeline, drawn over the photo. The SVG uses the image's
// own pixel size with "slice", so it crops exactly like the photo's object-fit: cover.
function drawBoxes(photo, boxes) {
  const img = photo.querySelector("img");
  if (!img || !boxes?.length) return;
  const render = () => {
    const w = img.naturalWidth;
    const h = img.naturalHeight;
    if (!w || !h) return;
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
    svg.setAttribute("preserveAspectRatio", "xMidYMid slice");
    svg.setAttribute("aria-hidden", "true");
    svg.classList.add("boxes");
    for (const b of boxes) {
      const [x, y, bw, bh] = b.bbox;
      const rect = document.createElementNS(ns, "rect");
      for (const [k, v] of Object.entries({ x: x * w, y: y * h, width: bw * w, height: bh * h, rx: Math.min(w, h) * 0.01 })) rect.setAttribute(k, v);
      rect.setAttribute("vector-effect", "non-scaling-stroke");
      rect.classList.toggle("maybe", Boolean(b.maybe));
      const label = document.createElementNS(ns, "text");
      label.setAttribute("x", x * w);
      label.setAttribute("y", Math.max(y * h - h * 0.015, h * 0.05));
      label.setAttribute("font-size", Math.round(h * 0.045));
      label.textContent = `${b.label}${b.maybe ? "?" : ""} ${Math.round(b.confidence * 100)}%`;
      svg.append(rect, label);
    }
    photo.append(svg);
  };
  if (img.complete) render();
  else img.addEventListener("load", render, { once: true });
  img.alt += ` Detected: ${boxes.map((b) => b.label).join(", ")}.`;
}

function autoChips(boxes) {
  const chips = [chip("Seen automatically", "auto")];
  if (boxes?.some((b) => b.maybe)) chips.push(chip("Maybe", "warn"));
  return chips;
}

// What Gemini saw in the photo, shown under the wearer's own words.
function fillSaw(node, scene) {
  const saw = node.querySelector(".saw");
  if (!scene || scene.startsWith("Seen automatically")) return;
  saw.hidden = false;
  saw.textContent = `Camera saw: ${scene}`;
}

function itemChips(item) {
  const chips = [chip(`Put down ${item.mentions}×`)];
  if (item.times_asked) chips.push(chip(`Asked about ${item.times_asked}×`, "warn"));
  return chips;
}

async function loadItems() {
  itemsCache = await api("/api/v1/dashboard/items");
  const { window_hours, items } = itemsCache;
  setWindowHours(window_hours);
  $("items-count").textContent = `${items.length} item${items.length === 1 ? "" : "s"}`;
  const featured = $("items-featured");
  const grid = $("items-grid");
  if (!items.length) {
    featured.replaceChildren();
    emptyState(grid, "When the wearer says something like “I'm putting my keys on the counter”, the item shows up here with a photo.", {
      title: "No items yet",
      checklist: "Hold REMEMBER and say where something is going",
      cta: { label: "Check again", onclick: () => showPage("items", { force: true }) },
    });
    return;
  }

  // The most recently put-down item gets the big "latest scene" treatment.
  const [latest, ...rest] = items;
  const hero = $("featured-template").content.firstElementChild.cloneNode(true);
  hero.dataset.name = latest.name;
  fillPhoto(hero.querySelector(".photo"), latest.image_url, `The ${latest.name} on the ${latest.location || "spot shown"}`);
  hero.querySelector(".when").textContent = `${timeAgo(latest.last_seen)} · ${clockTime(latest.last_seen)}`;
  hero.querySelector(".card-link").textContent = capitalize(latest.name);
  hero.querySelector(".card-link").onclick = () => openItem(latest.name);
  hero.querySelector(".item-location").textContent = capitalize(latest.location) || "Location not said";
  hero.querySelector(".quote").textContent = latest.transcript ? `“${latest.transcript}”` : latest.scene || "";
  fillSaw(hero, latest.scene);
  if (latest.detection) drawBoxes(hero.querySelector(".photo"), [latest.detection]);
  hero.querySelector(".chips").replaceChildren(...itemChips(latest), ...(latest.detection ? autoChips([latest.detection]) : []));
  featured.replaceChildren(hero);

  const tpl = $("item-template");
  grid.replaceChildren(
    ...rest.map((item) => {
      const node = tpl.content.firstElementChild.cloneNode(true);
      node.dataset.name = item.name;
      fillPhoto(node.querySelector(".photo"), item.image_url, `The ${item.name} on the ${item.location || "spot shown"}`);
      node.querySelector(".card-link").textContent = capitalize(item.name);
      node.querySelector(".card-link").onclick = () => openItem(item.name);
      node.querySelector(".when").textContent = shortAgo(item.last_seen);
      node.querySelector(".when").title = clockTime(item.last_seen);
      node.querySelector(".item-location").textContent = capitalize(item.location) || `“${item.transcript}”`;
      node.querySelector(".chips").replaceChildren(...itemChips(item), ...(item.detection ? autoChips([item.detection]) : []));
      if (item.detection) drawBoxes(node.querySelector(".photo"), [item.detection]);
      if (item.previous.length) {
        const before = node.querySelector(".before");
        before.hidden = false;
        before.querySelector("ul").replaceChildren(
          ...item.previous.map((p) => {
            const li = document.createElement("li");
            li.textContent = `${capitalize(p.location) || "Somewhere"} · ${timeAgo(p.timestamp)}`;
            return li;
          })
        );
      }
      return node;
    })
  );
  filterItems();
}

function filterItems() {
  const q = $("item-filter").value.trim().toLowerCase();
  for (const card of document.querySelectorAll("#page-items [data-name]")) {
    card.hidden = q !== "" && !card.dataset.name.includes(q);
  }
}

// ---------- Timeline page ----------

async function loadTimeline() {
  const entries = await api("/api/v1/dashboard/history");
  const feed = $("feed");
  if (!entries.length) {
    emptyState(feed, "Every memory from the last 24 hours shows here, newest first, with its photo.", {
      title: "Nothing recorded today",
      checklist: "Hold REMEMBER and say where something is going",
      cta: { label: "Check again", onclick: () => showPage("timeline", { force: true }) },
    });
    return;
  }
  const tpl = $("entry-template");
  feed.replaceChildren(
    ...entries.map((e) => {
      const node = tpl.content.firstElementChild.cloneNode(true);
      const auto = e.source === "auto";
      fillPhoto(node.querySelector(".photo"), e.image_url, auto ? `Camera frame: ${e.summary || "automatic sighting"}.` : `Photo taken when they said: ${e.transcript}`);
      const time = node.querySelector("time");
      time.dateTime = e.timestamp;
      time.textContent = new Date(e.timestamp).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
      node.querySelector(".ago").textContent = timeAgo(e.timestamp);
      node.querySelector(".quote").textContent = e.transcript ? `“${e.transcript}”` : capitalize(e.summary || "");
      fillSaw(node, e.scene);
      node.querySelector(".chips").replaceChildren(
        ...(e.items || []).map((i) => placeChip(i.name, i.location)),
        ...(auto ? autoChips(e.detections) : [])
      );
      if (auto) drawBoxes(node.querySelector(".photo"), e.detections);
      return node;
    })
  );
}

// ---------- Map page (deck.gl, loaded on first visit) ----------

const DECK_URL = "https://cdn.jsdelivr.net/npm/deck.gl@9.4.0/dist.min.js";
const MAP_COLORS = {
  roomLine: [62, 58, 52],
  roomFill: [23, 22, 20], // --surface
  lost: [201, 133, 0], // --series-2
  asked: [244, 167, 59], // --amber
  calm: [57, 135, 229], // --series-1
  label: [168, 161, 150],
  text: [243, 239, 232],
};
const GEO_TOOLTIP_STYLE = {
  background: "#f3efe8", color: "#0e0d0c", fontSize: "13px", fontFamily: "Geist, system-ui, sans-serif",
  borderRadius: "8px", padding: "8px 10px", boxShadow: "0 10px 30px -10px rgba(0,0,0,.8)", lineHeight: "1.45",
};
const LOOP_SECONDS = 10;
let mapState = null;

function loadScript(src) {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) return resolve();
    const el = document.createElement("script");
    el.src = src;
    el.onload = resolve;
    el.onerror = () => reject(new Error("Could not load the map library"));
    document.head.append(el);
  });
}

async function loadHouseMap() {
  const [data] = await Promise.all([api("/api/v1/dashboard/map"), loadScript(DECK_URL)]);
  setWindowHours(data.window_hours);
  renderMapTable(data);
  const container = $("home-map");
  if (!data.items.length) {
    mapState?.deck.finalize();
    mapState = null;
    return emptyState(container, "Items appear in their rooms once the wearable logs where they were put down.", {
      title: "Nothing to map yet",
      checklist: "Log a few items in different rooms",
      cta: { label: "Check again", onclick: () => showPage("map", { force: true }) },
    });
  }
  if (!mapState) {
    container.replaceChildren();
    const { Deck, OrthographicView } = deck;
    // Fit the floor plan (plus a margin) to the frame.
    const xs = data.rooms.flatMap((r) => r.polygon.map((pt) => pt[0]));
    const ys = data.rooms.flatMap((r) => r.polygon.map((pt) => pt[1]));
    const [minX, maxX, minY, maxY] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const fit = Math.min((container.clientWidth || 800) / (maxX - minX + 10), (container.clientHeight || 480) / (maxY - minY + 10));
    mapState = {
      data,
      time: 0,
      playing: !matchMedia("(prefers-reduced-motion: reduce)").matches,
      deck: new Deck({
        parent: container,
        views: new OrthographicView({ flipY: true }),
        initialViewState: { target: [(minX + maxX) / 2, (minY + maxY) / 2, 0], zoom: Math.log2(fit), minZoom: 1, maxZoom: 6 },
        controller: true,
        getTooltip: mapTooltip,
        style: { position: "absolute", inset: "0" },
      }),
      lastFrame: performance.now(),
    };
    $("map-play").onclick = () => {
      mapState.playing = !mapState.playing;
      updatePlayButton();
    };
    requestAnimationFrame(tickMap);
  }
  mapState.data = data;
  updatePlayButton();
  drawMap();
}

function updatePlayButton() {
  const btn = $("map-play");
  btn.textContent = mapState.playing ? "Pause" : "Play";
  btn.setAttribute("aria-pressed", String(mapState.playing));
  btn.hidden = !mapState.data.trips.length;
}

function tickMap(now) {
  if (!mapState) return;
  const span = Math.max(mapState.data.span_hours, 1);
  if (mapState.playing && mapMode === "house" && !$("page-map").hidden) {
    mapState.time = (mapState.time + ((now - mapState.lastFrame) / 1000) * (span / LOOP_SECONDS)) % (span + 2);
    drawMap();
  }
  mapState.lastFrame = now;
  requestAnimationFrame(tickMap);
}

function drawMap() {
  const { PolygonLayer, ScatterplotLayer, TextLayer, TripsLayer, PathLayer } = deck;
  const { data, playing, time } = mapState;
  const maxLost = Math.max(1, ...data.rooms.map((r) => r.lost));
  const maxAsked = Math.max(1, ...data.items.map((i) => i.times_asked));
  $("map-clock").textContent = data.trips.length
    ? new Date(Date.parse(data.start) + Math.min(time, data.span_hours) * 3.6e6).toLocaleString(undefined, {
        weekday: "short",
        hour: "numeric",
        minute: "2-digit",
      })
    : "";

  const layers = [
    new PolygonLayer({
      id: "rooms",
      data: data.rooms,
      getPolygon: (r) => r.polygon,
      getFillColor: (r) => (r.lost ? [...MAP_COLORS.lost, 40 + Math.round((r.lost / maxLost) * 150)] : [...MAP_COLORS.roomFill, 255]),
      getLineColor: MAP_COLORS.roomLine,
      lineWidthUnits: "pixels",
      getLineWidth: 2,
      pickable: true,
      updateTriggers: { getFillColor: maxLost },
    }),
    new TextLayer({
      id: "room-names",
      data: data.rooms,
      getPosition: (r) => [r.polygon[0][0] + 2, r.polygon[0][1] + 3],
      getText: (r) => r.name.toUpperCase(),
      getColor: MAP_COLORS.label,
      getSize: 11,
      getTextAnchor: "start",
      getAlignmentBaseline: "top",
      fontFamily: "Geist Mono, monospace",
      characterSet: "auto",
    }),
    playing
      ? new TripsLayer({
          id: "trails",
          data: data.trips,
          getPath: (t) => t.path,
          getTimestamps: (t) => t.timestamps,
          getColor: MAP_COLORS.asked,
          widthUnits: "pixels",
          getWidth: 3,
          capRounded: true,
          jointRounded: true,
          trailLength: Math.max(data.span_hours / 4, 2),
          currentTime: time,
          fadeTrail: true,
        })
      : new PathLayer({
          id: "trails",
          data: data.trips,
          getPath: (t) => t.path,
          getColor: [...MAP_COLORS.asked, 140],
          widthUnits: "pixels",
          getWidth: 2,
          capRounded: true,
          jointRounded: true,
        }),
    new ScatterplotLayer({
      id: "items",
      data: data.items,
      getPosition: (i) => i.position,
      getRadius: (i) => 6 + (i.times_asked / maxAsked) * 10,
      radiusUnits: "pixels",
      getFillColor: (i) => (i.times_asked ? MAP_COLORS.asked : MAP_COLORS.calm),
      getLineColor: [14, 13, 12],
      stroked: true,
      lineWidthUnits: "pixels",
      getLineWidth: 2,
      pickable: true,
    }),
    new TextLayer({
      id: "item-names",
      data: data.items,
      getPosition: (i) => i.position,
      getText: (i) => i.name,
      getColor: MAP_COLORS.text,
      getSize: 13,
      getPixelOffset: [0, 20],
      fontFamily: "Geist, system-ui, sans-serif",
      characterSet: "auto",
      outlineWidth: 3,
      outlineColor: [14, 13, 12, 255],
      fontSettings: { sdf: true },
    }),
  ];
  mapState.deck.setProps({ layers });
}

function mapTooltip({ object, layer }) {
  if (!object) return null;
  const style = GEO_TOOLTIP_STYLE;
  if (layer.id === "items") {
    return {
      text: `${capitalize(object.name)} — ${object.room_name}\n“${object.location || "no place said"}”, ${timeAgo(object.last_seen)}\nPut down ${object.mentions}×, asked about ${object.times_asked}×`,
      style,
    };
  }
  if (layer.id === "rooms") {
    const items = object.items.length ? object.items.join(", ") : "nothing now";
    return { text: `${object.name}\n${object.putdowns} put down here · ${object.lost} questions\nNow here: ${items}`, style };
  }
  return null;
}

function renderMapTable(data) {
  Charts.dataTable(
    $("map-table"),
    ["Room", "Put down here", "Questions about items last here", "Items there now"],
    data.rooms.map((r) => [r.name, r.putdowns, r.lost, r.items.join(", ") || "–"])
  );
}

// ---------- Map modes: House (floor plan) | Our places (real map) ----------

let mapMode = "house";

async function loadMap() {
  return mapMode === "house" ? loadHouseMap() : loadPlacesMap();
}

async function setMapMode(mode) {
  mapMode = mode;
  for (const btn of document.querySelectorAll(".segmented [data-mode]")) {
    btn.setAttribute("aria-pressed", String(btn.dataset.mode === mode));
  }
  $("house-card").hidden = mode !== "house";
  $("places-card").hidden = mode !== "places";
  $("map-play").hidden = mode !== "house" || !mapState?.data.trips.length;
  $("map-clock").hidden = mode !== "house";
  loaded.map = 0;
  await showPage("map");
}

// OpenStreetMap's standard tiles (no key; light use with attribution, per the OSM tile
// policy), desaturated and tinted down to sit on the dark theme.
const BASEMAP_URLS = ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"];

function basemap(tint = [70, 66, 60]) {
  return new deck.TileLayer({
    id: "basemap",
    data: BASEMAP_URLS,
    minZoom: 0,
    maxZoom: 19,
    tileSize: 256,
    renderSubLayers: (props) => {
      const [[west, south], [east, north]] = props.tile.boundingBox;
      return new deck.BitmapLayer(props, {
        data: null,
        image: props.data,
        bounds: [west, south, east, north],
        desaturate: 1,
        tintColor: tint,
      });
    },
  });
}

function fitView(points, container, extra = {}) {
  const width = container.clientWidth || 800;
  const height = container.clientHeight || 480;
  if (!points.length) return { longitude: -98, latitude: 38, zoom: 3, ...extra };
  if (points.length === 1) return { longitude: points[0][0], latitude: points[0][1], zoom: 15, ...extra };
  const lngs = points.map((p) => p[0]);
  const lats = points.map((p) => p[1]);
  const view = new deck.WebMercatorViewport({ width, height }).fitBounds(
    [[Math.min(...lngs), Math.min(...lats)], [Math.max(...lngs), Math.max(...lats)]],
    { padding: Math.min(80, width / 6) }
  );
  return { longitude: view.longitude, latitude: view.latitude, zoom: Math.min(view.zoom, 16.5), ...extra };
}

let placesDeck = null;

async function loadPlacesMap() {
  const [data] = await Promise.all([api("/api/v1/dashboard/places"), loadScript(DECK_URL)]);
  const places = data.households.flatMap((h) => h.places.map((p) => ({ ...p, household: h.name })));
  const container = $("places-map");
  if (!places.length) {
    placesDeck?.finalize();
    placesDeck = null;
    return emptyState(container, "Places appear once the wearable sends Wi-Fi positions with its memories.", {
      title: "No places yet",
      checklist: "Log a memory with Wi-Fi turned on",
      cta: { label: "Check again", onclick: () => showPage("map", { force: true }) },
    });
  }
  const layers = [
    basemap(),
    new deck.ScatterplotLayer({
      id: "place-areas",
      data: places,
      getPosition: (p) => [p.lng, p.lat],
      getRadius: (p) => p.radius_m,
      radiusUnits: "meters",
      radiusMinPixels: 10,
      getFillColor: (p) => (p.kind === "public" ? [...MAP_COLORS.asked, 90] : [...MAP_COLORS.calm, 70]),
      getLineColor: (p) => (p.kind === "public" ? MAP_COLORS.asked : MAP_COLORS.calm),
      stroked: true,
      lineWidthUnits: "pixels",
      getLineWidth: 2,
      pickable: true,
    }),
    new deck.ScatterplotLayer({
      id: "memories",
      data: data.memories,
      getPosition: (m) => [m.lng, m.lat],
      getRadius: 4,
      radiusUnits: "pixels",
      getFillColor: [...MAP_COLORS.text, 220],
      pickable: true,
    }),
    new deck.TextLayer({
      id: "place-labels",
      data: places,
      getPosition: (p) => [p.lng, p.lat],
      getText: (p) => p.label,
      getColor: MAP_COLORS.text,
      getSize: 14,
      getPixelOffset: [0, -22],
      fontFamily: "Geist, system-ui, sans-serif",
      characterSet: "auto",
      outlineWidth: 3,
      outlineColor: [14, 13, 12, 255],
      fontSettings: { sdf: true },
    }),
  ];
  const getTooltip = ({ object, layer }) => {
    if (!object) return null;
    if (layer.id === "place-areas") {
      return { text: `${object.label} · ${object.kind}\n${object.visits} memories here · ${object.household}`, style: GEO_TOOLTIP_STYLE };
    }
    if (layer.id === "memories") {
      return { text: `“${object.transcript}”\n${object.items || "no item"} · ${timeAgo(object.timestamp)}`, style: GEO_TOOLTIP_STYLE };
    }
    return null;
  };
  if (!placesDeck) {
    container.replaceChildren();
    placesDeck = new deck.Deck({
      parent: container,
      initialViewState: fitView(places.map((p) => [p.lng, p.lat]), container),
      controller: true,
      getTooltip,
      style: { position: "absolute", inset: "0" },
    });
  }
  placesDeck.setProps({ layers, getTooltip });
}

// ---------- Public map (no login) ----------

let publicDeck = null;

// Warm sequential ramp for the 3D grid: dark crimson (few) → bright yellow (many). Lightness
// rises monotonically (OKLab L 0.38 → 0.88), so it reads as "more" on the dark map; the
// legend under the map uses the same stops.
const HEAT_COLORS = [
  [122, 16, 48], [176, 24, 46], [222, 59, 36], [242, 106, 27], [249, 161, 27], [255, 210, 63],
];
const NAVY_TINT = [46, 56, 72]; // the public map's basemap: desaturated OSM, tinted slate-navy

function rampColor(t) {
  const x = Math.min(Math.max(t, 0), 1) * (HEAT_COLORS.length - 1);
  const i = Math.min(Math.floor(x), HEAT_COLORS.length - 2);
  const f = x - i;
  return HEAT_COLORS[i].map((v, k) => Math.round(v + (HEAT_COLORS[i + 1][k] - v) * f));
}

// 3D buildings: OpenFreeMap vector tiles (OpenStreetMap data, free, no key). The tile URL
// carries a dated version, so it's read from the TileJSON once; no buildings if that fails.
const BUILDINGS_TILEJSON = "https://tiles.openfreemap.org/planet";
let buildingTiles = null;

function buildingTileUrls() {
  buildingTiles ??= fetch(BUILDINGS_TILEJSON)
    .then((r) => (r.ok ? r.json() : null))
    .then((tilejson) => tilejson?.tiles || null)
    .catch(() => null);
  return buildingTiles;
}

function buildingsLayer(tiles) {
  return new deck.MVTLayer({
    id: "buildings",
    data: tiles,
    // Heights (render_height) only come with z14 tiles: always fetch z14, one level finer
    // than the view, so buildings stand up from zoom 13 inwards.
    minZoom: 14,
    maxZoom: 14,
    zoomOffset: 1,
    loadOptions: { mvt: { layers: ["building"] } },
    extruded: true,
    stroked: false,
    getElevation: (f) => (f.properties.hide_3d ? 0 : f.properties.render_height || 6),
    getFillColor: [74, 68, 60, 240],
    material: { ambient: 0.5, diffuse: 0.55, shininess: 6, specularColor: [30, 28, 24] },
  });
}

async function loadPublic() {
  const [data, , tiles] = await Promise.all([
    fetch("/api/v1/public/lost-map").then((r) => r.json()),
    loadScript(DECK_URL),
    buildingTileUrls(),
  ]);
  $("public-k").textContent = data.min_households;
  const container = $("public-map");
  const cells = data.cells.map((c) => ({ ...c, total: c.left + c.asked }));
  const maxTotal = Math.max(1, ...cells.map((c) => c.total));
  const layers = [
    basemap(NAVY_TINT),
    tiles && buildingsLayer(tiles),
    // 3D grid: one square column per ~100 m public spot (78 m across, so neighbours keep a
    // gap); height and colour both encode how often things were lost or asked about there.
    new deck.ColumnLayer({
      id: "public-cells",
      data: cells,
      diskResolution: 4,
      angle: 45,
      radius: 55,
      extruded: true,
      getPosition: (c) => [c.lng, c.lat],
      getElevation: (c) => 40 + (c.total / maxTotal) * 900,
      getFillColor: (c) => [...rampColor(c.total / maxTotal), 245],
      material: { ambient: 0.55, diffuse: 0.65, shininess: 24, specularColor: [60, 50, 40] },
      pickable: true,
      autoHighlight: true,
      highlightColor: [255, 255, 255, 70],
    }),
  ];
  const getTooltip = ({ object }) =>
    object && {
      text: `${object.total} times here\n` + object.items.slice(0, 5).map((i) => `${capitalize(i.name)}: left ${i.left}, asked ${i.asked}`).join("\n"),
      style: GEO_TOOLTIP_STYLE,
    };
  if (!publicDeck) {
    container.replaceChildren();
    publicDeck = new deck.Deck({
      parent: container,
      // A tilted overview of every spot (fit flat, then step back: a tilted camera shows less
      // than fitBounds assumes). Buildings appear from zoom 13; "Busiest spots" flies close.
      initialViewState: (() => {
        const view = fitView(coreSpots(cells), container, { pitch: 52, bearing: -18 });
        return { ...view, zoom: Math.min(view.zoom, 14.2) - 0.7 };
      })(),
      controller: true,
      getTooltip,
      style: { position: "absolute", inset: "0" },
    });
  }
  publicDeck.setProps({ layers, getTooltip });

  const list = $("public-list");
  if (!cells.length) {
    return emptyState(list, `A spot appears once ${data.min_households} or more households have shared lost items there.`, {
      title: "Nothing public yet",
      cta: signedIn ? { label: "Choose what to share", href: "#/household" } : { label: "Log in to contribute", onclick: () => $("login").click() },
    });
  }
  list.replaceChildren(
    ...cells.slice(0, 10).map((c) => el("li", {},
      el("button", {
        type: "button",
        class: "public-row",
        "aria-label": `Show spot ${c.lat.toFixed(3)}, ${c.lng.toFixed(3)} on the map: ${c.items.slice(0, 4).map((i) => i.name).join(", ")}`,
        onclick: () => {
          publicDeck.setProps({ initialViewState: { ...spotView(c), transitionDuration: 900, transitionInterpolator: new deck.FlyToInterpolator() } });
          $("public-map").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "center" });
        },
      },
        el("span", { class: "public-items" }, c.items.slice(0, 4).map((i) => `${i.name} (${i.left + i.asked})`).join(", ")),
        el("span", { class: "mono public-where" }, `${c.lat.toFixed(3)}, ${c.lng.toFixed(3)}`),
        el("span", { class: "mono public-total" }, `${c.total}×`))))
  );
}

// The spots to frame at first: with many spots, the middle 80% by latitude and longitude, so
// a few far-off ones don't zoom the whole map out until the columns are specks.
function coreSpots(cells) {
  const points = cells.map((c) => [c.lng, c.lat]);
  if (points.length < 10) return points;
  const range = (k) => {
    const v = points.map((p) => p[k]).sort((a, b) => a - b);
    return [v[Math.floor(v.length * 0.1)], v[Math.ceil(v.length * 0.9) - 1]];
  };
  const [[lng0, lng1], [lat0, lat1]] = [range(0), range(1)];
  return points.filter(([lng, lat]) => lng >= lng0 && lng <= lng1 && lat >= lat0 && lat <= lat1);
}

// A tilted close-up of one public spot, near enough that the buildings around it read.
function spotView(cell) {
  return { longitude: cell.lng, latitude: cell.lat, zoom: 15.4, pitch: 55, bearing: -20 };
}

// ---------- Household setup ----------

// Small DOM builder: el("div", {class: "x"}, "text", child) — text is never parsed as HTML.
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === false || v == null) continue;
    if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (k === "class") node.className = v;
    else node.setAttribute(k, v === true ? "" : v);
  }
  node.append(...children.flat().filter((c) => c != null && c !== false));
  return node;
}

let fieldIds = 0;

// A form field: a persistent visible label, the input, and an inline error under it.
function field(label, input) {
  input.id ||= `field-${++fieldIds}`;
  const error = el("span", { class: "field-error", id: `${input.id}-error`, "aria-live": "polite" });
  input.setAttribute("aria-describedby", error.id);
  return el("div", { class: "field" }, el("label", { for: input.id }, label, input.required ? null : el("span", { class: "optional" }, " (optional)")), input, error);
}

function validateField(input) {
  const error = $(`${input.id}-error`);
  const label = input.labels?.[0]?.firstChild?.textContent || "This field";
  const v = input.validity;
  const message = v.valid ? ""
    : v.valueMissing ? `Enter the ${/^[A-Z]{2}/.test(label.split(" ").pop()) ? label : label.toLowerCase()}.`
    : v.tooShort ? `${label} needs at least ${input.minLength} characters.`
    : v.patternMismatch ? input.title || "Check the format."
    : input.validationMessage;
  input.setAttribute("aria-invalid", String(!v.valid));
  if (error) error.textContent = message;
}

// A setup form: Submit stays disabled until every field is valid; fields explain what's
// wrong when you leave them; Submit shows a spinner while saving.
function formCard(title, hint, fields, submitLabel, onSubmit) {
  const msg = el("p", { class: "form-msg", role: "status" });
  const submit = el("button", { type: "submit", class: "primary" }, submitLabel);
  const form = el("form", {
    class: "card setup-card",
    novalidate: true,
    onsubmit: async (e) => {
      e.preventDefault();
      if (!form.checkValidity()) return form.querySelectorAll("input").forEach(validateField);
      msg.textContent = "";
      msg.classList.remove("error");
      const values = Object.fromEntries(new FormData(form).entries());
      setBusy(submit, true);
      try {
        msg.textContent = (await onSubmit(values)) || "";
      } catch (err) {
        msg.textContent = err.message.replace(/^\d+ /, "");
        msg.classList.add("error");
      } finally {
        setBusy(submit, false);
        sync();
      }
    },
  }, el("h2", {}, title), hint && el("p", { class: "hint" }, hint), ...fields, submit, msg);
  const sync = () => (submit.disabled = !form.checkValidity());
  form.addEventListener("input", (e) => {
    e.target.dataset.touched = "1";
    sync();
    if (e.target.getAttribute("aria-invalid") === "true") validateField(e.target); // clear as soon as it's fixed
  });
  // Explain a problem once the person has typed in a field and moved on.
  form.addEventListener("focusout", (e) => e.target.dataset.touched && validateField(e.target));
  sync();
  return form;
}

async function refreshHousehold() {
  for (const k of Object.keys(loaded)) loaded[k] = 0;
  await showPage("household", { force: true });
}

async function loadHousehold() {
  const { households } = await api("/api/v1/households/me");
  noteSetup(households);
  const body = $("household-body");
  const myName = currentUser?.name || currentUser?.email || "";
  if (!households.length) {
    $("household-title").textContent = "Set up LastScene.";
    body.replaceChildren(
      el("p", { class: "lede" }, "LastScene is usually set up by a family member for the person who wears it."),
      el("div", { class: "two-col" },
        formCard("Set up for someone", "Creates a household. You can add the wearable and invite family next.", [
          field("Household name", el("input", { name: "name", required: true, placeholder: "Mom's household", maxlength: 80 })),
          field("Your name", el("input", { name: "display_name", value: myName, maxlength: 80 })),
        ], "Create household", async (v) => {
          await api("/api/v1/households", { method: "POST", body: { name: v.name, display_name: v.display_name || null } });
          await refreshHousehold();
        }),
        formCard("Join a household", "Ask the person who set it up for an invite code.", [
          field("Invite code", el("input", { name: "code", required: true, minlength: 4, maxlength: 32, placeholder: "ABCD-1234-EF56", autocomplete: "off" })),
          field("Your name", el("input", { name: "display_name", value: myName, maxlength: 80 })),
        ], "Join", async (v) => {
          await api("/api/v1/invites/accept", { method: "POST", body: { code: v.code, display_name: v.display_name || null } });
          await refreshHousehold();
        })
      )
    );
    return;
  }
  $("household-title").textContent = households.length === 1 ? households[0].name : "Your households.";
  body.replaceChildren(...households.map((h) => householdView(h)));
}

function householdView(h) {
  const owner = h.role === "owner";
  const base = `/api/v1/households/${h.id}`;

  const devices = el("div", { class: "card setup-card" },
    el("h2", {}, "Wearable"),
    h.devices.length
      ? el("ul", { class: "rows" }, h.devices.map((d) => el("li", {},
          el("strong", {}, d.label || "Wearable"),
          el("span", { class: "mono muted" }, d.client_id),
          el("span", { class: "muted" }, d.last_seen ? `last heard ${timeAgo(d.last_seen)}` : "not heard from yet"))))
      : el("p", { class: "hint" }, "No wearable yet. Turn it on and connect it to Wi-Fi, then enter its Client ID and the pairing code on its label."));
  const claim = owner && formCard("Add a wearable", "The device must have connected at least once so LastScene knows its pairing code.", [
    field("Client ID", el("input", { name: "client_id", required: true, minlength: 4, maxlength: 128, autocomplete: "off", spellcheck: "false", placeholder: "from the device label" })),
    field("Pairing code", el("input", { name: "pairing_code", required: true, minlength: 4, maxlength: 32, autocomplete: "off", spellcheck: "false", placeholder: "ABCD-1234-EF56" })),
    field("Name", el("input", { name: "label", placeholder: "Mom's wearable", maxlength: 80 })),
  ], "Add wearable", async (v) => {
    const r = await api(`${base}/devices`, { method: "POST", body: { client_id: v.client_id, pairing_code: v.pairing_code, label: v.label || null } });
    await refreshHousehold();
    return `Added. ${r.memories_placed} earlier memories were placed on the map.`;
  });

  const inviteOut = el("p", { class: "invite-code mono", role: "status" });
  const people = el("div", { class: "card setup-card" },
    el("h2", {}, "People"),
    el("ul", { class: "rows" }, h.members.map((m) => el("li", {},
      el("strong", {}, m.display_name + (m.is_you ? " (you)" : "")),
      el("span", { class: "muted" }, m.role === "owner" ? "set this up" : "family")))),
    owner && el("button", { type: "button", class: "pill", onclick: async (e) => {
      const button = e.currentTarget;
      setBusy(button, true);
      try {
        const { code, expires_in_hours } = await api(`${base}/invites`, { method: "POST" });
        inviteOut.textContent = `Share this code: ${code} (works once, for ${expires_in_hours} hours)`;
      } catch (err) {
        inviteOut.textContent = `Couldn't make a code: ${err.message.replace(/^\d+ /, "")}`;
      } finally {
        setBusy(button, false);
      }
    } }, "Invite someone"),
    inviteOut);

  const placeRows = h.places.map((p) => {
    const save = async (patch) => {
      await api(`${base}/places/${p.id}`, { method: "PATCH", body: patch });
      for (const k of Object.keys(loaded)) loaded[k] = 0;
    };
    return el("li", { class: "place-row" },
      el("input", { class: "place-name", value: p.name || "", placeholder: p.label, "aria-label": `Name for ${p.label}`, maxlength: 80,
        onchange: (e) => save({ name: e.target.value.trim() || null }) }),
      el("select", { "aria-label": `Who can see ${p.label}`, onchange: (e) => save({ kind: e.target.value }) },
        el("option", { value: "private", selected: p.kind === "private" }, "Private"),
        el("option", { value: "public", selected: p.kind === "public" }, "Public")),
      el("span", { class: "mono muted" }, `${p.visits} memories`));
  });
  const places = el("div", { class: "card setup-card" },
    el("h2", {}, "Places"),
    el("p", { class: "hint" }, "Found automatically from the wearable's Wi-Fi position. Homes stay private. Mark shops, parks and transit public if you want them to count on the anonymous public map."),
    placeRows.length ? el("ul", { class: "rows" }, placeRows) : el("p", { class: "hint" }, "No places yet."));

  const share = el("input", { type: "checkbox", checked: h.share_public, disabled: !owner, onchange: async (e) => {
    try {
      await api(base, { method: "PATCH", body: { share_public: e.target.checked } });
    } catch (err) {
      e.target.checked = !e.target.checked;
      alert(err.message);
    }
  } });
  const sharing = el("div", { class: "card setup-card" },
    el("h2", {}, "Public map"),
    el("label", { class: "toggle" }, share, el("span", {}, "Share anonymous counts from our public places")),
    el("p", { class: "hint" }, "Off by default. Only item names and counts from places marked public, rounded to ~100 m, and only where 3+ households contribute. Private places never leave this household. New memories count from the moment this is on."),
    !owner && el("p", { class: "hint" }, "Only the person who set up the household can change this."));

  return el("div", { class: "household" }, el("div", { class: "two-col" }, devices, claim || people), claim ? el("div", { class: "two-col" }, people, sharing) : sharing, places);
}

// ---------- Search page ----------

const KIND_LABELS = { speech: "Said", scene: "Camera saw", item: "Item", text: "Readable text" };

async function runSearch(event) {
  event.preventDefault();
  const q = $("search-q").value.trim();
  const list = $("search-results");
  if (!q) return;
  const submit = $("search-submit");
  setBusy(submit, true);
  setStatus("Searching…", "loading");
  list.replaceChildren(...repeat(2, SK_RESULT));
  try {
    const { results } = await api(`/api/v1/dashboard/search?q=${encodeURIComponent(q)}`);
    setStatus(`${results.length} memor${results.length === 1 ? "y" : "ies"} found for “${q}”.`);
    if (!results.length) {
      return emptyState(list, "Nothing logged in the last 3 days matches that. Try a single item name, like “keys”.", {
        title: "No memories matched",
        cta: { label: "Try “keys”", onclick: () => { $("search-q").value = "keys"; $("search-form").requestSubmit(); } },
      });
    }
    const tpl = $("result-template");
    list.replaceChildren(
      ...results.map((r, i) => {
        const node = tpl.content.firstElementChild.cloneNode(true);
        const said = r.chunks.find((c) => c.kind === "speech")?.content;
        fillPhoto(node.querySelector(".photo"), r.image_url, `Photo from ${clockTime(r.timestamp)}${said ? `, when they said: ${said}` : ""}`);
        node.querySelector(".result-rank").textContent = `#${i + 1}`;
        const time = node.querySelector("time");
        time.dateTime = r.timestamp;
        time.textContent = `${clockTime(r.timestamp)} · ${timeAgo(r.timestamp)}`;
        node.querySelector(".result-score").textContent = `score ${r.score.toFixed(4)}`;
        node.querySelector(".chunks").replaceChildren(
          ...r.chunks.map((c) => {
            const li = document.createElement("li");
            li.className = `chunk${c.matched ? " matched" : ""}`;
            const kind = document.createElement("span");
            kind.className = `kind kind-${c.kind}`;
            kind.textContent = KIND_LABELS[c.kind] || c.kind;
            const text = document.createElement("span");
            text.className = "chunk-text";
            text.textContent = c.content;
            li.append(kind, text);
            if (c.matched) {
              const why = [];
              if (c.vector_rank) why.push(`meaning #${c.vector_rank}`);
              if (c.keyword_rank) why.push(`words #${c.keyword_rank}`);
              const tag = document.createElement("span");
              tag.className = "why mono";
              tag.textContent = why.join(" · ");
              li.append(tag);
            }
            return li;
          })
        );
        return node;
      })
    );
  } catch (err) {
    list.replaceChildren();
    setStatus(`Search failed: ${err.message}`, "error", () => $("search-form").requestSubmit());
  } finally {
    setBusy(submit, false);
    submit.disabled = !$("search-q").value.trim();
  }
}

// ---------- Item detail panel (drill-down from Items cards and Insights bars) ----------

let itemsCache = null; // last /dashboard/items response
let insightsCache = null; // last /dashboard/insights response

async function openItem(name) {
  const drawer = $("drawer");
  $("drawer-eyebrow").textContent = "Item";
  $("drawer-title").textContent = capitalize(name);
  const body = $("drawer-body");
  body.replaceChildren(el("div", { class: "photo sk detail-photo", "aria-hidden": "true" }), sk("line w60 h24"), sk("line w80"), sk("line w40"));
  if (!drawer.open) drawer.showModal();
  try {
    itemsCache ??= await api("/api/v1/dashboard/items");
    body.replaceChildren(...itemDetail(name, itemsCache.items.find((i) => i.name === name)));
  } catch (err) {
    body.replaceChildren(el("p", { class: "error" }, `Couldn't load this item: ${err.message}`),
      el("button", { type: "button", class: "pill", onclick: () => ((itemsCache = null), openItem(name)) }, "Try again"));
  }
}

function itemDetail(name, item) {
  const questions = (insightsCache?.recent_questions || []).filter((q) => q.item === name);
  const questionList = questions.length > 0 && el("section", { class: "detail-section" },
    el("h3", {}, "Questions about it"),
    el("ul", { class: "detail-list" }, questions.map((q) => el("li", {},
      el("strong", {}, `“${q.question}”`),
      el("span", { class: "muted" }, q.answer),
      el("time", { class: "mono", datetime: q.timestamp }, `${clockTime(q.timestamp)} · ${timeAgo(q.timestamp)}`)))));
  if (!item) {
    return [
      el("p", { class: "detail-where" }, "Never logged"),
      el("p", { class: "hint" }, "It was asked about but never put down with the wearable. Saying it out loud when putting it down (“I'm putting my glasses on the nightstand”) means the next question has an answer."),
      questionList,
    ];
  }
  const photo = el("div", { class: "photo detail-photo" }, el("img", { alt: "" }));
  fillPhoto(photo, item.image_url, `The ${item.name} on the ${item.location || "spot shown"}`);
  if (item.detection) drawBoxes(photo, [item.detection]);
  const spots = [{ location: item.location, timestamp: item.last_seen, now: true }, ...item.previous];
  return [
    photo,
    el("p", { class: "detail-where" }, capitalize(item.location) || "Location not said"),
    el("p", { class: "when mono" }, `${timeAgo(item.last_seen)} · ${clockTime(item.last_seen)}`),
    item.transcript && el("p", { class: "quote" }, `“${item.transcript}”`),
    el("div", { class: "chips" }, ...itemChips(item), ...(item.detection ? autoChips([item.detection]) : [])),
    el("section", { class: "detail-section" },
      el("h3", {}, "Where it's been"),
      el("ol", { class: "trail" }, spots.map((s) => el("li", { class: s.now ? "now" : "" },
        el("strong", {}, capitalize(s.location) || "Somewhere"),
        el("span", { class: "muted" }, s.now ? `now · ${timeAgo(s.timestamp)}` : timeAgo(s.timestamp)))))),
    questionList,
    el("a", { class: "pill", href: "#/map", onclick: () => $("drawer").close() }, "Show on the home map"),
  ];
}

// ---------- Insights page ----------

// Insights time range (hours); the API caps it at the 72h retention.
let insightsHours = 72;

function setInsightsRange(hours) {
  insightsHours = hours;
  for (const btn of document.querySelectorAll("#insights-range [data-hours]")) {
    btn.setAttribute("aria-pressed", String(Number(btn.dataset.hours) === hours));
  }
  showPage("insights", { force: true });
}

async function loadInsights() {
  const data = await api(`/api/v1/dashboard/insights?hours=${insightsHours}`);
  insightsCache = data;
  $("insights-hours").textContent = data.window_hours;
  $("stat-memories").textContent = data.memories_logged;
  $("stat-questions").textContent = data.questions_asked;
  $("stat-unanswered").textContent = data.unanswered_count;
  $("stat-items").textContent = data.items_tracked;
  $("tile-unanswered").classList.toggle("alert", data.unanswered_count > 0);

  renderHighlights(data);
  loadSummary();
  // Each bar opens that item's detail panel.
  const rows = (list) => list.map((r) => ({ label: capitalize(r.name), value: r.count, name: r.name }));
  const onSelect = (row) => openItem(row.name);
  Charts.barList($("chart-asked"), rows(data.most_asked), { unit: "times asked", empty: "No questions about specific items yet.", onSelect });
  Charts.barList($("chart-logged"), rows(data.most_logged), { unit: "times put down", empty: "No items logged yet.", onSelect });
  renderActivity(data.activity);
  renderQuestions(data.recent_questions);
}

// The LLM summary is slower than the numbers, so it fills in on its own.
async function loadSummary() {
  const card = $("summary-card");
  try {
    const { summary } = await api("/api/v1/dashboard/summary");
    card.hidden = !summary;
    $("summary-text").textContent = summary || "";
  } catch {
    card.hidden = true;
  }
}

// Builds a sentence with the item names in bold, without using innerHTML.
function sentence(...parts) {
  const p = document.createElement("p");
  for (const part of parts) {
    if (typeof part === "string") p.append(part);
    else {
      const strong = document.createElement("strong");
      strong.textContent = part.bold;
      p.append(strong);
    }
  }
  return p;
}

function renderHighlights(data) {
  const notes = [];
  const neverLogged = [...new Set(data.unanswered.map((q) => q.item).filter(Boolean))];
  // The most-asked item that *was* logged: keeps the advice distinct from the never-logged notes.
  const top = data.most_asked.find((r) => !neverLogged.includes(r.name));
  if (top && top.count >= 2) {
    notes.push(sentence("Questions about the ", { bold: top.name }, ` came up ${top.count} times. A fixed spot for it could help.`));
  }
  for (const item of neverLogged.slice(0, 3)) {
    notes.push(sentence("The ", { bold: item }, " was asked about but never logged. Encourage saying it out loud when putting it down."));
  }
  if (data.questions_asked && !data.unanswered_count) {
    notes.push(sentence("Every question was about an item that had been logged first. Nice."));
  }
  if (!data.memories_logged && !data.questions_asked) {
    notes.push(sentence("No activity yet. Insights appear once the wearable starts logging."));
  }
  $("highlights").hidden = notes.length === 0;
  $("highlight-list").replaceChildren(...notes);
}

// Activity buckets: 3-hour blocks of the viewer's local day.
const BUCKETS = ["12–3a", "3–6a", "6–9a", "9a–12", "12–3p", "3–6p", "6–9p", "9p–12"];

function renderActivity(activity) {
  const ofKind = (kind) => Charts.bucketByHour(activity.filter((a) => a.kind === kind), { buckets: BUCKETS.length });
  Charts.columns(
    $("chart-activity"),
    {
      categories: BUCKETS,
      series: [
        { name: "Things put down", values: ofKind("memory") },
        { name: "Questions asked", values: ofKind("question") },
      ],
    },
    { label: "Activity by time of day", table: $("activity-table") }
  );
}

function renderQuestions(questions) {
  const list = $("questions");
  if (!questions.length) return emptyState(list, "No questions asked yet.");
  list.replaceChildren(
    ...questions.map((q) => {
      const li = document.createElement("li");
      const head = document.createElement("div");
      head.className = "q-head";
      const text = document.createElement("strong");
      text.textContent = `“${q.question}”`;
      const badge = document.createElement("span");
      badge.className = `badge ${q.answerable ? "good" : "warning"}`;
      badge.textContent = q.answerable ? "✓ Logged earlier" : "! Never logged";
      head.append(text, badge);
      const answer = document.createElement("p");
      answer.className = "q-answer";
      answer.textContent = q.answer;
      const time = document.createElement("time");
      time.dateTime = q.timestamp;
      time.textContent = `${clockTime(q.timestamp)} · ${timeAgo(q.timestamp)}`;
      li.append(head, answer, time);
      return li;
    })
  );
}

init().catch((err) => {
  $("logged-out").hidden = false;
  $("logged-out").insertAdjacentHTML("beforeend", "<p class='error'></p>");
  $("logged-out").querySelector(".error").textContent = `Setup error: ${err.message}`;
});
