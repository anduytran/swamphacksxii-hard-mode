let auth0Client;
let audience;

const $ = (id) => document.getElementById(id);
const PAGES = ["items", "map", "timeline", "insights", "search"];
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

  $("login").onclick = () =>
    auth0Client.loginWithRedirect({ appState: { hash: window.location.hash } });
  $("logout").onclick = () =>
    auth0Client.logout({ logoutParams: { returnTo: window.location.origin } });
  $("refresh").onclick = () => showPage(currentPage(), { force: true });
  $("item-filter").oninput = filterItems;
  $("search-form").onsubmit = runSearch;

  if (await auth0Client.isAuthenticated()) {
    const user = await auth0Client.getUser();
    $("user-name").textContent = user?.name || user?.email || "";
    $("user-bar").hidden = false;
    $("tabs").hidden = false;
    window.addEventListener("hashchange", () => showPage(currentPage()));
    await showPage(currentPage());
  } else {
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
  for (const a of document.querySelectorAll("#tabs a")) {
    if (a.dataset.page === name) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  }
  if (!force && loaded[name] && Date.now() - loaded[name] < STALE_AFTER_MS) return;
  // Search has no loader: it only runs when the caregiver submits a question.
  const loaders = { items: loadItems, map: loadMap, timeline: loadTimeline, insights: loadInsights, search: async () => {} };
  const status = $("status");
  status.textContent = "Loading…";
  try {
    await loaders[name]();
    loaded[name] = Date.now();
    if (status.textContent === "Loading…") status.textContent = "";
  } catch (err) {
    status.textContent = `Couldn't load this page: ${err.message}`;
  }
}

async function api(path) {
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
  const res = await fetch(path, { headers: { Authorization: `Bearer ${token}` } });
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

function setWindowHours(hours) {
  for (const el of document.querySelectorAll(".window-hours")) el.textContent = hours;
}

function emptyState(container, text) {
  const p = document.createElement("p");
  p.className = "empty";
  p.textContent = text;
  container.replaceChildren(p);
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

// What Gemini saw in the photo, shown under the wearer's own words.
function fillSaw(node, scene) {
  const saw = node.querySelector(".saw");
  if (!scene) return;
  saw.hidden = false;
  saw.textContent = `Camera saw: ${scene}`;
}

function itemChips(item) {
  const chips = [chip(`Put down ${item.mentions}×`)];
  if (item.times_asked) chips.push(chip(`Asked about ${item.times_asked}×`, "warn"));
  return chips;
}

async function loadItems() {
  const { window_hours, items } = await api("/api/v1/dashboard/items");
  setWindowHours(window_hours);
  $("items-count").textContent = `${items.length} item${items.length === 1 ? "" : "s"}`;
  const featured = $("items-featured");
  const grid = $("items-grid");
  if (!items.length) {
    featured.replaceChildren();
    emptyState(grid, "No items yet. When the wearer says something like “I'm putting my keys on the counter”, the item shows up here.");
    return;
  }

  // The most recently put-down item gets the big "latest scene" treatment.
  const [latest, ...rest] = items;
  const hero = $("featured-template").content.firstElementChild.cloneNode(true);
  hero.dataset.name = latest.name;
  fillPhoto(hero.querySelector(".photo"), latest.image_url, `The ${latest.name} on the ${latest.location || "spot shown"}`);
  hero.querySelector(".when").textContent = `${timeAgo(latest.last_seen)} · ${clockTime(latest.last_seen)}`;
  hero.querySelector(".item-name").textContent = capitalize(latest.name);
  hero.querySelector(".item-location").textContent = capitalize(latest.location) || "Location not said";
  hero.querySelector(".quote").textContent = `“${latest.transcript}”`;
  fillSaw(hero, latest.scene);
  hero.querySelector(".chips").replaceChildren(...itemChips(latest));
  featured.replaceChildren(hero);

  const tpl = $("item-template");
  grid.replaceChildren(
    ...rest.map((item) => {
      const node = tpl.content.firstElementChild.cloneNode(true);
      node.dataset.name = item.name;
      fillPhoto(node.querySelector(".photo"), item.image_url, `The ${item.name} on the ${item.location || "spot shown"}`);
      node.querySelector(".item-name").textContent = capitalize(item.name);
      node.querySelector(".when").textContent = shortAgo(item.last_seen);
      node.querySelector(".when").title = clockTime(item.last_seen);
      node.querySelector(".item-location").textContent = capitalize(item.location) || `“${item.transcript}”`;
      node.querySelector(".chips").replaceChildren(...itemChips(item));
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
    emptyState(feed, "Nothing recorded in the last 24 hours.");
    return;
  }
  const tpl = $("entry-template");
  feed.replaceChildren(
    ...entries.map((e) => {
      const node = tpl.content.firstElementChild.cloneNode(true);
      fillPhoto(node.querySelector(".photo"), e.image_url, `Photo taken when they said: ${e.transcript}`);
      const time = node.querySelector("time");
      time.dateTime = e.timestamp;
      time.textContent = new Date(e.timestamp).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
      node.querySelector(".ago").textContent = timeAgo(e.timestamp);
      node.querySelector(".quote").textContent = `“${e.transcript}”`;
      fillSaw(node, e.scene);
      node.querySelector(".chips").replaceChildren(...(e.items || []).map((i) => placeChip(i.name, i.location)));
      return node;
    })
  );
}

// ---------- Map page (deck.gl, loaded on first visit) ----------

const DECK_URL = "https://cdn.jsdelivr.net/npm/deck.gl@9.4.0/dist.min.js";
const MAP_COLORS = {
  roomLine: [52, 49, 45],
  roomFill: [28, 27, 25],
  lost: [201, 133, 0], // --series-2
  asked: [244, 167, 59], // --amber
  calm: [57, 135, 229], // --series-1
  label: [168, 161, 150],
  text: [243, 239, 232],
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

async function loadMap() {
  const [data] = await Promise.all([api("/api/v1/dashboard/map"), loadScript(DECK_URL)]);
  setWindowHours(data.window_hours);
  renderMapTable(data);
  const container = $("home-map");
  if (!data.items.length) {
    mapState?.deck.finalize();
    mapState = null;
    return emptyState(container, "Nothing to map yet. Items appear here once the wearable logs where they were put down.");
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
  if (mapState.playing && !$("page-map").hidden) {
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
      fontFamily: "IBM Plex Mono, monospace",
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
      getLineColor: [18, 17, 16],
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
      fontFamily: "Bricolage Grotesque, system-ui, sans-serif",
      characterSet: "auto",
      outlineWidth: 3,
      outlineColor: [18, 17, 16, 255],
      fontSettings: { sdf: true },
    }),
  ];
  mapState.deck.setProps({ layers });
}

function mapTooltip({ object, layer }) {
  if (!object) return null;
  const style = { background: "#f3efe8", color: "#121110", fontSize: "13px", borderRadius: "6px", padding: "6px 8px" };
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

// ---------- Search page ----------

const KIND_LABELS = { speech: "Said", scene: "Camera saw", item: "Item", text: "Readable text" };

async function runSearch(event) {
  event.preventDefault();
  const q = $("search-q").value.trim();
  const list = $("search-results");
  if (!q) return;
  $("status").textContent = "Searching…";
  try {
    const { results } = await api(`/api/v1/dashboard/search?q=${encodeURIComponent(q)}`);
    $("status").textContent = "";
    if (!results.length) return emptyState(list, "No memories matched. Nothing has been logged about that yet.");
    const tpl = $("result-template");
    list.replaceChildren(
      ...results.map((r, i) => {
        const node = tpl.content.firstElementChild.cloneNode(true);
        fillPhoto(node.querySelector(".photo"), r.image_url, "Photo for this memory");
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
    $("status").textContent = `Search failed: ${err.message}`;
  }
}

// ---------- Insights page ----------

async function loadInsights() {
  const data = await api("/api/v1/dashboard/insights");
  setWindowHours(data.window_hours);
  $("stat-memories").textContent = data.memories_logged;
  $("stat-questions").textContent = data.questions_asked;
  $("stat-unanswered").textContent = data.unanswered_count;
  $("stat-items").textContent = data.items_tracked;
  $("tile-unanswered").classList.toggle("alert", data.unanswered_count > 0);

  renderHighlights(data);
  loadSummary();
  const rows = (list) => list.map((r) => ({ label: capitalize(r.name), value: r.count }));
  Charts.barList($("chart-asked"), rows(data.most_asked), { unit: "times asked", empty: "No questions about specific items yet." });
  Charts.barList($("chart-logged"), rows(data.most_logged), { unit: "times put down", empty: "No items logged yet." });
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
