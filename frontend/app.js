let auth0Client;
let audience;

const $ = (id) => document.getElementById(id);
const PAGES = ["items", "timeline", "insights"];
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

const loaded = {};

async function showPage(name, { force = false } = {}) {
  for (const p of PAGES) $(`page-${p}`).hidden = p !== name;
  for (const a of document.querySelectorAll("#tabs a")) {
    if (a.dataset.page === name) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  }
  if (loaded[name] && !force) return;
  const loaders = { items: loadItems, timeline: loadTimeline, insights: loadInsights };
  const status = $("status");
  status.textContent = "Loading…";
  try {
    await loaders[name]();
    loaded[name] = true;
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
