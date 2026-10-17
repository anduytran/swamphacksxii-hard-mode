let auth0Client;
let audience;

const $ = (id) => document.getElementById(id);

async function init() {
  const config = await fetch("/config.json").then((r) => r.json());
  audience = config.audience;
  auth0Client = await auth0.createAuth0Client({
    domain: config.domain,
    clientId: config.clientId,
    authorizationParams: { audience, redirect_uri: window.location.origin },
    cacheLocation: "memory",
    useRefreshTokens: true,
  });

  // Returning from Universal Login.
  const params = new URLSearchParams(window.location.search);
  if (params.has("code") && params.has("state")) {
    await auth0Client.handleRedirectCallback();
    window.history.replaceState({}, document.title, window.location.pathname);
  }

  $("login").onclick = () => auth0Client.loginWithRedirect();
  $("logout").onclick = () =>
    auth0Client.logout({ logoutParams: { returnTo: window.location.origin } });
  $("refresh").onclick = loadHistory;

  if (await auth0Client.isAuthenticated()) {
    const user = await auth0Client.getUser();
    $("user-name").textContent = user?.name || user?.email || "";
    $("user-bar").hidden = false;
    $("logged-in").hidden = false;
    await loadHistory();
  } else {
    $("logged-out").hidden = false;
  }
}

async function loadHistory() {
  const status = $("status");
  status.textContent = "Loading…";
  try {
    const token = await auth0Client.getTokenSilently({ authorizationParams: { audience } });
    const res = await fetch("/api/v1/dashboard/history", {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!res.ok) throw new Error(`${res.status} ${(await res.json()).detail || res.statusText}`);
    render(await res.json());
    status.textContent = "";
  } catch (err) {
    status.textContent = `Couldn't load history: ${err.message}`;
  }
}

function render(entries) {
  const feed = $("feed");
  feed.replaceChildren();
  if (!entries.length) {
    $("status").textContent = "Nothing recorded in the last 24 hours.";
    return;
  }
  const tpl = $("entry-template");
  for (const e of entries) {
    const node = tpl.content.cloneNode(true);
    const img = node.querySelector("img");
    if (e.image_url) {
      img.src = e.image_url;
      img.alt = `Photo taken when they said: ${e.transcript}`;
    } else {
      img.remove();
    }
    const time = node.querySelector("time");
    const when = new Date(e.timestamp);
    time.dateTime = e.timestamp;
    time.textContent = when.toLocaleString(undefined, {
      weekday: "short", hour: "numeric", minute: "2-digit",
    });
    node.querySelector(".transcript").textContent = `“${e.transcript}”`;
    feed.appendChild(node);
  }
}

init().catch((err) => {
  $("logged-out").hidden = false;
  $("logged-out").insertAdjacentHTML("beforeend", "<p class='error'></p>");
  $("logged-out").querySelector(".error").textContent = `Setup error: ${err.message}`;
});
