// Auth0 sign-in for dashboard users (Universal Login + PKCE via auth0-spa-js). The tenant settings
// come from /config.json at runtime, so one build works on every host.
import { createAuth0Client, type User } from "@auth0/auth0-spa-js";
import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

interface Client {
  isAuthenticated(): Promise<boolean>;
  getUser(): Promise<User | undefined>;
  getTokenSilently(): Promise<string>;
  loginWithRedirect(opts: { appState: { hash: string }; authorizationParams?: { screen_hint?: "signup" } }): Promise<void>;
  logout(opts: { logoutParams: { returnTo: string } }): Promise<void>;
  handleRedirectCallback(): Promise<{ appState?: { hash?: string } }>;
}

export interface Auth {
  status: "loading" | "ready" | "error";
  signedIn: boolean;
  user: User | null;
  error: string | null;
  login(): void;
  /** Universal Login opened on its sign-up screen; new accounts land on setup. */
  signup(): void;
  logout(): void;
  getToken(): Promise<string>;
}

const AuthContext = createContext<Auth | null>(null);

export function useAuth(): Auth {
  const auth = useContext(AuthContext);
  if (!auth) throw new Error("useAuth outside AuthProvider");
  return auth;
}

async function createClient(): Promise<Client> {
  // QA builds only (vite build --mode qa): a stand-in client for the local read-only QA
  // server. Signed in unless the URL has ?signedout. Never part of a production build.
  if (import.meta.env.MODE === "qa") {
    const signedIn = !location.search.includes("signedout");
    return {
      isAuthenticated: async () => signedIn,
      getUser: async () => ({ name: "Riley Chen" }),
      getTokenSilently: async () => "qa",
      loginWithRedirect: async () => {},
      logout: async () => {},
      handleRedirectCallback: async () => ({}),
    };
  }
  const config = await fetch("/config.json").then((r) => r.json());
  const client = await createAuth0Client({
    domain: config.domain,
    clientId: config.clientId,
    authorizationParams: { audience: config.audience, redirect_uri: window.location.origin },
    // Keep the session across reloads: silent re-auth via iframe is blocked by most browsers'
    // third-party cookie rules, so without this every reload meant a new login.
    cacheLocation: "localstorage",
    // Don't leave a blank page for the SDK's 60 s default if the silent session check stalls.
    authorizeTimeoutInSeconds: 5,
  });
  return {
    isAuthenticated: () => client.isAuthenticated(),
    getUser: () => client.getUser(),
    getTokenSilently: async () => {
      const token = await client.getTokenSilently({ authorizationParams: { audience: config.audience } });
      if (!token) throw new Error("No access token");
      return token;
    },
    loginWithRedirect: (opts) => client.loginWithRedirect(opts),
    logout: (opts) => client.logout(opts),
    handleRedirectCallback: () => client.handleRedirectCallback(),
  };
}

// Sign-in setup runs once per page load (StrictMode runs effects twice in development, and the
// redirect's one-time code can only be exchanged once).
let started: Promise<{ client: Client; signedIn: boolean; user: User | null }> | null = null;

function start() {
  started ??= (async () => {
    const client = await createClient();
    // Returning from Universal Login.
    const params = new URLSearchParams(window.location.search);
    if (params.has("error")) {
      window.history.replaceState({}, document.title, window.location.pathname);
      throw new Error(params.get("error_description") || params.get("error") || "Login failed");
    }
    if (params.has("code") && params.has("state")) {
      const { appState } = await client.handleRedirectCallback();
      window.history.replaceState({}, document.title, window.location.pathname + (appState?.hash || ""));
      window.dispatchEvent(new Event("hashchange")); // replaceState doesn't fire it; the router listens
    }
    const signedIn = await client.isAuthenticated();
    const user = signedIn ? ((await client.getUser()) ?? null) : null;
    return { client, signedIn, user };
  })();
  return started;
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [client, setClient] = useState<Client | null>(null);
  const [state, setState] = useState<Pick<Auth, "status" | "signedIn" | "user" | "error">>({
    status: "loading",
    signedIn: false,
    user: null,
    error: null,
  });

  useEffect(() => {
    let cancelled = false;
    start().then(({ client: c, signedIn, user }) => {
      if (!cancelled) {
        setClient(c);
        setState({ status: "ready", signedIn, user, error: null });
      }
    }).catch((err: Error) => {
      if (!cancelled) setState({ status: "error", signedIn: false, user: null, error: err.message });
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const auth = useMemo<Auth>(
    () => ({
      ...state,
      login: () => void client?.loginWithRedirect({ appState: { hash: window.location.hash } }),
      signup: () => void client?.loginWithRedirect({ appState: { hash: "#/household" }, authorizationParams: { screen_hint: "signup" } }),
      logout: () => void client?.logout({ logoutParams: { returnTo: window.location.origin } }),
      getToken: async () => {
        if (!client) throw new Error("Not signed in yet");
        try {
          return await client.getTokenSilently();
        } catch (err) {
          // Cached session expired: back through Universal Login (usually instant with SSO).
          const code = (err as { error?: string }).error;
          if (code && ["login_required", "consent_required", "missing_refresh_token"].includes(code)) {
            await client.loginWithRedirect({ appState: { hash: window.location.hash } });
          }
          throw err;
        }
      },
    }),
    [client, state],
  );

  return <AuthContext.Provider value={auth}>{children}</AuthContext.Provider>;
}
