// Signed out: what LastScene is, and the way in.
import { Icon, type IconName } from "../components/Icons";
import { useAuth } from "../lib/auth";

const FEATURES: { icon: IconName; title: string; text: string }[] = [
  { icon: "mic", title: "Press, say, done", text: "Hold a button and say where something is going. The wearable takes a photo at the same time." },
  { icon: "message", title: "Ask out loud", text: "“Where are my glasses?” gets a spoken answer from what was logged earlier." },
  { icon: "people", title: "Family stays in the loop", text: "See where each item is, the day scene by scene, and what keeps going missing." },
];

export function Landing() {
  const { login, error } = useAuth();
  return (
    <section className="landing">
      <div className="hero">
        <div className="hero-copy">
          <p className="eyebrow">
            <span className="live-dot" aria-hidden="true" />A memory aid for families
          </p>
          <h1 className="display hero-title">
            Where things
            <br />
            were left.
          </h1>
          <p className="lede">A small wearable remembers where the keys, glasses and pills went, and answers when someone asks. You see the day from here.</p>
          <div className="signin-actions">
            <button id="login" type="button" className="primary" onClick={login}>
              Log in
            </button>
            <a className="pill link-pill" href="#/public">
              Where people lose things <span aria-hidden="true">→</span>
            </a>
          </div>
        </div>
        <SceneCard />
      </div>

      <ul className="features">
        {FEATURES.map((f) => (
          <li key={f.title}>
            <span className="feature-icon">
              <Icon name={f.icon} />
            </span>
            <h3>{f.title}</h3>
            <p>{f.text}</p>
          </li>
        ))}
      </ul>

      <p className="privacy-line">
        <Icon name="shield" />
        Photos and words are deleted after 72 hours. Private places never leave your household.
      </p>
      {error && <p className="error">Setup error: {error}</p>}
      <div className="landing-cta">
        <button type="button" className="primary" onClick={login}>
          Log in
        </button>
      </div>
    </section>
  );
}

/** The hero illustration: a line-art counter with the keys the camera found. */
function SceneCard() {
  return (
    <figure className="hero-visual" aria-hidden="true">
      <div className="mock-card">
        <div className="mock-photo">
          <svg viewBox="0 0 520 330" preserveAspectRatio="xMidYMid slice">
            <defs>
              <linearGradient id="wall" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0" stopColor="#2d2621" />
                <stop offset="1" stopColor="#1c1815" />
              </linearGradient>
              <linearGradient id="top" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0" stopColor="#4a3b2f" />
                <stop offset="1" stopColor="#2b221b" />
              </linearGradient>
              <radialGradient id="glow" cx="0.82" cy="0.05" r="0.7">
                <stop offset="0" stopColor="#ffc77a" stopOpacity="0.22" />
                <stop offset="1" stopColor="#ffc77a" stopOpacity="0" />
              </radialGradient>
            </defs>
            <rect width="520" height="330" fill="url(#wall)" />
            <rect width="520" height="330" fill="url(#glow)" />
            <rect x="372" y="34" width="104" height="120" rx="6" fill="#ffd9a3" opacity="0.07" />
            <path d="M372 94h104M424 34v120" stroke="#ffd9a3" strokeOpacity="0.08" strokeWidth="3" />
            <path d="M0 214h520v116H0z" fill="url(#top)" />
            <path d="M0 214h520" stroke="#6a5644" strokeWidth="2" />
            {/* mail */}
            <g transform="rotate(-6 96 236)">
              <rect x="52" y="222" width="96" height="30" rx="3" fill="#e9e1d3" opacity="0.85" />
              <path d="M52 224l48 16 48-16" stroke="#b9ad9b" strokeWidth="2" fill="none" />
            </g>
            {/* keys */}
            <circle cx="205" cy="246" r="13" fill="none" stroke="#d8b46a" strokeWidth="3.5" />
            <g transform="rotate(12 222 252)">
              <circle cx="226" cy="252" r="10" fill="#d8b46a" />
              <circle cx="226" cy="252" r="3.5" fill="#3a2e24" />
              <rect x="234" y="248.5" width="44" height="7" rx="2.5" fill="#d8b46a" />
              <rect x="262" y="255" width="5" height="6" fill="#d8b46a" />
              <rect x="270" y="255" width="4" height="4" fill="#d8b46a" />
            </g>
            <g transform="rotate(-28 214 262)">
              <circle cx="210" cy="266" r="8" fill="#b9bec5" />
              <rect x="216" y="263" width="34" height="6" rx="2" fill="#b9bec5" />
            </g>
            {/* mug */}
            <rect x="338" y="164" width="52" height="58" rx="9" fill="#d8d1c4" />
            <path d="M390 178h8a10 10 0 0 1 0 20h-8" stroke="#d8d1c4" strokeWidth="6" fill="none" />
            <path d="M356 150c-6-8 6-12 0-20M370 150c-6-8 6-12 0-20" stroke="#f3efe8" strokeOpacity="0.18" strokeWidth="3" fill="none" strokeLinecap="round" />
            {/* plant */}
            <path d="M428 222l6-40h44l6 40z" fill="#7a5a43" />
            <ellipse cx="444" cy="170" rx="11" ry="24" transform="rotate(-24 444 170)" fill="#56744f" />
            <ellipse cx="468" cy="166" rx="11" ry="26" transform="rotate(20 468 166)" fill="#4b6846" />
            <ellipse cx="456" cy="158" rx="9" ry="28" fill="#628259" />
            {/* what the camera found */}
            <rect x="186" y="226" width="104" height="48" rx="7" fill="#E3A21A" fillOpacity="0.12" stroke="#E3A21A" strokeWidth="2.5" />
            <rect x="186" y="200" width="86" height="21" rx="5" fill="#E3A21A" />
            <text x="196" y="215" fontFamily="Geist Mono, monospace" fontSize="12" fontWeight="600" fill="#2E2A5C">
              keys 94%
            </text>
            {/* viewfinder corners */}
            <path d="M18 40V18h22M480 18h22v22M502 290v22h-22M40 312H18v-22" stroke="#f3efe8" strokeOpacity="0.35" strokeWidth="2" fill="none" />
          </svg>
          <span className="mock-rec">
            <span className="live-dot" />
            Seen 2 min ago
          </span>
        </div>
        <div className="mock-meta">
          <div>
            <strong>Keys</strong>
            <span className="mock-where">Hallway table</span>
          </div>
          <span className="mock-chip">Put down 3×</span>
        </div>
      </div>
      <div className="mock-bubble mock-q">“Where are my keys?”</div>
      <div className="mock-bubble mock-a">On the hallway table, next to the mail.</div>
    </figure>
  );
}
