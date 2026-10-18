// The first screen: what LastScene does, for whom, and the two ways in.
import { Icon } from "../../components/Icons";
import { useAuth } from "../../lib/auth";
import { SceneCard } from "./SceneCard";
import { sectionLink } from "./sections";

export function Hero(): JSX.Element {
  const { signup, error } = useAuth();
  return (
    <section className="l-hero l-grid" aria-labelledby="hero-title">
      <div className="l-hero-copy">
        <p className="eyebrow">
          <span className="live-dot" aria-hidden="true" />A memory aid for anyone
        </p>
        <h1 id="hero-title" className="l-h1">
          Know where things were left.
        </h1>
        <p className="l-lede">
          A small wearable remembers where the keys, glasses and wallet went, and answers out loud when you ask. You, and anyone you
          share it with, see the day from here.
        </p>
        <div className="l-actions">
          <button type="button" className="primary" onClick={signup}>
            Get started
          </button>
          <a className="pill outline" href="#features" onClick={sectionLink("features")}>
            See how it works
          </a>
        </div>
        <p className="l-note">
          <Icon name="shield" />
          Photos and words are deleted after 72 hours.
        </p>
        {error && <p className="error">Sign-in isn't available right now: {error}</p>}
      </div>
      <div className="l-hero-visual">
        <SceneCard />
      </div>
    </section>
  );
}
