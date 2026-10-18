// How it works: the three things LastScene does, as a 3 / 2 / 1-column grid.
import { Icon, type IconName } from "../../components/Icons";
import { RevealSection } from "./Reveal";

interface Feature {
  icon: IconName;
  title: string;
  text: string;
}

const FEATURES: Feature[] = [
  { icon: "mic", title: "Press, say, done", text: "Hold a button and say where something is going. The wearable takes a photo at the same moment." },
  { icon: "message", title: "Ask out loud", text: "“Where are my glasses?” gets a spoken answer from what was logged earlier, with the photo to back it up." },
  { icon: "people", title: "Share it if you like", text: "See where each item is, the day scene by scene, and what keeps going missing, floor by floor. Invite anyone you live or work with." },
];

export function Features(): JSX.Element {
  return (
    <RevealSection id="features" className="l-section" labelledBy="features-title">
      <div className="l-section-head">
        <p className="eyebrow">How it works</p>
        <h2 id="features-title" className="l-h2">
          Remembering, without the effort.
        </h2>
      </div>
      <ul className="l-cards" role="list">
        {FEATURES.map((f) => (
          <FeatureCard key={f.title} {...f} />
        ))}
      </ul>
    </RevealSection>
  );
}

function FeatureCard({ icon, title, text }: Feature): JSX.Element {
  return (
    <li className="l-card">
      <span className="feature-icon" aria-hidden="true">
        <Icon name={icon} />
      </span>
      <h3>{title}</h3>
      <p>{text}</p>
    </li>
  );
}
