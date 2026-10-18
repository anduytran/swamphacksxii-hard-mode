// Why people can trust it. In place of testimonials (there are no customers to quote yet):
// three things anyone can check in the product, on a tinted band.
import { Icon, type IconName } from "../../components/Icons";
import { RevealSection } from "./Reveal";

interface Point {
  icon: IconName;
  title: string;
  text: string;
}

const POINTS: Point[] = [
  {
    icon: "lock",
    title: "Private by default",
    text: "Photos and words are deleted after 72 hours. Private places never leave your household, and the public map only shows areas three or more households chose to share.",
  },
  {
    icon: "accessibility",
    title: "Easy for everyone",
    text: "Two big buttons and spoken answers on the wearable. On the dashboard, larger text, high contrast, dark colours and less motion are a click away.",
  },
  {
    icon: "camera",
    title: "Learns your own things",
    text: "Teach it your keys with a few photos, so the camera spots yours, not just any keys. The photos stay until you delete them.",
  },
];

export function Trust(): JSX.Element {
  return (
    <RevealSection id="privacy" className="l-section l-band" labelledBy="trust-title">
      <div className="l-section-head">
        <p className="eyebrow">Built to be trusted</p>
        <h2 id="trust-title" className="l-h2">
          Made for your whole day.
        </h2>
      </div>
      <div className="l-cards">
        {POINTS.map((p) => (
          <article key={p.title} className="l-card l-card-plain">
            <span className="feature-icon" aria-hidden="true">
              <Icon name={p.icon} />
            </span>
            <h3>{p.title}</h3>
            <p>{p.text}</p>
          </article>
        ))}
      </div>
    </RevealSection>
  );
}
