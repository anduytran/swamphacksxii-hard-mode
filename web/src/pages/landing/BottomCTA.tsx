// The last push: a full-width dark band with the two ways in.
import { useAuth } from "../../lib/auth";
import { RevealSection } from "./Reveal";

export function BottomCTA(): JSX.Element {
  const { signup, login } = useAuth();
  return (
    <RevealSection id="get-started" className="l-cta" labelledBy="cta-title">
      <h2 id="cta-title" className="l-h2">
        Set it up for someone you care about.
      </h2>
      <p>Create a household, add the wearable with the code on its label, and invite the family.</p>
      <div className="l-actions">
        <button type="button" className="primary on-dark" onClick={signup}>
          Get started
        </button>
        <button type="button" className="pill outline on-dark" onClick={login}>
          Log in
        </button>
      </div>
    </RevealSection>
  );
}
