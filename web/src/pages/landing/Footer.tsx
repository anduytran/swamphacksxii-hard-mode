// Four columns: the brand, the product, the account, and accessibility. Every link goes
// somewhere real.
import { DisplayButton } from "../../components/DisplaySettings";
import { BrandMark } from "../../components/Icons";
import { useAuth } from "../../lib/auth";
import { sectionLink } from "./sections";

export function Footer(): JSX.Element {
  const { signup, login } = useAuth();
  return (
    <footer className="l-footer">
      <div className="l-footer-brand">
        <a className="brand" href="#/" aria-label="LastScene home">
          <BrandMark />
        </a>
        <p>A memory aid that remembers where things were left.</p>
      </div>
      <nav aria-labelledby="footer-product">
        <h2 id="footer-product">Product</h2>
        <ul role="list">
          <li>
            <a href="#features" onClick={sectionLink("features")}>
              How it works
            </a>
          </li>
          <li>
            <a href="#privacy" onClick={sectionLink("privacy")}>
              Privacy
            </a>
          </li>
          <li>
            <a href="#/public">Public map</a>
          </li>
        </ul>
      </nav>
      <nav aria-labelledby="footer-account">
        <h2 id="footer-account">Account</h2>
        <ul role="list">
          <li>
            <button type="button" className="link-button" onClick={signup}>
              Get started
            </button>
          </li>
          <li>
            <button type="button" className="link-button" onClick={login}>
              Log in
            </button>
          </li>
        </ul>
      </nav>
      <div>
        <h2 id="footer-access">Accessibility</h2>
        <p>Text size, contrast, colours and motion can all be changed.</p>
        <DisplayButton className="pill" withLabel />
      </div>
      <p className="l-footer-legal">© {new Date().getFullYear()} LastScene</p>
    </footer>
  );
}
