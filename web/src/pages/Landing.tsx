// Signed out: what LastScene is, and the way in. The header is the landing Navbar (Shell).
import { BottomCTA } from "./landing/BottomCTA";
import { Features } from "./landing/Features";
import { Footer } from "./landing/Footer";
import { Hero } from "./landing/Hero";
import { Trust } from "./landing/Trust";

export function Landing(): JSX.Element {
  return (
    <div className="landing">
      <Hero />
      <Features />
      <Trust />
      <BottomCTA />
      <Footer />
    </div>
  );
}
