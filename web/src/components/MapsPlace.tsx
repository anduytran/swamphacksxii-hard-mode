// A public place's name from Grounding with Google Maps, with the attribution Google requires:
// the sources (linked) and "Google Maps" right after the name, visible without any extra click,
// "Google Maps" exactly as written, never translated or wrapped. The wearable's spoken answers
// don't use these names, because a speaker can't show the sources.
import type { PlaceName } from "../lib/types";

export function GoogleMaps() {
  return (
    <span className="google-maps" translate="no">
      Google Maps
    </span>
  );
}

export function MapsPlace({ place }: { place: PlaceName }) {
  return (
    <span className="maps-place">
      <span className="maps-place-name">{place.name}</span>{" "}
      <span className="maps-sources">
        (
        {place.sources.map((s, i) => (
          <span key={s.url}>
            {i > 0 && ", "}
            <a href={s.url} target="_blank" rel="noopener noreferrer">
              {s.title}
            </a>
          </span>
        ))}
        {" · "}
        <GoogleMaps />)
      </span>
    </span>
  );
}

/** The page-level notice that Google Maps data is used; only where a name is shown. */
export function MapsNote() {
  return (
    <p className="hint maps-note">
      Place names from <GoogleMaps />.
    </p>
  );
}
