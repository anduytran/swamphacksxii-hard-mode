// Display settings dialog: text size, contrast, theme and motion. Each choice applies at once
// (people see the effect while choosing) and is kept on this device.
import { useEffect, useRef, useState } from "react";
import { applyDisplay, DEFAULT_DISPLAY, followDevice, loadDisplay, saveDisplay, type Display } from "../lib/display";
import { Dialog } from "./Dialog";
import { Icon } from "./Icons";

const GROUPS: { key: keyof Display; legend: string; hint: string; options: { value: string; label: string; sample?: string }[] }[] = [
  {
    key: "text",
    legend: "Text size",
    hint: "Your browser's own text size is respected too.",
    options: [
      { value: "standard", label: "Standard", sample: "Aa" },
      { value: "large", label: "Large", sample: "Aa" },
      { value: "largest", label: "Largest", sample: "Aa" },
    ],
  },
  {
    key: "contrast",
    legend: "Contrast",
    hint: "High contrast darkens text and outlines every box.",
    options: [
      { value: "standard", label: "Standard" },
      { value: "high", label: "High" },
    ],
  },
  {
    key: "theme",
    legend: "Colours",
    hint: "Maps stay dark in every theme.",
    options: [
      { value: "system", label: "Match device" },
      { value: "light", label: "Light" },
      { value: "dark", label: "Dark" },
    ],
  },
  {
    key: "motion",
    legend: "Motion",
    hint: "Turns off animations and map fly-overs.",
    options: [
      { value: "system", label: "Match device" },
      { value: "reduce", label: "Reduce" },
    ],
  },
];

/** The current settings, kept in step with the device while "Match device" is chosen. */
export function useDisplay(): [Display, (d: Display) => void] {
  const [display, setDisplay] = useState(loadDisplay);
  const current = useRef(display);
  current.current = display;
  useEffect(() => followDevice(() => current.current), []);
  const set = (d: Display) => {
    setDisplay(d);
    saveDisplay(d);
    applyDisplay(d);
  };
  return [display, set];
}

export function DisplayButton({ className = "icon-btn", withLabel = false }: { className?: string; withLabel?: boolean }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button type="button" className={className} aria-haspopup="dialog" aria-label={withLabel ? undefined : "Display settings"} title="Display settings" onClick={() => setOpen(true)}>
        <Icon name="sliders" />
        {withLabel && "Display"}
      </button>
      {open && <DisplayDialog onClose={() => setOpen(false)} />}
    </>
  );
}

function DisplayDialog({ onClose }: { onClose(): void }) {
  const [display, setDisplay] = useDisplay();
  const changed = JSON.stringify(display) !== JSON.stringify(DEFAULT_DISPLAY);
  return (
    <Dialog open onClose={onClose} className="settings-dialog" labelledBy="display-title">
      <header className="box-head">
        <div>
          <h2 id="display-title">Display</h2>
          <p className="hint">Changes apply straight away and are saved on this device.</p>
        </div>
        <button type="button" className="icon-btn" data-close aria-label="Close display settings">
          <Icon name="close" />
        </button>
      </header>
      {GROUPS.map((g) => (
        <fieldset key={g.key} className="choice-group">
          <legend>{g.legend}</legend>
          <p className="hint" id={`display-${g.key}-hint`}>
            {g.hint}
          </p>
          <div className={`choices choices-${g.options.length}`}>
            {g.options.map((o) => (
              <label key={o.value} className={`choice choice-${g.key}-${o.value}`}>
                <input
                  type="radio"
                  name={`display-${g.key}`}
                  value={o.value}
                  checked={display[g.key] === o.value}
                  aria-describedby={`display-${g.key}-hint`}
                  onChange={() => setDisplay({ ...display, [g.key]: o.value })}
                />
                {o.sample && (
                  <span className="choice-sample" aria-hidden="true">
                    {o.sample}
                  </span>
                )}
                <span className="choice-label">{o.label}</span>
              </label>
            ))}
          </div>
        </fieldset>
      ))}
      <footer className="box-actions">
        <button type="button" className="pill quiet" disabled={!changed} onClick={() => setDisplay(DEFAULT_DISPLAY)}>
          Reset to default
        </button>
        <button type="button" className="primary" data-close>
          Done
        </button>
      </footer>
    </Dialog>
  );
}
