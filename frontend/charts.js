// LastScene chart kit: small, dependency-free charts that match the dashboard's
// tokens (styles.css → "Insights"). Every chart is plain DOM, has a hover tooltip,
// and an accessible text alternative. See README → "Adding a visualization".
//
// Colors: series use the validated slots --series-1 (blue) and --series-2 (deep
// amber). To add a third series color, validate it against the surface first:
//   node <dataviz skill>/scripts/validate_palette.js "#3987e5,#c98500,#NEW" --mode dark --surface "#1c1b19"
// and add --series-3 / .s3 in styles.css. Never reuse the status colors for data.

const Charts = (() => {
  const el = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  };

  function empty(container, text) {
    container.replaceChildren(el("p", "empty", text));
  }

  // One tooltip element shared by every chart on the page.
  let tip;
  function tooltip(target, text) {
    tip ??= document.getElementById("tooltip") || document.body.appendChild(el("div", "tooltip"));
    target.addEventListener("pointerenter", () => {
      tip.textContent = text;
      tip.hidden = false;
    });
    target.addEventListener("pointermove", (e) => {
      const x = Math.min(e.clientX + 12, window.innerWidth - tip.offsetWidth - 8);
      tip.style.transform = `translate(${x}px, ${e.clientY + 12}px)`;
    });
    target.addEventListener("pointerleave", () => (tip.hidden = true));
  }

  /**
   * Horizontal bars for one series (ranking, top-N).
   * rows: [{ label, value }]   opts: { unit: "times asked", empty: "No data yet." }
   * The card's title names the series, so there is no legend.
   */
  function barList(container, rows, { unit = "", empty: emptyText = "No data yet." } = {}) {
    container.classList.add("bar-list");
    if (!rows.length) return empty(container, emptyText);
    const max = Math.max(...rows.map((r) => r.value), 1);
    container.replaceChildren(
      ...rows.map((r) => {
        const row = el("div", "bar-row");
        const track = el("span", "bar-track");
        const bar = el("span", "bar s1");
        bar.style.width = `${(r.value / max) * 100}%`;
        track.append(bar);
        row.append(el("span", "bar-label", r.label), track, el("span", "bar-value", r.value));
        tooltip(track, `${r.label}: ${r.value} ${unit}`.trim());
        return row;
      })
    );
  }

  /**
   * Grouped vertical columns over shared categories (e.g. time buckets), 1–2 series.
   * data: { categories: ["12–3a", …], series: [{ name: "Things put down", values: [..] }, …] }
   * opts: { label: "Activity by time of day", table: <table> to fill as the text view }
   * Series i is drawn with --series-(i+1); pair the chart with a .legend in the markup.
   */
  function columns(container, { categories, series }, { label = "", table } = {}) {
    container.classList.add("column-chart");
    container.style.gridTemplateColumns = `repeat(${categories.length}, minmax(0, 1fr))`;
    container.setAttribute("role", "img");
    container.setAttribute(
      "aria-label",
      `${label}: ` +
        categories.map((c, i) => `${c}: ${series.map((s) => `${s.values[i]} ${s.name.toLowerCase()}`).join(", ")}`).join("; ")
    );
    const max = Math.max(1, ...series.flatMap((s) => s.values));
    container.replaceChildren(
      ...categories.map((category, i) => {
        const group = el("div", "col-group");
        const bars = el("div", "col-bars");
        series.forEach((s, si) => {
          const n = s.values[i];
          const bar = el("span", `col s${si + 1}`);
          bar.style.height = n ? `${(n / max) * 100}%` : "0";
          const hit = el("span", "col-hit");
          hit.append(bar);
          tooltip(hit, `${category}: ${n} ${s.name.toLowerCase()}`);
          bars.append(hit);
        });
        group.append(bars, el("span", "col-label", category));
        return group;
      })
    );
    if (table) dataTable(table, ["", ...series.map((s) => s.name)], categories.map((c, i) => [c, ...series.map((s) => s.values[i])]));
  }

  /** Fills a <table> — the always-available text view of a chart. */
  function dataTable(table, headers, rows) {
    const head = el("thead");
    const tr = head.insertRow();
    for (const h of headers) tr.append(el("th", "", h));
    const body = el("tbody");
    for (const r of rows) {
      const row = body.insertRow();
      for (const v of r) row.insertCell().textContent = v;
    }
    table.replaceChildren(head, body);
  }

  /** Counts timestamped events into N equal buckets of the viewer's local day. */
  function bucketByHour(events, { buckets = 8, key = "timestamp" } = {}) {
    const size = 24 / buckets;
    const counts = Array(buckets).fill(0);
    for (const e of events) counts[Math.floor(new Date(e[key]).getHours() / size)] += 1;
    return counts;
  }

  return { barList, columns, dataTable, bucketByHour, tooltip, empty };
})();
