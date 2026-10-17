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
   * rows: [{ label, value }]   opts: { unit: "times asked", empty: "No data yet.", onSelect }
   * With onSelect(row), each row is a button (44px tall) that drills down into that row.
   * The card's title names the series, so there is no legend.
   */
  function barList(container, rows, { unit = "", empty: emptyText = "No data yet.", onSelect } = {}) {
    container.classList.add("bar-list");
    container.classList.toggle("interactive", Boolean(onSelect));
    if (!rows.length) return empty(container, emptyText);
    const max = Math.max(...rows.map((r) => r.value), 1);
    container.replaceChildren(
      ...rows.map((r) => {
        const row = el(onSelect ? "button" : "div", "bar-row");
        if (onSelect) {
          row.type = "button";
          row.setAttribute("aria-label", `${r.label}: ${r.value} ${unit}. Show details`.trim());
          row.addEventListener("click", () => onSelect(r));
        }
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

  /**
   * Fills a <table> — the always-available text view of a chart. Column headers are
   * buttons that sort the rows (numbers numerically), with aria-sort on the header.
   */
  function dataTable(table, headers, rows, { sortable = true } = {}) {
    const head = el("thead");
    const tr = head.insertRow();
    const body = el("tbody");
    let sortCol = -1;
    let dir = "none";
    const cmp = (a, b) =>
      typeof a === "number" && typeof b === "number" ? a - b : String(a).localeCompare(String(b), undefined, { numeric: true });
    const render = () => {
      const sorted = sortCol < 0 ? rows : [...rows].sort((a, b) => cmp(a[sortCol], b[sortCol]) * (dir === "ascending" ? 1 : -1));
      body.replaceChildren();
      for (const r of sorted) {
        const row = body.insertRow();
        for (const v of r) row.insertCell().textContent = v;
      }
    };
    headers.forEach((h, i) => {
      const th = el("th");
      th.scope = "col";
      if (sortable && h) {
        const btn = el("button", "sort-btn", h);
        btn.type = "button";
        btn.addEventListener("click", () => {
          dir = sortCol === i && dir === "ascending" ? "descending" : "ascending";
          sortCol = i;
          for (const other of tr.children) if (other.hasAttribute("aria-sort")) other.setAttribute("aria-sort", "none");
          th.setAttribute("aria-sort", dir);
          render();
        });
        th.setAttribute("aria-sort", "none");
        th.append(btn);
      } else {
        th.textContent = h;
      }
      tr.append(th);
    });
    render();
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
