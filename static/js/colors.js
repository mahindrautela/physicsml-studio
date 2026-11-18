// Color roles. Categorical slots are a validated palette used in fixed order
// (color follows the model, never its rank). Fields use a diverging blue/red
// scale with a gray midpoint when values change sign, otherwise one-hue blue.
// Errors use a separate one-hue orange scale so they never read as the field.

const CAT = {
  light: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'],
  dark: ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#0ca30c', '#9085e9', '#e66767'],
};
export const MAX_SERIES = 8;

const SCALES = {
  light: {
    diverging: ['#0d366b', '#1c5cab', '#2a78d6', '#6da7ec', '#b7d3f6', '#f0efec', '#f6cdc9', '#ee9790', '#e34948', '#b52b2b', '#7c1818'],
    sequential: ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'],
    error: ['#fdf1ea', '#f9cdb3', '#f39f74', '#eb6834', '#c24e1d', '#8a3510'],
  },
  dark: {
    diverging: ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#14365e', '#383835', '#5e1f1d', '#9d2e2c', '#e66767', '#f09a98', '#f9d0ce'],
    sequential: ['#0d366b', '#184f95', '#256abf', '#3987e5', '#6da7ec', '#9ec5f4', '#cde2fb'],
    error: ['#241a14', '#5e2f17', '#a1461d', '#d95926', '#f08c5f', '#f9c4a6'],
  },
};

const mq = window.matchMedia('(prefers-color-scheme: dark)');
export const isDark = () => mq.matches;
export const onSchemeChange = (fn) => mq.addEventListener('change', fn);
const mode = () => (isDark() ? 'dark' : 'light');

export const seriesColor = (i) => CAT[mode()][i % MAX_SERIES];
export const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function toPlotly(stops) {
  return stops.map((c, i) => [i / (stops.length - 1), c]);
}
export const scale = (kind) => toPlotly(SCALES[mode()][kind]);
export const gradient = (kind) => `linear-gradient(90deg, ${SCALES[mode()][kind].join(', ')})`;

// Shared Plotly layout pieces so every chart matches the page.
export function baseLayout(extra = {}) {
  const ink2 = css('--ink-2'), muted = css('--muted'), line = css('--line');
  return {
    font: { family: css('--font') || 'system-ui', size: 12, color: ink2 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    margin: { l: 52, r: 16, t: 8, b: 40 },
    hoverlabel: { font: { family: css('--font'), size: 12 }, bgcolor: css('--panel'), bordercolor: line },
    xaxis: { gridcolor: line, linecolor: css('--line-strong'), zeroline: false, tickfont: { color: muted }, automargin: true },
    yaxis: { gridcolor: line, linecolor: css('--line-strong'), zeroline: false, tickfont: { color: muted }, automargin: true },
    legend: { orientation: 'h', x: 0, y: 1.02, yanchor: 'bottom', font: { size: 12, color: ink2 } },
    ...extra,
  };
}
export const plotConfig = { displayModeBar: false, responsive: true };

// Readable ticks for a log axis: 1-2-5 steps, thinned to at most 4 labels.
export function logTicks(lo, hi) {
  if (!(lo > 0) || !(hi > 0)) return {};
  let vals = [];
  for (let e = Math.floor(Math.log10(lo)) - 1; e <= Math.ceil(Math.log10(hi)) + 1; e++) {
    for (const m of [1, 2, 5]) { const v = m * 10 ** e; if (v >= lo * 0.999 && v <= hi * 1.001) vals.push(v); }
  }
  if (vals.length > 4) {
    const decades = vals.filter((v) => Math.abs(Math.log10(v) % 1) < 1e-9);
    vals = decades.length >= 2 ? decades : vals.filter((_, i) => i % Math.ceil(vals.length / 4) === 0);
  }
  if (vals.length < 2) return {}; // narrow range: let Plotly choose
  return { tickvals: vals, ticktext: vals.map((v) => String(+v.toPrecision(2))) };
}

// ---------------------------------------------------------------- field colour scales
/**
 * Colour scale for a field, from the truth's value range over the whole trajectory.
 * Values that change sign get the diverging blue/red scale centred on zero.
 */
export function fieldScale([lo, hi]) {
  if (lo < 0 && hi > 0) {
    const m = Math.max(-lo, hi);
    return { kind: 'diverging', zmin: -m, zmax: m };
  }
  return { kind: 'sequential', zmin: lo, zmax: hi === lo ? lo + 1e-9 : hi };
}

/**
 * Error scale for one trajectory: the largest error over time, so growth stays visible.
 * It stops at the field's own amplitude; a model that blows up would otherwise wash
 * everyone else out, and an error that large means the prediction is wrong anyway.
 */
export function errorScale([lo, hi], errorMax) {
  const amplitude = Math.max(Math.abs(lo), Math.abs(hi)) || 1e-9;
  const e = errorMax > 0 ? errorMax : 1e-9;
  return e > amplitude ? { emax: amplitude, capped: true } : { emax: e, capped: false };
}

/** A legend: title, then min value, colour ramp, max value. */
export function colorLegend(title, kind, lo, hi, format, capped = false) {
  const div = document.createElement('div');
  div.className = 'legend';
  div.innerHTML = `<b></b><span>${format(lo)}</span><span class="ramp"></span><span>${capped ? '≥ ' : ''}${format(hi)}</span>`;
  div.querySelector('b').textContent = title;
  div.querySelector('.ramp').style.background = gradient(kind);
  return div;
}

/** value -> [r, g, b] for canvas drawing: a 256-entry lookup table built from a scale's colour stops. */
export function colorLookup(kind, zmin, zmax) {
  const stops = SCALES[mode()][kind].map((hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16)));
  const table = Array.from({ length: 256 }, (_, i) => {
    const x = (i / 255) * (stops.length - 1);
    const a = stops[Math.floor(x)], b = stops[Math.min(Math.ceil(x), stops.length - 1)], f = x - Math.floor(x);
    return a.map((c, k) => Math.round(c + (b[k] - c) * f));
  });
  const span = zmax - zmin || 1;
  return (v) => table[Math.max(0, Math.min(255, Math.round(((v - zmin) / span) * 255)))];
}
