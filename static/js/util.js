// Small helpers used across the page.

export const $ = (id) => document.getElementById(id);

/** Escape text for use inside innerHTML. */
export const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

/** Short, readable number: 4 significant digits, scientific notation for very large/small. */
export function fmt(v) {
  if (v == null || Number.isNaN(v)) return '–';
  const a = Math.abs(v);
  if (a < 1e-12) return '0';
  if (a >= 1e4 || a < 1e-3) return v.toExponential(2);
  return String(+v.toPrecision(4));
}

export const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

export const formatDate = (iso) => new Date(iso).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });

/** A short message at the bottom of the page. */
export function toast(message) {
  const el = $('toast');
  el.textContent = message;
  el.classList.add('show');
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => el.classList.remove('show'), 3200);
}

/** Remember small things (selections, settings) in this browser. Never required: storage may be blocked. */
export const saved = {
  get(key) { try { return JSON.parse(localStorage.getItem(key)); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage unavailable */ } },
};

/** (rows, cols) of the picture a viewer frame shows: the grid for 2D data, the chosen plane for 3D data. */
export function sliceShape(frame) {
  const [D, H, W] = frame.grid;
  if (frame.kind === '2d') return [H, W];
  return { d: [H, W], h: [D, W], w: [D, H] }[frame.axis];
}
