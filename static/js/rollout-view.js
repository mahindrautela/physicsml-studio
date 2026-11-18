// "All rollout steps": one model's whole rollout at a glance.
//
//            t=2   t=3   t=4   ...
//   Truth    [  ]  [  ]  [  ]
//   Model    [  ]  [  ]  [  ]
//
// 2D and 3D data are drawn as small canvas tiles (many tiles, so canvas instead
// of a chart per tile). 1D data becomes two space-time pictures: grid point
// across, time step down. Data comes from /api/runs/<id>/rollout (app/frames.py).
import { api } from './api.js';
import { baseLayout, colorLegend, colorLookup, fieldScale, plotConfig, scale } from './colors.js';
import { $, esc, fmt, sliceShape } from './util.js';

const MIN_TILE = 64;    // px; below this the tiles get hard to read
const MAX_TILE = 150;
const EVERY_CHOICES = [1, 2, 4, 5, 10, 20];

export class RolloutView {
  /** viewer: the Viewer that owns the shared controls (dataset, model, trajectory, field, slice). */
  constructor(viewer) {
    this.v = viewer;
    this.el = {
      note: $('mode-note'), legend: $('rollout-legend'), paper: $('rollout-paper'), grid: $('rollout-grid'),
      readout: $('rollout-readout'), every: $('rollout-every'), everyWrap: $('rollout-every-wrap'),
    };
    this.cache = new Map();
    this.seq = 0;
    this.every = null; // show every n-th step; null = pick automatically from the width
    this.el.every.addEventListener('change', () => { this.every = Number(this.el.every.value); this.draw(); });
    let wait;
    window.addEventListener('resize', () => { clearTimeout(wait); wait = setTimeout(() => this.draw(), 150); });
  }

  reset() {
    this.cache.clear();
    this.data = null;
  }

  params() {
    const s = this.v.selection;
    const p = { dataset: s.dataset, model: s.model, sample: s.sample, field: s.field, component: s.comp, axis: s.axis };
    if (this.v.info().spatial_dims === 3) p.index = s.index;
    return p;
  }

  async update() {
    const s = this.v.selection;
    if (!s.dataset || !s.model) return;
    const params = this.params();
    const key = JSON.stringify(params);
    if (!this.cache.has(key)) this.cache.set(key, api.rollout(this.v.run.id, params).catch((e) => { this.cache.delete(key); throw e; }));
    const seq = ++this.seq;
    try {
      const data = await this.cache.get(key);
      if (seq !== this.seq) return; // a newer request is on its way
      this.data = data;
      this.draw();
    } catch (err) {
      if (seq === this.seq) this.message(`Couldn't load this rollout: ${err.message}`);
    }
  }

  message(text) {
    this.data = null;
    this.el.grid.className = 'rollout-grid';
    this.el.grid.innerHTML = `<p class="stage-msg">${esc(text)}</p>`;
    this.el.legend.replaceChildren();
    this.el.readout.textContent = '';
    this.el.everyWrap.hidden = true;
  }

  // ---------------------------------------------------------------- drawing
  draw() {
    const data = this.data;
    if (!data || this.el.paper.offsetParent === null) return; // nothing loaded, or the view is hidden
    const name = this.v.name(data.model);
    const n = data.steps.length;
    this.el.note.textContent = `Rollout of ${name}: given the true field at t = ${data.start}, it predicts `
      + `${n} step${n === 1 ? '' : 's'}, each from its own previous prediction. Click a step to look at it closely.`;
    const f = fieldScale(data.range);
    const fieldName = this.v.info().field_names[this.v.selection.field];
    this.el.legend.replaceChildren(colorLegend(`${fieldName}: same scale for truth and prediction`, f.kind, f.zmin, f.zmax, fmt));
    this.el.readout.textContent = '';
    if (data.kind === '1d') this.drawSpaceTime(data, f);
    else this.drawTiles(data, f);
  }

  /** Columns shown: every n-th step, always ending with the last one. */
  chooseSteps(count, tileSpace) {
    const fits = (every) => Math.ceil(count / every) * (MIN_TILE + 6) <= tileSpace;
    const every = this.every ?? EVERY_CHOICES.find(fits) ?? EVERY_CHOICES[EVERY_CHOICES.length - 1];
    const options = EVERY_CHOICES.filter((e) => e === 1 || e < count);
    const ordinal = (e) => `${e}${e === 2 ? 'nd' : 'th'}`;
    this.el.every.replaceChildren(...options.map((e) => new Option(e === 1 ? 'all' : `every ${ordinal(e)}`, e)));
    this.el.every.value = String(Math.min(every, options[options.length - 1]));
    this.el.everyWrap.hidden = count < 2;
    const chosen = [];
    for (let i = count - 1; i >= 0; i -= Number(this.el.every.value)) chosen.unshift(i);
    return chosen;
  }

  drawTiles(data, f) {
    const grid = this.el.grid;
    const labelWidth = 130;
    const tileSpace = this.el.paper.clientWidth - 32 - labelWidth;
    const columns = this.chooseSteps(data.steps.length, tileSpace);
    const [rows, cols] = [data.truth[0].length, data.truth[0][0].length];
    const [fullRows, fullCols] = sliceShape(data);
    const tile = Math.max(MIN_TILE, Math.min(MAX_TILE, Math.floor(tileSpace / columns.length) - 6));
    grid.className = 'rollout-grid';
    grid.style.setProperty('--columns', columns.length);
    grid.style.setProperty('--tile', `${tile}px`);
    grid.style.setProperty('--label', `${labelWidth}px`);
    grid.style.setProperty('--aspect', `${cols} / ${rows}`);

    const color = colorLookup(f.kind, f.zmin, f.zmax);
    const cells = ['<span></span>', ...columns.map((i) => `<span class="step-head" title="${data.steps[i] - data.start} steps ahead">t = ${data.steps[i]}</span>`)];
    for (const [label, swatch] of [['Truth', null], [this.v.name(data.model), this.v.color(data.model)]]) {
      cells.push(`<span class="row-label">${swatch ? `<span class="sw" style="background:${swatch}"></span>` : ''}<span class="title" title="${esc(label)}">${esc(label)}</span></span>`);
      cells.push(...columns.map((i) => `<canvas width="${cols}" height="${rows}" data-i="${i}" aria-label="${esc(label)} at t = ${data.steps[i]}"></canvas>`));
    }
    grid.innerHTML = cells.join('');

    const canvases = [...grid.querySelectorAll('canvas')];
    canvases.forEach((canvas, k) => {
      const frames = k < columns.length ? data.truth : data.pred;
      paint(canvas, frames[Number(canvas.dataset.i)], color);
      canvas.addEventListener('mousemove', (ev) => this.hoverTile(ev, canvas, rows, cols, fullRows, fullCols));
      canvas.addEventListener('mouseleave', () => { this.el.readout.textContent = ''; this.highlight(null); });
      canvas.addEventListener('click', () => this.v.showStep(data.steps[Number(canvas.dataset.i)]));
    });
  }

  /** Readout of truth and prediction at the cell under the cursor, and both tiles of that step outlined. */
  hoverTile(ev, canvas, rows, cols, fullRows, fullCols) {
    const box = canvas.getBoundingClientRect();
    const c = Math.min(cols - 1, Math.floor(((ev.clientX - box.left) / box.width) * cols));
    const r = Math.min(rows - 1, Math.floor((1 - (ev.clientY - box.top) / box.height) * rows)); // row 0 is at the bottom
    const i = Number(canvas.dataset.i);
    const d = this.data;
    const truth = d.truth[i][r]?.[c], pred = d.pred[i][r]?.[c];
    // tiles may be thinned out; report positions on the full grid
    const col = c * Math.ceil(fullCols / cols), row = r * Math.ceil(fullRows / rows);
    this.el.readout.textContent = `t = ${d.steps[i]}, column ${col}, row ${row}: truth ${fmt(truth)}; `
      + `${this.v.name(d.model)} ${fmt(pred)} (error ${fmt(pred != null && truth != null ? Math.abs(pred - truth) : null)})`;
    this.highlight(i);
  }

  highlight(i) {
    for (const canvas of this.el.grid.querySelectorAll('canvas')) {
      canvas.classList.toggle('active', i != null && Number(canvas.dataset.i) === i);
    }
  }

  /** 1D data: truth and prediction as pictures with grid point across and time step down. */
  drawSpaceTime(data, f) {
    this.el.everyWrap.hidden = true;
    const grid = this.el.grid;
    grid.className = 'rollout-grid one-d';
    grid.innerHTML = `
      <div class="panel"><h3><span class="title">Truth</span></h3><div class="plot"></div></div>
      <div class="panel"><h3><span class="sw" style="background:${this.v.color(data.model)}"></span><span class="title"></span></h3><div class="plot"></div></div>`;
    grid.querySelectorAll('.title')[1].textContent = this.v.name(data.model);
    const plots = grid.querySelectorAll('.plot');
    const x = data.truth[0].map((_, i) => i);
    const base = baseLayout();
    const layout = () => baseLayout({
      margin: { l: 56, r: 12, t: 8, b: 44 },
      xaxis: { ...base.xaxis, title: { text: 'Grid point' }, showgrid: false },
      yaxis: { ...base.yaxis, title: { text: 'Time step t' }, autorange: 'reversed', showgrid: false, dtick: data.steps.length > 25 ? undefined : 1 },
    });
    [[data.truth, 'truth'], [data.pred, this.v.name(data.model)]].forEach(([z, label], k) => {
      Plotly.newPlot(plots[k], [{
        type: 'heatmap', x, y: data.steps, z, zmin: f.zmin, zmax: f.zmax, colorscale: scale(f.kind), showscale: false,
        hovertemplate: `t = %{y}, point %{x}: ${label} %{z:.4g}<extra></extra>`,
      }], layout(), plotConfig);
      plots[k].on('plotly_click', (ev) => this.v.showStep(ev.points[0].y));
    });
  }
}

/** Draw a 2D array into a canvas, one canvas pixel per value, row 0 at the bottom. */
function paint(canvas, z, color) {
  const rows = z.length, cols = z[0].length;
  const ctx = canvas.getContext('2d');
  const image = ctx.createImageData(cols, rows);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = z[r][c];
      const o = ((rows - 1 - r) * cols + c) * 4;
      if (v == null) { image.data[o + 3] = 0; continue; }
      const [R, G, B] = color(v);
      image.data[o] = R; image.data[o + 1] = G; image.data[o + 2] = B; image.data[o + 3] = 255;
    }
  }
  ctx.putImageData(image, 0, 0);
}
