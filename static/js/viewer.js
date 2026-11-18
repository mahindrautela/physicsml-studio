// Field viewer: truth next to each model's prediction and error, a timeline
// whose track is the error curve, and a linked readout under the cursor.
import { api } from './api.js';
import { baseLayout, css, gradient, logTicks, plotConfig, scale, seriesColor } from './colors.js';

const $ = (id) => document.getElementById(id);
export const fmt = (v) => {
  if (v == null || Number.isNaN(v)) return '–';
  const a = Math.abs(v);
  if (a < 1e-12) return '0';
  if (a !== 0 && (a >= 1e4 || a < 1e-3)) return v.toExponential(2);
  return String(+v.toPrecision(4));
};
const PLAY = 'M4 2.5v11l9-5.5z';
const PAUSE = 'M4 2.5h3v11H4zM9 2.5h3v11H9z';

export class Viewer {
  constructor() {
    this.el = {
      dataset: $('v-dataset'), sample: $('v-sample'), field: $('v-field'), comp: $('v-comp'),
      compWrap: $('v-comp-wrap'), axis: $('v-axis'), axisWrap: $('v-axis-wrap'), index: $('v-index'),
      indexWrap: $('v-index-wrap'), note: $('mode-note'), legends: $('legends'), stage: $('stage'),
      readout: $('readout'), play: $('play'), playIcon: $('play-icon'), chart: $('timeline-chart'),
      slider: $('t-slider'), label: $('t-label'),
    };
    this.run = null;
    this.state = { dataset: null, sample: 0, field: 0, comp: 0, axis: 'd', index: null, mode: 'rollout', t: null };
    this.cache = new Map();
    this.layoutKey = null;
    this.seq = 0;
    this.timer = null;
    this.bind();
  }

  // ---------------------------------------------------------------- helpers
  pair(m, d = this.state.dataset) { return this.run.pairs.find((p) => p.model === m && p.dataset === d); }
  done(d = this.state.dataset) { return this.run.models.filter((m) => this.pair(m, d)?.status === 'done'); }
  color(m) { return seriesColor(this.run.models.indexOf(m)); }
  name(m) { return this.run.model_cards[m]?.name || m; }
  info() { return this.run.dataset_info[this.state.dataset]; }

  bounds() {
    const ms = this.done().map((m) => this.pair(m).metrics);
    const T = this.info().time_steps;
    if (this.state.mode === 'rollout') {
      const s0 = ms[0].rollout_start;
      return [s0, s0 + ms[0].rollout_steps];
    }
    return [Math.min(...ms.map((m) => m.ar_order)), T - 1];
  }

  // ---------------------------------------------------------------- events
  bind() {
    const e = this.el;
    const on = (el, key, parse = Number) => el.addEventListener('change', () => {
      this.state[key] = parse(el.value);
      if (key === 'dataset') { this.state.sample = 0; this.state.field = 0; this.state.comp = 0; this.state.index = null; this.layoutKey = null; this.syncControls(); }
      if (key === 'axis') { this.state.index = null; this.layoutKey = null; this.syncControls(); }
      this.update();
    });
    on(e.dataset, 'dataset', String); on(e.sample, 'sample'); on(e.field, 'field'); on(e.comp, 'comp');
    on(e.axis, 'axis', String); on(e.index, 'index');
    document.querySelectorAll('.seg button').forEach((b) => b.addEventListener('click', () => {
      document.querySelectorAll('.seg button').forEach((x) => x.setAttribute('aria-checked', String(x === b)));
      this.state.mode = b.dataset.mode;
      this.syncControls();
      this.update();
    }));
    e.slider.addEventListener('input', () => { this.state.t = Number(e.slider.value); this.update(); });
    e.play.addEventListener('click', () => (this.timer ? this.stop() : this.start()));
    let rt;
    window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => this.resize(), 150); });
  }

  start() {
    if (!this.run || !this.state.dataset) return;
    const [lo, hi] = this.bounds();
    if (this.state.t >= hi) this.state.t = lo;
    this.el.playIcon.setAttribute('d', PAUSE);
    this.el.play.setAttribute('aria-label', 'Pause');
    this.timer = setInterval(() => {
      const [, h] = this.bounds();
      if (this.state.t >= h) return this.stop();
      this.state.t += 1;
      this.update();
    }, 450);
  }

  stop() {
    clearInterval(this.timer);
    this.timer = null;
    this.el.playIcon.setAttribute('d', PLAY);
    this.el.play.setAttribute('aria-label', 'Play');
  }

  // ---------------------------------------------------------------- run updates
  setRun(run) {
    const fresh = !this.run || this.run.id !== run.id;
    if (fresh) {
      this.stop();
      this.state = { ...this.state, dataset: null, sample: 0, field: 0, comp: 0, index: null, t: null };
      this.cache.clear();
      this.layoutKey = null;
      this.doneKey = null;
    }
    this.run = run;
    const avail = run.datasets.filter((d) => run.dataset_info[d] && this.done(d).length);
    const opts = avail.map((d) => [d, run.dataset_cards[d]?.name || d]);
    fillSelect(this.el.dataset, opts);
    if (!avail.length) {
      this.layoutKey = null;
      const busy = ['queued', 'running'].includes(run.status);
      this.message(busy ? 'Fields appear here as soon as the first model finishes on a dataset.'
        : 'No model finished on any dataset in this run. Check the messages above or the run log.');
      this.el.legends.replaceChildren();
      Plotly.purge(this.el.chart);
      return;
    }
    if (!avail.includes(this.state.dataset)) {
      this.state.dataset = avail[0];
      this.state.sample = 0; this.state.field = 0; this.state.comp = 0; this.state.t = null;
    }
    this.el.dataset.value = this.state.dataset;
    const doneKey = JSON.stringify(avail.map((d) => this.done(d)));
    if (fresh || doneKey !== this.doneKey) {
      this.doneKey = doneKey;
      this.cache.clear();
      this.layoutKey = null;
      this.syncControls();
      this.update();
    }
  }

  syncControls() {
    const info = this.info();
    const s = this.state;
    const nViz = Math.min(this.run.settings.viz_samples, info.eval_indices.length);
    fillSelect(this.el.sample, Array.from({ length: nViz }, (_, i) => [i, `#${info.eval_indices[i]}`]));
    fillSelect(this.el.field, info.field_names.map((f, i) => [i, f]));
    fillSelect(this.el.comp, info.component_names.map((c, i) => [i, c]));
    this.el.compWrap.hidden = info.component_names.length < 2;
    const is3d = info.spatial_dims === 3;
    this.el.axisWrap.hidden = !is3d;
    this.el.indexWrap.hidden = !is3d;
    if (is3d) {
      const [D, H, W] = info.grid;
      const size = { d: D, h: H, w: W }[s.axis];
      this.el.index.max = size - 1;
      if (s.index == null) s.index = Math.floor(size / 2);
      this.el.index.value = s.index;
    }
    this.el.sample.value = s.sample; this.el.field.value = s.field; this.el.comp.value = s.comp;
    const [lo, hi] = this.bounds();
    this.el.slider.min = lo; this.el.slider.max = hi;
    if (s.t == null || s.t < lo || s.t > hi) s.t = s.mode === 'rollout' ? Math.min(lo + 1, hi) : lo;
    this.el.slider.value = s.t;
    const ms = this.pair(this.done()[0]).metrics;
    this.el.note.textContent = s.mode === 'rollout'
      ? `Rollout: each model gets the true field at t = ${ms.rollout_start} and then predicts ${ms.rollout_steps} steps by feeding its own predictions back in. Errors build up over time.`
      : 'One-step: every frame is predicted from the true frame(s) just before it, so errors don\'t accumulate.';
  }

  params(t) {
    const s = this.state;
    const p = { dataset: s.dataset, models: this.done().join(','), sample: s.sample, t, mode: s.mode,
      field: s.field, component: s.comp, axis: s.axis };
    if (this.info().spatial_dims === 3) p.index = s.index;
    return p;
  }

  frame(t) {
    const p = this.params(t);
    const key = JSON.stringify(p);
    if (!this.cache.has(key)) {
      if (this.cache.size > 240) this.cache.delete(this.cache.keys().next().value);
      const pr = api.frame(this.run.id, p);
      pr.catch(() => this.cache.delete(key));
      this.cache.set(key, pr);
    }
    return this.cache.get(key);
  }

  async update() {
    if (!this.run || !this.state.dataset) return;
    const seq = ++this.seq;
    const t = this.state.t;
    this.el.slider.value = t;
    let data;
    try {
      data = await this.frame(t);
    } catch (err) {
      if (seq === this.seq) this.message(`Couldn't load this frame: ${err.message}`);
      return;
    }
    if (seq !== this.seq) return;
    this.render(data);
    this.renderTimeline();
    if (this.timer) this.frame(Math.min(t + 1, this.bounds()[1])); // prefetch next frame while playing
  }

  message(text) {
    const p = document.createElement('p');
    p.className = 'stage-msg';
    p.textContent = text;
    this.el.stage.classList.remove('wide');
    this.el.stage.replaceChildren(p);
    this.el.readout.textContent = '';
  }

  // ---------------------------------------------------------------- rendering
  render(data) {
    this.data = data;
    const models = this.done();
    const key = [this.state.dataset, models.join(), data.kind, data.axis, this.state.mode].join('|');
    if (key !== this.layoutKey) {
      this.layoutKey = key;
      data.kind === '1d' ? this.build1d(models) : this.build2d(models, data);
    }
    data.kind === '1d' ? this.draw1d(data, models) : this.draw2d(data, models);
    const [lo] = this.bounds();
    const ahead = this.state.mode === 'rollout' ? this.state.t - lo : null;
    this.el.label.innerHTML = `t = ${this.state.t}<br><small>${ahead == null ? 'one step' : ahead === 0 ? 'starting frame' : `${ahead} step${ahead > 1 ? 's' : ''} ahead`}</small>`;
  }

  fieldScale(data) {
    const [lo, hi] = data.range;
    if (lo < 0 && hi > 0) {
      const m = Math.max(-lo, hi);
      return { zmin: -m, zmax: m, colorscale: scale('diverging'), ramp: gradient('diverging') };
    }
    return { zmin: lo, zmax: hi === lo ? lo + 1e-9 : hi, colorscale: scale('sequential'), ramp: gradient('sequential') };
  }

  sliceShape(data) {
    const [D, H, W] = data.grid;
    if (data.kind === '2d') return [H, W];
    return { d: [H, W], h: [D, W], w: [D, H] }[data.axis];
  }

  build2d(models, data) {
    const stage = this.el.stage;
    stage.classList.remove('wide');
    stage.replaceChildren();
    const [rows, cols] = this.sliceShape(data);
    stage.style.setProperty('--aspect', `${cols} / ${rows}`);
    this.panels = {};
    const mk = (title, color, id) => {
      const panel = document.createElement('div');
      panel.className = 'panel';
      panel.style.position = 'relative';
      const h = document.createElement('h3');
      if (color) { const sw = document.createElement('span'); sw.className = 'sw'; sw.style.background = color; h.append(sw); }
      h.append(title);
      const small = document.createElement('small');
      h.append(small);
      const plot = document.createElement('div');
      plot.className = 'plot';
      const na = document.createElement('div');
      na.className = 'na';
      na.style.cssText = 'position:absolute; left:6px; right:6px; bottom:4px;';
      na.hidden = true;
      panel.append(h, plot, na);
      this.panels[id] = { panel, plot, na, small };
      return panel;
    };
    const truthCol = document.createElement('div');
    truthCol.className = 'col';
    truthCol.append(mk('Truth', null, 'truth'));
    stage.append(truthCol);
    for (const m of models) {
      const col = document.createElement('div');
      col.className = 'col';
      col.append(mk(this.name(m), this.color(m), `p:${m}`), mk('Absolute error', null, `e:${m}`));
      stage.append(col);
    }
    this.sizeColumns(models.length + 1);
    for (const { plot } of Object.values(this.panels)) {
      Plotly.newPlot(plot, [], this.heatLayout(rows, cols), { ...plotConfig, staticPlot: false });
      plot.on('plotly_hover', (ev) => this.hover(ev.points[0].x, ev.points[0].y));
      plot.on('plotly_unhover', () => this.hover(null));
    }
  }

  sizeColumns(n) {
    const avail = this.el.stage.clientWidth - 32;
    const w = Math.max(170, Math.min(340, Math.floor((avail - 16 * (n - 1)) / n)));
    this.el.stage.style.setProperty('--panel-w', `${w}px`);
  }

  resize() {
    if (!this.panels || !this.data) return;
    if (this.data.kind !== '1d') this.sizeColumns(this.done().length + 1);
    for (const { plot } of Object.values(this.panels)) Plotly.Plots.resize(plot);
    Plotly.Plots.resize(this.el.chart);
  }

  heatLayout(rows, cols) {
    return {
      margin: { l: 0, r: 0, t: 0, b: 0 }, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      xaxis: { visible: false, range: [-0.5, cols - 0.5], fixedrange: true },
      yaxis: { visible: false, range: [-0.5, rows - 0.5], fixedrange: true },
      shapes: [],
    };
  }

  draw2d(data, models) {
    const f = this.fieldScale(data);
    const { emax, capped } = errorScale(data);
    const [rows, cols] = this.sliceShape(data);
    const info = this.info();
    const fieldName = info.field_names[this.state.field];
    this.el.legends.innerHTML = '';
    this.el.legends.append(
      legend(`${fieldName}: same scale for truth and all predictions`, f.ramp, f.zmin, f.zmax),
      legend(capped ? 'Absolute error |prediction − truth|: errors larger than the field itself show in the darkest color'
        : 'Absolute error |prediction − truth|: same scale at every time step', gradient('error'), 0, emax, capped));
    const heat = (z, zmin, zmax, colorscale) => [{ type: 'heatmap', z, zmin, zmax, colorscale, showscale: false, hoverinfo: 'none' }];
    const layout = () => this.heatLayout(rows, cols);
    const tp = this.panels.truth;
    Plotly.react(tp.plot, heat(data.truth, f.zmin, f.zmax, f.colorscale), layout(), plotConfig);
    tp.small.textContent = '';
    const [lo] = this.bounds();
    for (const m of models) {
      const p = this.panels[`p:${m}`], e = this.panels[`e:${m}`];
      const d = data.models[m];
      const k = this.pair(m).metrics.ar_order;
      const msg = this.state.mode === 'rollout' && this.state.t === lo
        ? 'Starting frame: the model is given the true field here'
        : `No prediction at t = ${this.state.t}: this model needs ${k} earlier frame${k > 1 ? 's' : ''}`;
      for (const [panel, z, zmin, zmax, cs] of [[p, d?.pred, f.zmin, f.zmax, f.colorscale], [e, d?.error, 0, emax, scale('error')]]) {
        panel.na.hidden = !!d;
        panel.na.textContent = d ? '' : msg;
        if (d) panel.na.style.height = `${panel.plot.clientHeight}px`;
        Plotly.react(panel.plot, d ? heat(z, zmin, zmax, cs) : [], layout(), plotConfig);
      }
      p.small.textContent = d ? `NRMSE ${fmt(frameNrmse(d.error, data.truth))}` : '';
      e.small.textContent = d ? `max ${fmt(maxOf(d.error))}` : '';
    }
    if (this.hoverAt) this.hover(...this.hoverAt);
  }

  hover(x, y) {
    if (!this.data || this.data.kind === '1d') return;
    if (x == null) {
      this.hoverAt = null;
      this.el.readout.textContent = '';
      for (const { plot } of Object.values(this.panels)) Plotly.relayout(plot, { shapes: [] });
      return;
    }
    this.hoverAt = [x, y];
    const d = this.data;
    const parts = [`column ${x}, row ${y}`, `truth ${fmt(d.truth[y]?.[x])}`];
    for (const m of this.done()) {
      const md = d.models[m];
      if (md) parts.push(`${this.name(m)} ${fmt(md.pred[y]?.[x])} (error ${fmt(md.error[y]?.[x])})`);
    }
    this.el.readout.textContent = parts.join('; ');
    const shape = { type: 'rect', x0: x - 0.5, x1: x + 0.5, y0: y - 0.5, y1: y + 0.5, line: { color: css('--ink'), width: 1.5 } };
    cancelAnimationFrame(this.raf);
    this.raf = requestAnimationFrame(() => {
      for (const { plot } of Object.values(this.panels)) Plotly.relayout(plot, { shapes: [shape] });
    });
  }

  build1d() {
    const stage = this.el.stage;
    stage.classList.add('wide');
    stage.replaceChildren();
    this.panels = {};
    for (const [id, title] of [['field', 'Field'], ['error', 'Absolute error |prediction − truth|']]) {
      const panel = document.createElement('div');
      panel.className = 'panel';
      const h = document.createElement('h3');
      h.textContent = title;
      const plot = document.createElement('div');
      plot.className = 'plot';
      panel.append(h, plot);
      stage.append(panel);
      this.panels[id] = { panel, plot };
    }
    this.el.readout.textContent = '';
  }

  draw1d(data, models) {
    this.el.legends.replaceChildren();
    const x = data.truth.map((_, i) => i);
    const [lo, hi] = data.range;
    const pad = 0.08 * (hi - lo || 1);
    const field = [{ x, y: data.truth, name: 'Truth', mode: 'lines', line: { color: css('--ink'), width: 2.5 },
      hovertemplate: 'truth %{y:.4g}<extra></extra>' }];
    const err = [];
    for (const m of models) {
      const d = data.models[m];
      if (!d) continue;
      const line = { color: this.color(m), width: 2 };
      field.push({ x, y: d.pred, name: this.name(m), mode: 'lines', line, hovertemplate: `${this.name(m)} %{y:.4g}<extra></extra>` });
      err.push({ x, y: d.error, name: this.name(m), mode: 'lines', line, showlegend: false, hovertemplate: `${this.name(m)} %{y:.4g}<extra></extra>` });
    }
    const fname = this.info().field_names[this.state.field];
    Plotly.react(this.panels.field.plot, field, baseLayout({
      hovermode: 'x unified', yaxis: { ...baseLayout().yaxis, range: [lo - pad, hi + pad], title: { text: fname } },
      xaxis: { ...baseLayout().xaxis, title: { text: 'Grid point' } } }), plotConfig);
    Plotly.react(this.panels.error.plot, err, baseLayout({
      hovermode: 'x unified', showlegend: false,
      yaxis: { ...baseLayout().yaxis, range: [0, errorScale(data).emax * 1.05], title: { text: 'Absolute error' } },
      xaxis: { ...baseLayout().xaxis, title: { text: 'Grid point' } } }), plotConfig);
    const missing = models.filter((m) => !data.models[m]).map((m) => this.name(m));
    this.el.readout.textContent = missing.length ? `No prediction at this step for: ${missing.join(', ')}` : '';
  }

  renderTimeline() {
    const [lo, hi] = this.bounds();
    const traces = [];
    for (const m of this.done()) {
      const met = this.pair(m).metrics;
      const block = this.state.mode === 'rollout' ? met.rollout : met.onestep;
      const curve = block?.per_step_nrmse;
      if (!curve) continue;
      const x0 = this.state.mode === 'rollout' ? met.rollout_start + 1 : met.onestep_from;
      traces.push({ x: curve.map((_, i) => x0 + i), y: curve, mode: 'lines', name: this.name(m),
        line: { color: this.color(m), width: 2 }, hovertemplate: `${this.name(m)} %{y:.3g}<extra></extra>` });
    }
    const ys = traces.flatMap((tr) => tr.y).filter((y) => y > 0);
    const ticks = ys.length ? logTicks(Math.min(...ys), Math.max(...ys)) : {};
    const layout = baseLayout({
      margin: { l: 44, r: 16, t: 6, b: 20 }, showlegend: false, hovermode: 'x unified',
      xaxis: { ...baseLayout().xaxis, range: [lo, hi], dtick: hi - lo > 30 ? undefined : 1, automargin: false, showgrid: false },
      yaxis: { ...baseLayout().yaxis, automargin: false, type: 'log', ...ticks, title: { text: 'NRMSE (log)', font: { size: 10.5 } } },
      shapes: [{ type: 'line', x0: this.state.t, x1: this.state.t, yref: 'paper', y0: 0, y1: 1, line: { color: css('--ink'), width: 1.5 } }],
    });
    if (!this.chartReady) {
      Plotly.newPlot(this.el.chart, traces, layout, plotConfig);
      this.el.chart.on('plotly_click', (ev) => {
        const t = Math.round(ev.points[0].x);
        const [a, b] = this.bounds();
        this.state.t = Math.max(a, Math.min(b, t));
        this.update();
      });
      this.chartReady = true;
    } else {
      Plotly.react(this.el.chart, traces, layout, plotConfig);
    }
  }
}

function fillSelect(sel, options) {
  const sig = JSON.stringify(options);
  if (sel.dataset.sig === sig) return;
  sel.dataset.sig = sig;
  const v = sel.value;
  sel.replaceChildren(...options.map(([value, label]) => new Option(label, value)));
  if (options.some(([value]) => String(value) === v)) sel.value = v;
}

// Errors are drawn on one scale for the whole trajectory, so growth is visible while
// playing. A model that blows up would wash everyone else out, so the scale stops at
// the field's own amplitude: an error that large means the prediction is wrong anyway.
function errorScale(data) {
  const amp = Math.max(Math.abs(data.range[0]), Math.abs(data.range[1])) || 1e-9;
  const e = data.error_max > 0 ? data.error_max : 1e-9;
  return e > amp ? { emax: amp, capped: true } : { emax: e, capped: false };
}

function legend(title, ramp, lo, hi, capped = false) {
  const div = document.createElement('div');
  div.className = 'legend';
  div.innerHTML = `<b></b><span>${fmt(lo)}</span><span class="ramp"></span><span>${capped ? '≥ ' : ''}${fmt(hi)}</span>`;
  div.querySelector('b').textContent = title;
  div.querySelector('.ramp').style.background = ramp;
  return div;
}

function frameNrmse(err, truth) {
  let se = 0, st = 0;
  for (let i = 0; i < err.length; i++) {
    for (let j = 0; j < err[i].length; j++) {
      const e = err[i][j], t = truth[i][j];
      if (e == null || t == null) continue;
      se += e * e; st += t * t;
    }
  }
  return Math.sqrt(se / (st || 1e-12));
}

function maxOf(a) {
  let m = 0;
  for (const row of a) for (const v of row) if (v != null && v > m) m = v;
  return m;
}
