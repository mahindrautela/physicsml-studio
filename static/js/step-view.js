// "One step at a time": the truth next to each model's prediction and error at one
// time step, a timeline drawn on top of the error curves, and a readout under the cursor.
//
// Data comes from /api/runs/<id>/frame (app/frames.py), one time step per request.
import { api } from './api.js';
import { baseLayout, colorLegend, css, errorScale, fieldScale, logTicks, plotConfig, scale } from './colors.js';
import { $, fmt, sliceShape } from './util.js';

const PLAY = 'M4 2.5v11l9-5.5z';
const PAUSE = 'M4 2.5h3v11H4zM9 2.5h3v11H9z';

export class StepView {
  /** viewer: the Viewer that owns the shared controls (dataset, trajectory, field, slice). */
  constructor(viewer) {
    this.v = viewer;
    this.el = {
      note: $('mode-note'), legends: $('legends'), stage: $('stage'), readout: $('readout'),
      play: $('play'), playIcon: $('play-icon'), chart: $('timeline-chart'), slider: $('t-slider'), label: $('t-label'),
      modeButtons: [...document.querySelectorAll('#mode-switch button')],
    };
    this.mode = 'rollout';  // 'rollout' or 'onestep'
    this.t = null;          // time step shown
    this.timer = null;      // playback
    this.cache = new Map();
    this.seq = 0;
    this.reset();
    this.bind();
  }

  /** Start over: forget data, drawings and the time position. */
  reset() {
    this.stop();
    this.forgetData();
    this.chartReady = false;
    this.t = null;
    if (window.Plotly) Plotly.purge(this.el.chart);
  }

  /** Another model finished: fetch and lay out again, but stay at the same time step. */
  forgetData() {
    this.cache.clear();
    this.layoutKey = null;
  }

  bind() {
    for (const button of this.el.modeButtons) {
      button.addEventListener('click', () => { this.mode = button.dataset.mode; this.update(); });
    }
    this.el.slider.addEventListener('input', () => { this.t = Number(this.el.slider.value); this.update(); });
    this.el.play.addEventListener('click', () => (this.timer ? this.stop() : this.start()));
    let wait;
    window.addEventListener('resize', () => { clearTimeout(wait); wait = setTimeout(() => this.resize(), 150); });
  }

  /** Show a given time step in a given mode (used when a tile in the rollout grid is clicked). */
  goTo(mode, t) {
    this.mode = mode;
    this.t = t;
  }

  // ---------------------------------------------------------------- time
  bounds() {
    const metrics = this.v.done().map((m) => this.v.pair(m).metrics);
    if (this.mode === 'rollout') {
      const start = metrics[0].rollout_start;
      return [start, start + metrics[0].rollout_steps];
    }
    return [Math.min(...metrics.map((m) => m.ar_order)), this.v.info().time_steps - 1];
  }

  start() {
    if (!this.v.selection.dataset) return;
    const [lo, hi] = this.bounds();
    if (this.t >= hi) this.t = lo;
    this.el.playIcon.setAttribute('d', PAUSE);
    this.el.play.setAttribute('aria-label', 'Pause');
    this.timer = setInterval(() => {
      if (this.t >= this.bounds()[1]) return this.stop();
      this.t += 1;
      this.update();
    }, 450);
  }

  stop() {
    clearInterval(this.timer);
    this.timer = null;
    this.el.playIcon.setAttribute('d', PLAY);
    this.el.play.setAttribute('aria-label', 'Play');
  }

  // ---------------------------------------------------------------- data
  frame(t) {
    const s = this.v.selection;
    const params = { dataset: s.dataset, models: this.v.done().join(','), sample: s.sample, t, mode: this.mode,
      field: s.field, component: s.comp, axis: s.axis };
    if (this.v.info().spatial_dims === 3) params.index = s.index;
    const key = JSON.stringify(params);
    if (!this.cache.has(key)) {
      if (this.cache.size > 240) this.cache.delete(this.cache.keys().next().value);
      const request = api.frame(this.v.run.id, params);
      request.catch(() => this.cache.delete(key));
      this.cache.set(key, request);
    }
    return this.cache.get(key);
  }

  async update() {
    if (!this.v.selection.dataset) return;
    this.syncTime();
    const seq = ++this.seq;
    let data;
    try {
      data = await this.frame(this.t);
    } catch (err) {
      if (seq === this.seq) this.message(`Couldn't load this frame: ${err.message}`);
      return;
    }
    if (seq !== this.seq) return; // a newer request is on its way
    this.render(data);
    this.renderTimeline();
    if (this.timer) this.frame(Math.min(this.t + 1, this.bounds()[1])); // prefetch while playing
  }

  syncTime() {
    for (const b of this.el.modeButtons) b.setAttribute('aria-checked', String(b.dataset.mode === this.mode));
    const [lo, hi] = this.bounds();
    if (this.t == null || this.t < lo || this.t > hi) this.t = this.mode === 'rollout' ? Math.min(lo + 1, hi) : lo;
    Object.assign(this.el.slider, { min: lo, max: hi, value: this.t });
    const m = this.v.pair(this.v.done()[0]).metrics;
    this.el.note.textContent = this.mode === 'rollout'
      ? `Rollout: each model gets the true field at t = ${m.rollout_start} and then predicts ${m.rollout_steps} steps by feeding its own predictions back in. Errors build up over time.`
      : 'One-step: every frame is predicted from the true frame(s) just before it, so errors don\'t accumulate.';
  }

  message(text) {
    const p = document.createElement('p');
    p.className = 'stage-msg';
    p.textContent = text;
    this.el.stage.classList.remove('wide');
    this.el.stage.replaceChildren(p);
    this.el.readout.textContent = '';
    this.el.legends.replaceChildren();
    this.layoutKey = null;
  }

  // ---------------------------------------------------------------- drawing
  render(data) {
    this.data = data;
    const models = this.v.done();
    const key = [this.v.selection.dataset, models.join(), data.kind, data.axis, this.mode].join('|');
    if (key !== this.layoutKey) {
      this.layoutKey = key;
      data.kind === '1d' ? this.build1d() : this.build2d(models, data);
    }
    data.kind === '1d' ? this.draw1d(data, models) : this.draw2d(data, models);
    const ahead = this.mode === 'rollout' ? this.t - this.bounds()[0] : null;
    const sub = ahead == null ? 'one step' : ahead === 0 ? 'starting frame' : `${ahead} step${ahead > 1 ? 's' : ''} ahead`;
    this.el.label.innerHTML = `t = ${this.t}<br><small>${sub}</small>`;
  }

  build2d(models, data) {
    const stage = this.el.stage;
    stage.classList.remove('wide');
    stage.replaceChildren();
    const [rows, cols] = sliceShape(data);
    stage.style.setProperty('--aspect', `${cols} / ${rows}`);
    this.panels = {};
    const column = (...panels) => {
      const col = document.createElement('div');
      col.className = 'col';
      col.append(...panels);
      stage.append(col);
    };
    column(this.panel('truth', 'Truth'));
    for (const m of models) column(this.panel(`p:${m}`, this.v.name(m), this.v.color(m)), this.panel(`e:${m}`, 'Absolute error'));
    this.sizeColumns(models.length + 1);
    for (const { plot } of Object.values(this.panels)) {
      Plotly.newPlot(plot, [], heatLayout(rows, cols), plotConfig);
      plot.on('plotly_hover', (ev) => this.hover(ev.points[0].x, ev.points[0].y));
      plot.on('plotly_unhover', () => this.hover(null));
    }
  }

  /** One titled panel with a plot and an overlay for "no prediction here". */
  panel(id, title, color) {
    const panel = document.createElement('div');
    panel.className = 'panel';
    panel.innerHTML = `<h3>${color ? `<span class="sw" style="background:${color}"></span>` : ''}<span class="title"></span><small></small></h3>
      <div class="plot-wrap"><div class="plot"></div><div class="na" hidden></div></div>`;
    const name = panel.querySelector('.title');
    name.textContent = title;
    name.title = title; // full name on hover when it's cut short
    this.panels[id] = { panel, plot: panel.querySelector('.plot'), na: panel.querySelector('.na'), small: panel.querySelector('small') };
    return panel;
  }

  sizeColumns(n) {
    const avail = this.el.stage.clientWidth - 32;
    const width = Math.max(170, Math.min(340, Math.floor((avail - 16 * (n - 1)) / n)));
    this.el.stage.style.setProperty('--panel-w', `${width}px`);
  }

  resize() {
    if (!this.panels || !this.data || this.el.stage.offsetParent === null) return; // hidden view
    if (this.data.kind !== '1d') this.sizeColumns(this.v.done().length + 1);
    for (const { plot } of Object.values(this.panels)) Plotly.Plots.resize(plot);
    Plotly.Plots.resize(this.el.chart);
  }

  draw2d(data, models) {
    const f = fieldScale(data.range);
    const { emax, capped } = errorScale(data.range, data.error_max);
    const [rows, cols] = sliceShape(data);
    const fieldName = this.v.info().field_names[this.v.selection.field];
    this.el.legends.replaceChildren(
      colorLegend(`${fieldName}: same scale for truth and all predictions`, f.kind, f.zmin, f.zmax, fmt),
      colorLegend(capped ? 'Absolute error |prediction − truth|: errors larger than the field itself show in the darkest color'
        : 'Absolute error |prediction − truth|: same scale at every time step', 'error', 0, emax, fmt, capped));
    const heat = (z, zmin, zmax, kind) => [{ type: 'heatmap', z, zmin, zmax, colorscale: scale(kind), showscale: false, hoverinfo: 'none' }];

    Plotly.react(this.panels.truth.plot, heat(data.truth, f.zmin, f.zmax, f.kind), heatLayout(rows, cols), plotConfig);
    const start = this.bounds()[0];
    for (const m of models) {
      const p = this.panels[`p:${m}`], e = this.panels[`e:${m}`];
      const d = data.models[m];
      const k = this.v.pair(m).metrics.ar_order;
      const missing = this.mode === 'rollout' && this.t === start
        ? 'Starting frame: the model is given the true field here'
        : `No prediction at t = ${this.t}: this model needs ${k} earlier frame${k > 1 ? 's' : ''}`;
      for (const [panel, z, zmin, zmax, kind] of [[p, d?.pred, f.zmin, f.zmax, f.kind], [e, d?.error, 0, emax, 'error']]) {
        panel.na.hidden = Boolean(d);
        panel.na.textContent = d ? '' : missing;
        Plotly.react(panel.plot, d ? heat(z, zmin, zmax, kind) : [], heatLayout(rows, cols), plotConfig);
      }
      p.small.textContent = d ? `NRMSE ${fmt(frameNrmse(d.error, data.truth))}` : '';
      e.small.textContent = d ? `max ${fmt(maxOf(d.error))}` : '';
    }
    if (this.hoverAt) this.hover(...this.hoverAt);
  }

  /** Readout of every panel's value at one grid cell, and a box around that cell in each panel. */
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
    for (const m of this.v.done()) {
      const md = d.models[m];
      if (md) parts.push(`${this.v.name(m)} ${fmt(md.pred[y]?.[x])} (error ${fmt(md.error[y]?.[x])})`);
    }
    this.el.readout.textContent = parts.join('; ');
    const box = { type: 'rect', x0: x - 0.5, x1: x + 0.5, y0: y - 0.5, y1: y + 0.5, line: { color: css('--ink'), width: 1.5 } };
    cancelAnimationFrame(this.raf);
    this.raf = requestAnimationFrame(() => {
      for (const { plot } of Object.values(this.panels)) Plotly.relayout(plot, { shapes: [box] });
    });
  }

  build1d() {
    this.el.stage.classList.add('wide');
    this.el.stage.replaceChildren();
    this.panels = {};
    for (const [id, title] of [['field', 'Field'], ['error', 'Absolute error |prediction − truth|']]) {
      const panel = document.createElement('div');
      panel.className = 'panel';
      panel.innerHTML = '<h3></h3><div class="plot"></div>';
      panel.querySelector('h3').textContent = title;
      this.el.stage.append(panel);
      this.panels[id] = { panel, plot: panel.querySelector('.plot') };
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
    const errors = [];
    for (const m of models) {
      const d = data.models[m];
      if (!d) continue;
      const line = { color: this.v.color(m), width: 2 };
      const hovertemplate = `${this.v.name(m)} %{y:.4g}<extra></extra>`;
      field.push({ x, y: d.pred, name: this.v.name(m), mode: 'lines', line, hovertemplate });
      errors.push({ x, y: d.error, name: this.v.name(m), mode: 'lines', line, showlegend: false, hovertemplate });
    }
    const base = baseLayout();
    const fieldName = this.v.info().field_names[this.v.selection.field];
    Plotly.react(this.panels.field.plot, field, baseLayout({
      hovermode: 'x unified', yaxis: { ...base.yaxis, range: [lo - pad, hi + pad], title: { text: fieldName } },
      xaxis: { ...base.xaxis, title: { text: 'Grid point' } } }), plotConfig);
    Plotly.react(this.panels.error.plot, errors, baseLayout({
      hovermode: 'x unified', showlegend: false,
      yaxis: { ...base.yaxis, range: [0, errorScale(data.range, data.error_max).emax * 1.05], title: { text: 'Absolute error' } },
      xaxis: { ...base.xaxis, title: { text: 'Grid point' } } }), plotConfig);
    const missing = models.filter((m) => !data.models[m]).map((m) => this.v.name(m));
    this.el.readout.textContent = missing.length ? `No prediction at this step for: ${missing.join(', ')}` : '';
  }

  /** Each model's error at every step, with a line at the current step. Click to jump there. */
  renderTimeline() {
    const [lo, hi] = this.bounds();
    const traces = [];
    for (const m of this.v.done()) {
      const metrics = this.v.pair(m).metrics;
      const curve = (this.mode === 'rollout' ? metrics.rollout : metrics.onestep)?.per_step_nrmse;
      if (!curve) continue;
      const first = this.mode === 'rollout' ? metrics.rollout_start + 1 : metrics.onestep_from;
      traces.push({ x: curve.map((_, i) => first + i), y: curve, mode: 'lines', name: this.v.name(m),
        line: { color: this.v.color(m), width: 2 }, hovertemplate: `${this.v.name(m)} %{y:.3g}<extra></extra>` });
    }
    const ys = traces.flatMap((tr) => tr.y).filter((y) => y > 0);
    const base = baseLayout();
    const layout = baseLayout({
      margin: { l: 44, r: 16, t: 6, b: 20 }, showlegend: false, hovermode: 'x unified',
      xaxis: { ...base.xaxis, range: [lo, hi], dtick: hi - lo > 30 ? undefined : 1, automargin: false, showgrid: false },
      yaxis: { ...base.yaxis, automargin: false, type: 'log', ...(ys.length ? logTicks(Math.min(...ys), Math.max(...ys)) : {}),
        title: { text: 'NRMSE (log)', font: { size: 10.5 } } },
      shapes: [{ type: 'line', x0: this.t, x1: this.t, yref: 'paper', y0: 0, y1: 1, line: { color: css('--ink'), width: 1.5 } }],
    });
    if (this.chartReady) {
      Plotly.react(this.el.chart, traces, layout, plotConfig);
      return;
    }
    Plotly.newPlot(this.el.chart, traces, layout, plotConfig);
    this.el.chart.on('plotly_click', (ev) => {
      const [a, b] = this.bounds();
      this.t = Math.max(a, Math.min(b, Math.round(ev.points[0].x)));
      this.update();
    });
    this.chartReady = true;
  }
}

function heatLayout(rows, cols) {
  return {
    margin: { l: 0, r: 0, t: 0, b: 0 }, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    xaxis: { visible: false, range: [-0.5, cols - 0.5], fixedrange: true },
    yaxis: { visible: false, range: [-0.5, rows - 0.5], fixedrange: true },
    shapes: [],
  };
}

function frameNrmse(err, truth) {
  let se = 0, st = 0;
  err.forEach((row, i) => row.forEach((e, j) => {
    const t = truth[i][j];
    if (e != null && t != null) { se += e * e; st += t * t; }
  }));
  return Math.sqrt(se / (st || 1e-12));
}

function maxOf(a) {
  let m = 0;
  for (const row of a) for (const v of row) if (v != null && v > m) m = v;
  return m;
}
