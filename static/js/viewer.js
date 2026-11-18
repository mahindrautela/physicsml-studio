// The "Predicted fields" section: the shared controls (dataset, model, trajectory,
// field, slice) and the switch between its two views:
//   rollout-view.js  All rollout steps: truth and one model's prediction at every step
//   step-view.js     One step at a time: every model at one time step, with a timeline
import { seriesColor } from './colors.js';
import { RolloutView } from './rollout-view.js';
import { StepView } from './step-view.js';
import { $ } from './util.js';

export class Viewer {
  constructor() {
    this.el = {
      dataset: $('v-dataset'), model: $('v-model'), modelWrap: $('v-model-wrap'), sample: $('v-sample'),
      field: $('v-field'), comp: $('v-comp'), compWrap: $('v-comp-wrap'), axis: $('v-axis'), axisWrap: $('v-axis-wrap'),
      index: $('v-index'), indexWrap: $('v-index-wrap'), modeSwitch: $('mode-switch'),
      rolloutBox: $('rollout-view'), stepBox: $('step-view'),
      viewButtons: [...document.querySelectorAll('#view-switch button')],
    };
    this.run = null;
    this.selection = { view: 'rollout', dataset: null, model: null, sample: 0, field: 0, comp: 0, axis: 'd', index: null };
    this.rolloutView = new RolloutView(this);
    this.stepView = new StepView(this);
    this.bind();
  }

  // ---------------------------------------------------------------- run helpers (used by both views)
  pair(model, dataset = this.selection.dataset) { return this.run.pairs.find((p) => p.model === model && p.dataset === dataset); }
  /** Models that finished on a dataset, in the run's order. */
  done(dataset = this.selection.dataset) { return this.run.models.filter((m) => this.pair(m, dataset)?.status === 'done'); }
  color(model) { return seriesColor(this.run.models.indexOf(model)); }
  name(model) { return this.run.model_cards[model]?.name || model; }
  info() { return this.run.dataset_info[this.selection.dataset]; }
  get activeView() { return this.selection.view === 'rollout' ? this.rolloutView : this.stepView; }

  // ---------------------------------------------------------------- controls
  bind() {
    const e = this.el;
    const onChange = (el, key, parse = Number, after = () => {}) => el.addEventListener('change', () => {
      this.selection[key] = parse(el.value);
      after();
      this.syncControls();
      this.activeView.update();
    });
    onChange(e.dataset, 'dataset', String, () => {
      Object.assign(this.selection, { model: null, sample: 0, field: 0, comp: 0, index: null });
      this.stepView.t = null;
    });
    onChange(e.model, 'model', String);
    onChange(e.sample, 'sample');
    onChange(e.field, 'field');
    onChange(e.comp, 'comp');
    onChange(e.axis, 'axis', String, () => { this.selection.index = null; });
    onChange(e.index, 'index');
    for (const button of e.viewButtons) {
      button.addEventListener('click', () => this.showView(button.dataset.view));
    }
  }

  showView(view) {
    this.selection.view = view;
    this.syncControls();
    this.activeView.update();
  }

  /** Open "One step at a time" at time step t of the rollout (clicked in the rollout grid). */
  showStep(t) {
    this.stepView.goTo('rollout', t);
    this.showView('step');
  }

  /** Fill the menus from the run and show only what the active view uses. */
  syncControls() {
    const s = this.selection;
    const rollout = s.view === 'rollout';
    for (const b of this.el.viewButtons) b.setAttribute('aria-checked', String(b.dataset.view === s.view));
    this.el.rolloutBox.hidden = !rollout;
    this.el.stepBox.hidden = rollout;
    this.el.modelWrap.hidden = !rollout;
    this.el.modeSwitch.hidden = rollout;
    if (!s.dataset) return;

    const info = this.info();
    const models = this.done();
    if (!models.includes(s.model)) s.model = models[0];
    fillSelect(this.el.model, models.map((m) => [m, this.name(m)]));
    const kept = Math.min(this.run.settings.viz_samples, info.eval_indices.length);
    fillSelect(this.el.sample, Array.from({ length: kept }, (_, i) => [i, `#${info.eval_indices[i]}`]));
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
    this.el.model.value = s.model;
    this.el.sample.value = s.sample;
    this.el.field.value = s.field;
    this.el.comp.value = s.comp;
  }

  // ---------------------------------------------------------------- run updates
  /** Forget everything drawn so the next setRun() redraws from scratch (e.g. new color scheme). */
  reset() {
    this.rolloutView.reset();
    this.stepView.reset();
    this.doneKey = null;
  }

  /** Called whenever the run record changes (every second while it runs). */
  setRun(run) {
    if (!this.run || this.run.id !== run.id) {
      this.reset();
      Object.assign(this.selection, { dataset: null, model: null, sample: 0, field: 0, comp: 0, index: null });
    }
    this.run = run;
    const ready = run.datasets.filter((d) => run.dataset_info[d] && this.done(d).length);
    fillSelect(this.el.dataset, ready.map((d) => [d, run.dataset_cards[d]?.name || d]));
    if (!ready.length) {
      this.selection.dataset = null;
      this.syncControls();
      const busy = ['queued', 'running'].includes(run.status);
      this.activeView.message(busy ? 'Fields appear here as soon as the first model finishes on a dataset.'
        : 'No model finished on any dataset in this run. Check the messages above or the run log.');
      return;
    }
    if (!ready.includes(this.selection.dataset)) {
      Object.assign(this.selection, { dataset: ready[0], model: null, sample: 0, field: 0, comp: 0, index: null });
    }
    this.el.dataset.value = this.selection.dataset;
    // Redraw only when another model finished; polling alone doesn't change anything shown.
    const doneKey = JSON.stringify(ready.map((d) => this.done(d)));
    if (doneKey !== this.doneKey) {
      this.doneKey = doneKey;
      this.rolloutView.reset();
      this.stepView.forgetData();
      this.syncControls();
      this.activeView.update();
    }
  }
}

/** Replace a select's options only when they changed, keeping the chosen value if it still exists. */
function fillSelect(select, options) {
  const signature = JSON.stringify(options);
  if (select.dataset.sig === signature) return;
  select.dataset.sig = signature;
  const value = select.value;
  select.replaceChildren(...options.map(([v, label]) => new Option(label, v)));
  if (options.some(([v]) => String(v) === value)) select.value = value;
}
