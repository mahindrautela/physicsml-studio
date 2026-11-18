// Page startup. Each part of the page lives in its own module:
//   catalog.js  model and dataset lists (left column)
//   runs.js     starting runs, the Results menu, progress
//   viewer.js   predicted fields and the timeline
//   metrics.js  metrics table and charts
//   forms.js    "Add model" / "Add dataset" dialogs
import { api } from './api.js';
import { loadCatalog, select } from './catalog.js';
import { onSchemeChange } from './colors.js';
import { initForms } from './forms.js';
import { initRuns, openLatestRun, redraw } from './runs.js';
import { state } from './state.js';
import { $, saved, toast } from './util.js';

async function start() {
  state.system = await api.system();
  showSystem();
  restoreSettings();
  for (const kind of ['models', 'datasets']) (saved.get(`selected-${kind}`) || []).forEach((id) => state.selected[kind].add(id));
  await loadCatalog();

  initRuns();
  initForms({
    onSaved: async (kind, id) => { select(kind, id); await loadCatalog(); },
  });
  $('bar-metric').addEventListener('change', () => redraw());
  $('growth-log').addEventListener('change', () => redraw());
  onSchemeChange(() => redraw({ rebuildViewer: true }));

  await openLatestRun();
}

function showSystem() {
  const { torch, gpus, devices } = state.system;
  $('sys-line').textContent = torch
    ? `PyTorch ${torch}, ${gpus.length ? gpus.join(', ') : 'no GPU found, using CPU'}`
    : 'PyTorch is not installed';
  $('s-device').replaceChildren(...devices.map((d) => new Option(d === 'auto' ? 'Automatic (GPU if available)' : d, d)));
}

function restoreSettings() {
  const s = saved.get('settings');
  if (!s) return;
  $('s-traj').value = s.max_trajectories;
  $('s-steps').value = s.rollout_steps;
  $('s-start').value = s.rollout_start;
  $('s-viz').value = s.viz_samples;
  if (state.system.devices.includes(s.device)) $('s-device').value = s.device;
}

window.addEventListener('DOMContentLoaded', () => {
  start().catch((e) => toast(`Couldn't reach the server: ${e.message}`));
});
