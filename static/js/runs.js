// Starting runs, the Results menu, and showing one run while it updates.
import { api } from './api.js';
import { loadCatalog, selectedIds, updateRunButton } from './catalog.js';
import { renderMetrics } from './metrics.js';
import { state } from './state.js';
import { $, esc, formatDate, saved, toast } from './util.js';
import { Viewer } from './viewer.js';

const RUNNING = ['queued', 'running'];
const STATUS_TEXT = {
  queued: 'Waiting to start', running: 'Running', done: 'Finished',
  done_with_issues: 'Finished, some combinations did not run', failed: 'Failed',
  cancelled: 'Stopped', interrupted: 'Interrupted (the server restarted)',
};
const ICONS = {
  done: '<svg class="icon ico-done" viewBox="0 0 16 16"><path d="M3 8.5l3 3 7-7" fill="none" stroke="currentColor" stroke-width="2"/></svg>',
  failed: '<svg class="icon ico-failed" viewBox="0 0 16 16"><path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" stroke-width="2"/></svg>',
  skipped: '<svg class="icon ico-skipped" viewBox="0 0 16 16"><path d="M3 8h10" stroke="currentColor" stroke-width="2"/></svg>',
  running: '<svg class="icon ico-running" viewBox="0 0 16 16"><path d="M8 2a6 6 0 1 1-6 6" fill="none" stroke="currentColor" stroke-width="2"/></svg>',
  queued: '<svg class="icon" viewBox="0 0 16 16"><circle cx="8" cy="8" r="5" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>',
};

let viewer;
let pollTimer;

export function initRuns() {
  viewer = new Viewer();
  $('run-btn').addEventListener('click', startRun);
  $('run-select').addEventListener('change', (e) => e.target.value && openRun(e.target.value));
  $('stop-btn').addEventListener('click', stopRun);
  $('delete-run-btn').addEventListener('click', deleteRun);
}

/** Show the last run viewed in this browser, or the newest one. */
export async function openLatestRun() {
  const runs = await refreshRunMenu();
  const last = runs.find((r) => r.id === saved.get('last-run')) || runs[0];
  if (last) await openRun(last.id);
}

/** Redraw the current run (e.g. after a chart option or the color scheme changes). */
export function redraw({ rebuildViewer = false } = {}) {
  if (!state.run) return;
  if (rebuildViewer) viewer.reset();
  render();
}

// ---------------------------------------------------------------- actions
function readSettings() {
  return {
    max_trajectories: +$('s-traj').value || 8,
    rollout_steps: +$('s-steps').value || 0,
    rollout_start: +$('s-start').value || 0,
    viz_samples: +$('s-viz').value || 3,
    device: $('s-device').value,
  };
}

async function startRun() {
  $('run-btn').disabled = true;
  const settings = readSettings();
  saved.set('settings', settings);
  try {
    const run = await api.startRun({
      models: selectedIds('models'), datasets: selectedIds('datasets'), settings, name: $('s-name').value.trim() || null,
    });
    $('s-name').value = '';
    await refreshRunMenu(run.id);
    show(run);
  } catch (e) {
    toast(e.message);
  } finally {
    updateRunButton();
  }
}

async function openRun(id) {
  try {
    show(await api.run(id));
  } catch (e) {
    toast(e.message);
  }
}

async function stopRun() {
  try {
    await api.cancelRun(state.run.id);
    toast('Stopping after the current step');
  } catch (e) {
    toast(e.message);
  }
}

async function deleteRun() {
  if (!confirm(`Delete the results of "${state.run.name}"? This can't be undone.`)) return;
  try {
    await api.deleteRun(state.run.id);
    toast('Run deleted');
    state.run = null;
    $('run-view').hidden = true;
    $('empty').hidden = false;
    await openLatestRun();
  } catch (e) {
    toast(e.message);
  }
}

async function refreshRunMenu(selectId) {
  const runs = await api.runs();
  const label = (r) => `${r.name}, ${formatDate(r.created)}${RUNNING.includes(r.status) ? ' (running)' : ''}`;
  $('run-select').replaceChildren(
    new Option(runs.length ? 'Choose a run' : 'No runs yet', ''),
    ...runs.map((r) => new Option(label(r), r.id)));
  if (selectId) $('run-select').value = selectId;
  return runs;
}

// ---------------------------------------------------------------- showing a run
function show(run) {
  state.run = run;
  saved.set('last-run', run.id);
  $('empty').hidden = true;
  $('run-view').hidden = false;
  $('run-select').value = run.id;
  render();
  clearTimeout(pollTimer);
  if (RUNNING.includes(run.status)) pollTimer = setTimeout(poll, 800);
}

/** While a run is going, fetch it every second; results appear as each model finishes. */
async function poll() {
  const id = state.run?.id;
  if (!id) return;
  try {
    const run = await api.run(id);
    if (state.run?.id !== id) return; // the user switched runs meanwhile
    state.run = run;
    render();
    if (RUNNING.includes(run.status)) {
      pollTimer = setTimeout(poll, 1000);
    } else {
      refreshRunMenu(run.id);
      loadCatalog(); // a finished download changes a card's status
    }
  } catch {
    pollTimer = setTimeout(poll, 3000);
  }
}

function render() {
  const run = state.run;
  const busy = RUNNING.includes(run.status);
  const s = run.settings;
  $('run-title').textContent = run.name;
  $('run-meta').textContent = `${STATUS_TEXT[run.status] || run.status}. Started ${formatDate(run.created)}`
    + `${run.device ? ` on ${run.device}` : ''}. Up to ${s.max_trajectories} trajectories per dataset, `
    + `${s.rollout_steps || 'all available'} rollout steps.`;
  $('stop-btn').hidden = !busy;
  $('delete-run-btn').hidden = busy;
  $('csv-btn').href = `/api/runs/${run.id}/metrics.csv`;
  renderProgress(run, busy);
  viewer.setRun(run);
  renderMetrics(run);
  $('log').textContent = run.log.map((line) => `${line.t.slice(11, 19)}  ${line.msg}`).join('\n');
}

/** One tile per model × dataset. Hidden once everything finished fine. */
function renderProgress(run, busy) {
  const allDone = run.pairs.every((p) => p.status === 'done');
  $('progress').innerHTML = !busy && allDone ? '' : run.pairs.map((p) => {
    const title = `${esc(run.model_cards[p.model]?.name || p.model)} on ${esc(run.dataset_cards[p.dataset]?.name || p.dataset)}`;
    const problem = p.status === 'failed' || p.status === 'skipped';
    const stage = problem ? `${p.status === 'skipped' ? 'Skipped' : 'Failed'}: ${p.error}` : p.stage;
    const bar = p.status === 'running' ? `<span class="bar" style="width:${Math.round((p.progress || 0) * 100)}%"></span>` : '';
    return `<li><span class="pair">${ICONS[p.status] || ICONS.queued}${title}</span>`
      + `<span class="pstage${problem ? ' err' : ''}" title="${esc(stage)}">${esc(stage)}</span>${bar}</li>`;
  }).join('');
}
