// Page wiring: catalog lists, run setup, run history, polling for progress.
import { api } from './api.js';
import { MAX_SERIES, onSchemeChange, seriesColor } from './colors.js';
import { initForms } from './forms.js';
import { renderMetrics } from './metrics.js';
import { Viewer } from './viewer.js';

const $ = (id) => document.getElementById(id);
const S = { catalog: null, system: null, models: new Set(), datasets: new Set(), run: null, pollTimer: null, open: new Set() };
let viewer;

const store = {
  get(k) { try { return JSON.parse(localStorage.getItem(k)); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* storage unavailable */ } },
};

function toast(msg) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(toast.h);
  toast.h = setTimeout(() => t.classList.remove('show'), 3200);
}

const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// ---------------------------------------------------------------- catalog
async function loadCatalog() {
  S.catalog = await api.catalog();
  const ids = (k) => new Set(S.catalog[k].map((c) => c.id));
  for (const k of ['models', 'datasets']) for (const id of [...S[k]]) if (!ids(k).has(id)) S[k].delete(id);
  renderList('models');
  renderList('datasets');
  if (S.catalog.problems.length) {
    toast(`Some cards couldn't be read: ${S.catalog.problems.map((p) => `${p.file} (${p.error})`).join('; ')}`);
  }
  updateRunButton();
}

const STATE_TEXT = { ready: 'Ready', download: 'Downloads on first run', needs_install: 'Needs a package', error: 'Problem' };

function renderList(kind) {
  const ul = $(kind === 'models' ? 'model-list' : 'dataset-list');
  const cards = S.catalog[kind];
  const selectedOrder = kind === 'models' ? cards.filter((c) => S.models.has(c.id)).map((c) => c.id) : [];
  ul.replaceChildren(...cards.map((c) => {
    const li = document.createElement('li');
    const row = document.createElement('div');
    row.className = 'card-row';
    const id = `${kind}-${c.id}`;
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.id = id;
    cb.checked = S[kind].has(c.id);
    cb.addEventListener('change', () => {
      if (cb.checked && kind === 'models' && S.models.size >= MAX_SERIES) {
        cb.checked = false;
        toast(`Compare up to ${MAX_SERIES} models at a time so each keeps a distinct color.`);
        return;
      }
      cb.checked ? S[kind].add(c.id) : S[kind].delete(c.id);
      store.set(`sel-${kind}`, [...S[kind]]);
      if (kind === 'models') renderList('models');
      updateRunButton();
    });
    const text = document.createElement('label');
    text.htmlFor = id;
    text.style.cssText = 'display:block;font-weight:400;font-size:14px';
    const st = c.status.state;
    text.innerHTML = `<span class="card-name">${esc(c.name)}</span>
      <span class="card-sub"><span class="state ${st}">${STATE_TEXT[st] || st}</span>${c.family ? `, ${esc(c.family)}` : ''}</span>`;
    const right = document.createElement('div');
    right.style.cssText = 'display:flex;gap:6px;align-items:flex-start';
    const idx = selectedOrder.indexOf(c.id);
    if (idx >= 0) {
      const sw = document.createElement('span');
      sw.className = 'swatch';
      sw.style.background = seriesColor(idx);
      sw.title = 'Color in charts';
      right.append(sw);
    }
    const more = document.createElement('button');
    more.className = 'more';
    more.type = 'button';
    const key = `${kind}:${c.id}`;
    more.setAttribute('aria-expanded', String(S.open.has(key)));
    more.textContent = S.open.has(key) ? 'Hide card' : 'Card';
    more.setAttribute('aria-label', `${S.open.has(key) ? 'Hide' : 'Show'} details for ${c.name}`);
    more.addEventListener('click', () => {
      S.open.has(key) ? S.open.delete(key) : S.open.add(key);
      renderList(kind);
    });
    right.append(more);
    row.append(cb, text, right);
    li.append(row);
    if (S.open.has(key)) li.append(cardDetail(kind, c));
    return li;
  }));
}

function cardDetail(kind, c) {
  const div = document.createElement('div');
  div.className = 'card-detail';
  const rows = [];
  if (c.description) rows.push(['About', esc(c.description)]);
  rows.push(['Status', esc(c.status.message)]);
  rows.push(['Source', c.source_url ? `<a href="${esc(c.source_url)}" target="_blank" rel="noopener">${esc(c.source_text)}</a>` : esc(c.source_text)]);
  if (kind === 'models') {
    rows.push(['Type', `<code>${esc(c.adapter)}</code>`]);
    if (c.config && Object.keys(c.config).length) rows.push(['Settings', Object.entries(c.config).map(([k, v]) => `<code>${esc(k)}: ${esc(v)}</code>`).join(' ')]);
    if (c.status.state === 'needs_install') rows.push(['Install', `<code>${esc(c.install_hint || c.status.message)}</code>`]);
  } else {
    const l = c.layout;
    rows.push(['Layout', l.axes ? `<code>axes ${esc(l.axes)}</code>` : `<code>F${l.fields} C${l.components} D${l.depth} H${l.height} W${l.width}</code>`]);
    if (c.field_names?.length) rows.push(['Fields', esc(c.field_names.join(', '))]);
    const ev = c.eval;
    rows.push(['Evaluates', ev.mode === 'last_fraction' ? `last ${Math.round(ev.fraction * 100)}% of trajectories` : ev.mode === 'first_n' ? `first ${ev.n} trajectories` : ev.mode === 'all' ? 'all trajectories' : `trajectories ${ev.indices.join(', ')}`]);
  }
  if (c.notes) rows.push(['Notes', esc(c.notes)]);
  div.innerHTML = `<dl>${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('')}</dl>` +
    (c.links?.length ? `<div class="links">${c.links.map((l) => `<a href="${esc(l.url)}" target="_blank" rel="noopener">${esc(l.label)}</a>`).join('')}</div>` : '');
  const del = document.createElement('button');
  del.className = 'ghost small danger';
  del.type = 'button';
  del.textContent = `Remove this ${kind === 'models' ? 'model' : 'dataset'} from the list`;
  del.addEventListener('click', async () => {
    if (!confirm(`Remove "${c.name}"? This deletes its card file. Downloaded files and past results stay.`)) return;
    try {
      await api.deleteCard(kind, c.id);
      S[kind].delete(c.id);
      toast(`Removed ${c.name}`);
      loadCatalog();
    } catch (e) { toast(e.message); }
  });
  div.append(del);
  return div;
}

function updateRunButton() {
  const m = S.models.size, d = S.datasets.size;
  const btn = $('run-btn');
  btn.disabled = !(m && d);
  btn.textContent = m && d ? `Run ${m} model${m > 1 ? 's' : ''} on ${d} dataset${d > 1 ? 's' : ''}` : 'Choose models and datasets';
  const notes = [];
  if (!m) notes.push('Pick at least one model.');
  if (!d) notes.push('Pick at least one dataset.');
  const need = S.catalog?.models.filter((c) => S.models.has(c.id) && c.status.state === 'needs_install') || [];
  if (need.length) notes.push(`${need.map((c) => c.name).join(', ')} need a package installed first (open its card).`);
  const dl = [...(S.catalog?.models || []), ...(S.catalog?.datasets || [])].filter((c) => (S.models.has(c.id) || S.datasets.has(c.id)) && c.status.state === 'download');
  if (dl.length) notes.push(`Downloads on first run: ${dl.map((c) => c.name).join(', ')}.`);
  $('run-hint').textContent = notes.join(' ');
}

// ---------------------------------------------------------------- runs
async function startRun() {
  const btn = $('run-btn');
  btn.disabled = true;
  const order = (k) => S.catalog[k].filter((c) => S[k].has(c.id)).map((c) => c.id);
  const settings = {
    max_trajectories: +$('s-traj').value || 8, rollout_steps: +$('s-steps').value || 0,
    rollout_start: +$('s-start').value || 0, viz_samples: +$('s-viz').value || 3, device: $('s-device').value,
  };
  store.set('settings', settings);
  try {
    const run = await api.startRun({ models: order('models'), datasets: order('datasets'), settings, name: $('s-name').value.trim() || null });
    $('s-name').value = '';
    await refreshRunList(run.id);
    showRun(run);
  } catch (e) {
    toast(e.message);
  } finally {
    updateRunButton();
  }
}

async function refreshRunList(selectId) {
  const runs = await api.runs();
  const sel = $('run-select');
  const label = (r) => `${r.name}, ${new Date(r.created).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })}${r.status === 'running' || r.status === 'queued' ? ' (running)' : ''}`;
  sel.replaceChildren(new Option(runs.length ? 'Choose a run' : 'No runs yet', ''), ...runs.map((r) => new Option(label(r), r.id)));
  if (selectId) sel.value = selectId;
  return runs;
}

function showRun(run) {
  S.run = run;
  store.set('lastRun', run.id);
  $('empty').hidden = true;
  $('run-view').hidden = false;
  $('run-select').value = run.id;
  renderRun();
  clearTimeout(S.pollTimer);
  if (['queued', 'running'].includes(run.status)) S.pollTimer = setTimeout(poll, 800);
}

async function poll() {
  if (!S.run) return;
  const id = S.run.id;
  try {
    const run = await api.run(id);
    if (!S.run || S.run.id !== id) return;
    S.run = run;
    renderRun();
    if (['queued', 'running'].includes(run.status)) {
      S.pollTimer = setTimeout(poll, 1000);
    } else {
      refreshRunList(run.id);
      loadCatalog(); // downloads may have changed card status
    }
  } catch (e) {
    S.pollTimer = setTimeout(poll, 3000);
  }
}

const RUN_STATUS = {
  queued: 'Waiting to start', running: 'Running', done: 'Finished', done_with_issues: 'Finished, some combinations did not run',
  failed: 'Failed', cancelled: 'Stopped', interrupted: 'Interrupted (the server restarted)',
};
const ICONS = {
  done: '<svg class="icon ico-done" viewBox="0 0 16 16"><path d="M3 8.5l3 3 7-7" fill="none" stroke="currentColor" stroke-width="2"/></svg>',
  failed: '<svg class="icon ico-failed" viewBox="0 0 16 16"><path d="M4 4l8 8M12 4l-8 8" stroke="currentColor" stroke-width="2"/></svg>',
  skipped: '<svg class="icon ico-skipped" viewBox="0 0 16 16"><path d="M3 8h10" stroke="currentColor" stroke-width="2"/></svg>',
  running: '<svg class="icon ico-running" viewBox="0 0 16 16"><path d="M8 2a6 6 0 1 1-6 6" fill="none" stroke="currentColor" stroke-width="2"/></svg>',
  queued: '<svg class="icon" viewBox="0 0 16 16"><circle cx="8" cy="8" r="5" fill="none" stroke="currentColor" stroke-width="1.5"/></svg>',
};

function renderRun() {
  const run = S.run;
  $('run-title').textContent = run.name;
  const t = new Date(run.created).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' });
  const s = run.settings;
  $('run-meta').textContent = `${RUN_STATUS[run.status] || run.status}. Started ${t}${run.device ? ` on ${run.device}` : ''}. ` +
    `Up to ${s.max_trajectories} trajectories per dataset, ${s.rollout_steps || 'all available'} rollout steps.`;
  const busy = ['queued', 'running'].includes(run.status);
  $('stop-btn').hidden = !busy;
  $('delete-run-btn').hidden = busy;
  $('csv-btn').href = `/api/runs/${run.id}/metrics.csv`;

  const prog = $('progress');
  const show = busy || run.pairs.some((p) => p.status !== 'done');
  prog.innerHTML = !show ? '' : run.pairs.map((p) => {
    const name = `${esc(run.model_cards[p.model]?.name || p.model)} on ${esc(run.dataset_cards[p.dataset]?.name || p.dataset)}`;
    const icon = ICONS[p.status] || ICONS.queued;
    const err = ['failed', 'skipped'].includes(p.status);
    const stage = err ? `${p.status === 'skipped' ? 'Skipped' : 'Failed'}: ${p.error}` : p.stage;
    const bar = p.status === 'running' ? `<span class="bar" style="width:${Math.round((p.progress || 0) * 100)}%"></span>` : '';
    return `<li><span class="pair">${icon}${name}</span><span class="pstage ${err ? "err" : ""}" title="${esc(stage)}">${esc(stage)}</span>${bar}</li>`;
  }).join('');

  viewer.setRun(run);
  renderMetrics(run);
  $('log').textContent = run.log.map((l) => `${l.t.slice(11, 19)}  ${l.msg}`).join('\n');
}

// ---------------------------------------------------------------- boot
async function boot() {
  viewer = new Viewer();
  S.system = await api.system();
  const dev = $('s-device');
  dev.replaceChildren(...S.system.devices.map((d) => new Option(d === 'auto' ? 'Automatic (GPU if available)' : d, d)));
  $('sys-line').textContent = S.system.torch
    ? `PyTorch ${S.system.torch}, ${S.system.gpus.length ? S.system.gpus.join(', ') : 'no GPU found, using CPU'}`
    : 'PyTorch is not installed';
  const saved = store.get('settings');
  if (saved) {
    $('s-traj').value = saved.max_trajectories; $('s-steps').value = saved.rollout_steps;
    $('s-start').value = saved.rollout_start; $('s-viz').value = saved.viz_samples;
    if (S.system.devices.includes(saved.device)) dev.value = saved.device;
  }
  for (const k of ['models', 'datasets']) (store.get(`sel-${k}`) || []).forEach((id) => S[k].add(id));
  await loadCatalog();

  initForms({
    getAdapters: () => S.catalog.adapters,
    toast,
    onSaved: async (kind, id) => { S[kind].add(id); store.set(`sel-${kind}`, [...S[kind]]); await loadCatalog(); },
  });

  $('run-btn').addEventListener('click', startRun);
  $('run-select').addEventListener('change', async (e) => {
    if (!e.target.value) return;
    try { showRun(await api.run(e.target.value)); } catch (err) { toast(err.message); }
  });
  $('stop-btn').addEventListener('click', async () => {
    try { await api.cancelRun(S.run.id); toast('Stopping after the current step'); } catch (e) { toast(e.message); }
  });
  $('delete-run-btn').addEventListener('click', async () => {
    if (!confirm(`Delete the results of "${S.run.name}"? This can't be undone.`)) return;
    try {
      await api.deleteRun(S.run.id);
      toast('Run deleted');
      S.run = null;
      $('run-view').hidden = true;
      $('empty').hidden = false;
      const runs = await refreshRunList();
      if (runs.length) showRun(await api.run(runs[0].id));
    } catch (e) { toast(e.message); }
  });
  $('bar-metric').addEventListener('change', () => S.run && renderMetrics(S.run));
  $('growth-log').addEventListener('change', () => S.run && renderMetrics(S.run));
  onSchemeChange(() => { if (S.run) { viewer.layoutKey = null; viewer.chartReady = false; renderRun(); viewer.update(); } });

  const runs = await refreshRunList();
  const last = store.get('lastRun');
  const pick = runs.find((r) => r.id === last) || runs[0];
  if (pick) showRun(await api.run(pick.id));
}

window.addEventListener('DOMContentLoaded', () => {
  const go = () => boot().catch((e) => toast(`Couldn't reach the server: ${e.message}`));
  if (window.Plotly) go(); else window.addEventListener('load', go);
});
