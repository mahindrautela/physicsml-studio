// Metrics table and comparison charts for a run.
import { baseLayout, css, logTicks, plotConfig, seriesColor } from './colors.js';
import { fmt } from './viewer.js';

const get = (obj, path) => path.split('.').reduce((o, k) => (o == null ? o : o[k]), obj);

const COLS = [
  ['onestep.nrmse', 'NRMSE'], ['onestep.vrmse', 'VRMSE'], ['onestep.rmse', 'RMSE'], ['onestep.mae', 'MAE'],
  ['rollout.nrmse', 'Mean NRMSE'], ['rollout.final', 'At last step'],
];

function value(pair, key) {
  if (key === 'rollout.final') {
    const c = pair.metrics?.rollout?.per_step_nrmse;
    return c ? c[c.length - 1] : null;
  }
  return get(pair.metrics, key);
}

export function renderMetrics(run) {
  renderTable(run);
  renderGrowth(run);
  renderBars(run);
}

function renderTable(run) {
  const table = document.getElementById('metrics-table');
  const color = (m) => seriesColor(run.models.indexOf(m));
  const name = (m) => run.model_cards[m]?.name || m;
  const head = `<thead>
    <tr><th rowspan="2">Model</th><th rowspan="2">Trajectories</th><th colspan="4" style="text-align:center">One-step</th>
        <th colspan="2" style="text-align:center">Rollout</th><th rowspan="2">ms per frame</th><th rowspan="2">Params (M)</th></tr>
    <tr>${COLS.map(([, l]) => `<th>${l}</th>`).join('')}</tr></thead>`;
  const rows = [];
  for (const d of run.datasets) {
    const pairs = run.pairs.filter((p) => p.dataset === d);
    const done = pairs.filter((p) => p.status === 'done');
    const best = Object.fromEntries(COLS.map(([k]) => [k, Math.min(...done.map((p) => value(p, k)).filter((v) => v != null))]));
    const info = run.dataset_info[d];
    const extra = info ? ` <small>${info.spatial_dims}D, ${info.grid.filter((n) => n > 1).join('×')} grid, ${info.time_steps} steps</small>` : '';
    rows.push(`<tr class="group"><td colspan="10">${esc(run.dataset_cards[d]?.name || d)}${extra}</td></tr>`);
    const errs = new Set(pairs.map((p) => p.error));
    if (pairs.length > 1 && pairs.every((p) => p.status === 'failed') && errs.size === 1) {
      rows.push(`<tr><td>All models</td><td class="msg" colspan="9">${esc(`Failed: ${pairs[0].error}`)}</td></tr>`);
      continue;
    }
    for (const p of pairs) {
      const label = `<td><span class="sw" style="background:${color(p.model)}"></span>${esc(name(p.model))}</td>`;
      if (p.status !== 'done') {
        const msg = p.status === 'running' || p.status === 'queued' ? (p.stage || 'Waiting') : `${cap(p.status)}: ${p.error || p.stage || ''}`;
        rows.push(`<tr>${label}<td class="msg" colspan="9">${esc(msg)}</td></tr>`);
        continue;
      }
      const cells = COLS.map(([k]) => {
        const v = value(p, k);
        const isBest = done.length > 1 && v != null && v === best[k];
        return `<td class="${isBest ? 'best' : ''}">${fmt(v)}</td>`;
      }).join('');
      rows.push(`<tr>${label}<td>${p.metrics.trajectories}</td>${cells}<td>${fmt(p.timing?.ms_per_frame)}</td><td>${fmt(p.info?.parameters_m)}</td></tr>`);
    }
  }
  table.innerHTML = head + `<tbody>${rows.join('')}</tbody>`;
}

function renderGrowth(run) {
  const host = document.getElementById('growth-charts');
  const want = run.datasets.filter((d) => run.pairs.some((p) => p.dataset === d && p.status === 'done'));
  const sig = JSON.stringify([run.id, want]);
  if (host.dataset.sig !== sig) {
    host.dataset.sig = sig;
    host.replaceChildren(...want.map(() => document.createElement('div')));
  }
  if (!want.length) {
    host.innerHTML = '<p class="muted">Curves appear when a model finishes.</p>';
    host.dataset.sig = '';
    return;
  }
  want.forEach((d, i) => {
    const div = host.children[i];
    const pairs = run.pairs.filter((p) => p.dataset === d && p.status === 'done');
    const traces = pairs.map((p) => {
      const c = p.metrics.rollout.per_step_nrmse;
      return { x: c.map((_, j) => j + 1), y: c, name: run.model_cards[p.model]?.name || p.model, mode: 'lines+markers',
        line: { color: seriesColor(run.models.indexOf(p.model)), width: 2 }, marker: { size: c.length > 30 ? 0 : 6 },
        hovertemplate: '%{fullData.name}: %{y:.4g}<extra></extra>' };
    });
    const log = document.getElementById('growth-log').checked;
    const ys = traces.flatMap((t) => t.y).filter((v) => v > 0);
    const ymax = Math.max(...ys, 1e-9) * 1.08;
    const ymin = Math.min(...ys, ymax);
    const H = Math.max(...traces.map((t) => t.x.length));
    const title = run.dataset_cards[d]?.name || d;
    const annotations = traces.length <= 4 ? endLabels(traces, log, ymin, ymax, 240 - 54 - 40) : [];
    Plotly.react(div, traces, baseLayout({
      title: { text: title, font: { size: 12.5, color: css('--ink') }, x: 0, xanchor: 'left', y: 0.98 },
      margin: { l: 52, r: traces.length <= 4 ? Math.min(200, 16 + 6.2 * Math.max(...traces.map((t) => t.name.length))) : 16, t: 54, b: 40 },
      hovermode: 'x unified', showlegend: traces.length > 1, legend: { ...baseLayout().legend, y: 1.0 },
      annotations,
      xaxis: { ...baseLayout().xaxis, title: { text: 'Steps ahead' }, range: [0.5, H + 0.5] },
      yaxis: log
        ? { ...baseLayout().yaxis, title: { text: 'NRMSE (log)' }, type: 'log', ...logTicks(ymin, ymax), range: [Math.log10(ymin / 1.3), Math.log10(ymax * 1.1)] }
        : { ...baseLayout().yaxis, title: { text: 'NRMSE' }, range: [0, ymax] },
    }), plotConfig);
  });
}

// Direct labels at the line ends, nudged apart so they never overlap.
function endLabels(traces, log, ymin, ymax, plotPx) {
  const toPx = (v) => log
    ? (Math.log10(Math.max(v, ymin)) - Math.log10(ymin / 1.3)) / (Math.log10(ymax * 1.1) - Math.log10(ymin / 1.3)) * plotPx
    : (v / ymax) * plotPx;
  const items = traces.map((t) => ({ t, y: t.y[t.y.length - 1], px: toPx(t.y[t.y.length - 1]) }))
    .sort((a, b) => a.px - b.px);
  const GAP = 14;
  for (let i = 1; i < items.length; i++) items[i].px = Math.max(items[i].px, items[i - 1].px + GAP);
  return items.map((it) => ({
    x: it.t.x[it.t.x.length - 1], y: log ? Math.log10(Math.max(it.y, ymin)) : it.y, yshift: it.px - toPx(it.y), text: it.t.name, xanchor: 'left', xshift: 8,
    showarrow: false, font: { size: 11, color: css('--ink-2') } }));
}

function renderBars(run) {
  const div = document.getElementById('bar-chart');
  const key = document.getElementById('bar-metric').value;
  const datasets = run.datasets.filter((d) => run.pairs.some((p) => p.dataset === d && p.status === 'done'));
  const traces = run.models.map((m) => {
    const ys = datasets.map((d) => {
      const p = run.pairs.find((q) => q.model === m && q.dataset === d && q.status === 'done');
      return p ? get(p, key.startsWith('timing') ? key : `metrics.${key}`) : null;
    });
    return { type: 'bar', name: run.model_cards[m]?.name || m, x: datasets.map((d) => run.dataset_cards[d]?.name || d), y: ys,
      marker: { color: seriesColor(run.models.indexOf(m)), line: { width: 0 } },
      hovertemplate: '%{fullData.name}: %{y:.4g}<extra></extra>' };
  }).filter((t) => t.y.some((v) => v != null));
  const label = document.querySelector(`#bar-metric option[value="${key}"]`).textContent;
  Plotly.react(div, traces, baseLayout({
    barmode: 'group', bargap: 0.3, bargroupgap: 0.06, barcornerradius: 4, showlegend: traces.length > 1,
    margin: { l: 52, r: 16, t: 30, b: 56 },
    yaxis: { ...baseLayout().yaxis, title: { text: label }, rangemode: 'tozero' },
    xaxis: { ...baseLayout().xaxis, showgrid: false },
  }), plotConfig);
}

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1).replace(/_/g, ' ');
