// Left column: the model and dataset lists, their cards, and the Run button's label.
import { api } from './api.js';
import { MAX_SERIES, seriesColor } from './colors.js';
import { state } from './state.js';
import { $, esc, plural, saved, toast } from './util.js';

const STATE_TEXT = { ready: 'Ready', download: 'Downloads on first run', needs_install: 'Needs a package', error: 'Problem' };

/** Fetch the cards and redraw both lists. */
export async function loadCatalog() {
  state.catalog = await api.catalog();
  for (const kind of ['models', 'datasets']) {
    const exists = new Set(state.catalog[kind].map((c) => c.id));
    for (const id of [...state.selected[kind]]) if (!exists.has(id)) state.selected[kind].delete(id);
  }
  renderList('models');
  renderList('datasets');
  updateRunButton();
  if (state.catalog.problems.length) {
    toast(`Some cards couldn't be read: ${state.catalog.problems.map((p) => `${p.file} (${p.error})`).join('; ')}`);
  }
}

/** Ticked ids in catalog order (that order also fixes each model's chart color). */
export const selectedIds = (kind) => state.catalog[kind].filter((c) => state.selected[kind].has(c.id)).map((c) => c.id);

export function select(kind, id) {
  state.selected[kind].add(id);
  saved.set(`selected-${kind}`, [...state.selected[kind]]);
}

function toggle(kind, id, on) {
  if (on && kind === 'models' && state.selected.models.size >= MAX_SERIES) {
    toast(`Compare up to ${MAX_SERIES} models at a time so each keeps a distinct color.`);
    return false;
  }
  on ? state.selected[kind].add(id) : state.selected[kind].delete(id);
  saved.set(`selected-${kind}`, [...state.selected[kind]]);
  return true;
}

function renderList(kind) {
  const list = $(kind === 'models' ? 'model-list' : 'dataset-list');
  const colorSlot = kind === 'models' ? selectedIds('models') : [];
  list.replaceChildren(...state.catalog[kind].map((card) => listItem(kind, card, colorSlot.indexOf(card.id))));
}

function listItem(kind, card, colorIndex) {
  const key = `${kind}:${card.id}`;
  const open = state.openCards.has(key);
  const li = document.createElement('li');
  li.innerHTML = `
    <div class="card-row">
      <input type="checkbox" id="${kind}-${esc(card.id)}" ${state.selected[kind].has(card.id) ? 'checked' : ''}>
      <label class="card-text" for="${kind}-${esc(card.id)}">
        <span class="card-name">${esc(card.name)}</span>
        <span class="card-sub"><span class="state ${card.status.state}">${STATE_TEXT[card.status.state] || card.status.state}</span>${card.family ? `, ${esc(card.family)}` : ''}</span>
      </label>
      <div class="card-actions">
        ${colorIndex >= 0 ? `<span class="swatch" title="Color in charts" style="background:${seriesColor(colorIndex)}"></span>` : ''}
        <button type="button" class="more" aria-expanded="${open}" aria-label="${open ? 'Hide' : 'Show'} details for ${esc(card.name)}">${open ? 'Hide card' : 'Card'}</button>
      </div>
    </div>`;
  const box = li.querySelector('input');
  box.addEventListener('change', () => {
    if (!toggle(kind, card.id, box.checked)) box.checked = false;
    if (kind === 'models') renderList('models'); // color swatches follow the selection
    updateRunButton();
  });
  li.querySelector('.more').addEventListener('click', () => {
    open ? state.openCards.delete(key) : state.openCards.add(key);
    renderList(kind);
  });
  if (open) li.append(cardDetails(kind, card));
  return li;
}

function cardDetails(kind, card) {
  const rows = [];
  if (card.description) rows.push(['About', esc(card.description)]);
  rows.push(['Status', esc(card.status.message)]);
  rows.push(['Source', card.source_url
    ? `<a href="${esc(card.source_url)}" target="_blank" rel="noopener">${esc(card.source_text)}</a>`
    : esc(card.source_text)]);
  if (kind === 'models') {
    rows.push(['Type', `<code>${esc(card.adapter)}</code>`]);
    const config = Object.entries(card.config || {});
    if (config.length) rows.push(['Settings', config.map(([k, v]) => `<code>${esc(k)}: ${esc(v)}</code>`).join(' ')]);
    if (card.status.state === 'needs_install') rows.push(['Install', `<code>${esc(card.install_hint || card.status.message)}</code>`]);
  } else {
    const l = card.layout;
    rows.push(['Layout', `<code>${l.axes ? `axes ${esc(l.axes)}` : `F${l.fields} C${l.components} D${l.depth} H${l.height} W${l.width}`}</code>`]);
    if (card.field_names?.length) rows.push(['Fields', esc(card.field_names.join(', '))]);
    rows.push(['Evaluates', describeSplit(card.eval)]);
  }
  if (card.notes) rows.push(['Notes', esc(card.notes)]);

  const div = document.createElement('div');
  div.className = 'card-detail';
  div.innerHTML = `<dl>${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('')}</dl>`
    + (card.links?.length ? `<div class="links">${card.links.map((l) => `<a href="${esc(l.url)}" target="_blank" rel="noopener">${esc(l.label)}</a>`).join('')}</div>` : '')
    + `<button type="button" class="ghost small danger">Remove this ${kind === 'models' ? 'model' : 'dataset'} from the list</button>`;
  div.querySelector('.danger').addEventListener('click', () => removeCard(kind, card));
  return div;
}

function describeSplit(split) {
  switch (split.mode) {
    case 'last_fraction': return `last ${Math.round(split.fraction * 100)}% of trajectories`;
    case 'first_n': return `first ${split.n} trajectories`;
    case 'all': return 'all trajectories';
    default: return `trajectories ${split.indices.join(', ')}`;
  }
}

async function removeCard(kind, card) {
  if (!confirm(`Remove "${card.name}"? This deletes its card file. Downloaded files and past results stay.`)) return;
  try {
    await api.deleteCard(kind, card.id);
    state.selected[kind].delete(card.id);
    toast(`Removed ${card.name}`);
    await loadCatalog();
  } catch (e) {
    toast(e.message);
  }
}

/** Label the Run button and explain what's missing or what will download. */
export function updateRunButton() {
  const m = state.selected.models.size;
  const d = state.selected.datasets.size;
  const button = $('run-btn');
  button.disabled = !(m && d);
  button.textContent = m && d ? `Run ${plural(m, 'model')} on ${plural(d, 'dataset')}` : 'Choose models and datasets';

  const isSelected = (c) => state.selected.models.has(c.id) || state.selected.datasets.has(c.id);
  const cards = [...state.catalog.models, ...state.catalog.datasets].filter(isSelected);
  const notes = [];
  if (!m) notes.push('Pick at least one model.');
  if (!d) notes.push('Pick at least one dataset.');
  const needInstall = cards.filter((c) => c.status.state === 'needs_install');
  if (needInstall.length) notes.push(`${needInstall.map((c) => c.name).join(', ')} need a package installed first (open its card).`);
  const downloads = cards.filter((c) => c.status.state === 'download');
  if (downloads.length) notes.push(`Downloads on first run: ${downloads.map((c) => c.name).join(', ')}.`);
  $('run-hint').textContent = notes.join(' ');
}
