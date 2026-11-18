// "Add model" and "Add dataset" dialogs. Each saves a card (a YAML file in catalog/) on the server.
import { api } from './api.js';
import { state } from './state.js';
import { $, toast } from './util.js';

const MORPH_INSTALL = 'pip install "git+https://github.com/lanl/MORPH.git@package-morph"';

// Pre-filled settings for each model type (see the docstring at the top of app/adapters/<type>.py).
const DEFAULT_SETTINGS = {
  morph: { size: 'Ti', max_ar: 1, ar_order: 1, normalization: 'trajectory', lora_r_attn: 0, lora_r_mlp: 0, batch_size: 8 },
  pytorch: { ar_order: 1, normalization: 'none', batch_size: 8 },
};
const TYPE_LABELS = {
  morph: 'MORPH checkpoint (.pth)',
  pytorch: 'PyTorch file: torch.export (.pt2) or TorchScript (.pt)',
};
// What "Trajectories to evaluate" options mean on a dataset card.
const SPLITS = {
  last_fraction: { mode: 'last_fraction', fraction: 0.1 },
  all: { mode: 'all' },
  first_n: { mode: 'first_n', n: 10 },
};

/** onSaved(kind, id) runs after a card is saved; kind is 'models' or 'datasets'. */
export function initForms({ onSaved }) {
  initModelForm(onSaved);
  initDatasetForm(onSaved);
}

/** Card ids: lowercase words joined by '-', made from the name. */
export const slug = (name) => name.toLowerCase().normalize('NFKD').replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 60) || 'item';

const listOf = (text) => text.split(',').map((s) => s.trim()).filter(Boolean);

/**
 * A "link or path" input. The server turns the text into a card source
 * (Hugging Face, GitHub, URL or local file) and we show how it was read.
 */
class LinkField {
  constructor(input, note) {
    this.input = input;
    this.note = note;
    this.reset();
    input.addEventListener('change', () => this.get());
  }

  /** The source for the current text. Each distinct link is sent to the server once. */
  get() {
    const link = this.input.value.trim();
    if (link !== this.link) {
      this.link = link;
      this.reading = link ? this.read(link) : Promise.resolve(null);
    }
    return this.reading;
  }

  async read(link) {
    this.note.textContent = '';
    try {
      const source = await api.parseLink(link);
      if (link === this.link) this.note.textContent = `Reads as: ${describeSource(source)}`;
      return source;
    } catch (e) {
      if (link === this.link) this.note.textContent = `Can't read that link: ${e.message}`;
      return null;
    }
  }

  reset() {
    this.link = null;
    this.reading = Promise.resolve(null);
    this.note.textContent = '';
  }
}

function describeSource(src) {
  switch (src.type) {
    case 'huggingface':
      return `Hugging Face ${src.repo_type} repo ${src.repo_id}, file ${src.subfolder ? `${src.subfolder}/` : ''}${src.filename}`
        + (src.revision ? ` (revision ${src.revision})` : '');
    case 'github': return `GitHub repo ${src.repo_id}, branch ${src.revision || 'default'}, file ${src.filename}`;
    case 'url': return `Web address ${src.url}`;
    case 'local': return `File on the server: ${src.path}`;
    default: return '';
  }
}

// ---------------------------------------------------------------- Add model
function initModelForm(onSaved) {
  const dialog = $('dlg-model');
  const form = $('form-model');
  const typeSelect = $('dm-adapter');
  const error = $('dm-error');
  const weights = new LinkField(form.link, $('dm-parsed'));
  const modelTypes = () => state.catalog.adapters;

  function showTypeDefaults() {
    const type = typeSelect.value;
    $('dm-adapter-help').textContent = modelTypes()[type]?.doc || '';
    $('dm-source').hidden = modelTypes()[type]?.needs_weights === false;
    form.config.value = JSON.stringify(DEFAULT_SETTINGS[type] || {}, null, 2);
    form.requires.value = type === 'morph' ? 'morph_pde' : '';
  }
  typeSelect.addEventListener('change', showTypeDefaults);

  $('add-model').addEventListener('click', () => {
    form.reset();
    weights.reset();
    error.textContent = '';
    const types = Object.keys(modelTypes());
    typeSelect.replaceChildren(...types.map((t) => new Option(TYPE_LABELS[t] || t, t)));
    typeSelect.value = types.includes('morph') ? 'morph' : types[0];
    showTypeDefaults();
    dialog.showModal();
  });

  form.addEventListener('submit', async (event) => {
    if (event.submitter?.value !== 'save') return; // Cancel closes the dialog normally
    event.preventDefault();
    error.textContent = '';
    let config;
    try {
      config = JSON.parse(form.config.value || '{}');
    } catch {
      error.textContent = 'Settings must be valid JSON, e.g. {"size": "Ti"}.';
      return;
    }
    const type = typeSelect.value;
    let source = { type: 'none' };
    if (modelTypes()[type]?.needs_weights !== false) {
      source = await weights.get();
      if (!source) { error.textContent = 'Add a link or path to the weights file.'; return; }
    }
    const name = form.card_name.value.trim();
    const card = {
      id: slug(name), name, adapter: type, source, config,
      description: form.description.value.trim(),
      requires: listOf(form.requires.value),
      ...(type === 'morph' ? { install_hint: MORPH_INSTALL } : {}),
    };
    try {
      await api.addCard('models', card);
      dialog.close();
      toast(`Saved model ${name}`);
      onSaved('models', card.id);
    } catch (e) {
      error.textContent = e.message;
    }
  });
}

// ---------------------------------------------------------------- Add dataset
function initDatasetForm(onSaved) {
  const dialog = $('dlg-dataset');
  const form = $('form-dataset');
  const error = $('dd-error');
  const arraySelect = $('dd-var');
  const arrayRow = $('dd-var-wrap');
  const note = $('dd-parsed');
  const file = new LinkField(form.link, note);
  let arrays = [];

  form.link.addEventListener('change', () => { arrayRow.hidden = true; });

  function showArray() {
    const a = arrays.find((x) => x.name === arraySelect.value);
    if (!a) return;
    const axes = suggestAxes(a.shape);
    if (axes && !form.axes.value) form.axes.value = axes;
    note.textContent = `Array "${a.name}" has shape (${a.shape.join(', ')}), ${a.dtype}.`
      + (axes ? ` Suggested axis letters: ${axes}. Check that they match your data.` : '');
  }
  arraySelect.addEventListener('change', showArray);

  $('dd-inspect').addEventListener('click', async (event) => {
    const button = event.currentTarget;
    error.textContent = '';
    const source = await file.get();
    if (!source) { error.textContent = 'Add a link or path first.'; return; }
    button.disabled = true;
    button.textContent = 'Looking inside…';
    try {
      arrays = (await api.inspect(source)).variables;
      arraySelect.replaceChildren(...arrays.map((a) => new Option(`${a.name}  (${a.shape.join('×')})`, a.name)));
      arrayRow.hidden = false;
      form.axes.value = '';
      showArray();
    } catch (e) {
      error.textContent = e.message;
    } finally {
      button.disabled = false;
      button.textContent = 'Look inside the file';
    }
  });

  $('add-dataset').addEventListener('click', () => {
    form.reset();
    file.reset();
    error.textContent = '';
    arrayRow.hidden = true;
    arrays = [];
    dialog.showModal();
  });

  form.addEventListener('submit', async (event) => {
    if (event.submitter?.value !== 'save') return;
    event.preventDefault();
    error.textContent = '';
    const source = await file.get();
    if (!source) { error.textContent = 'Add a link or path to the data file.'; return; }
    const axes = form.axes.value.trim();
    const layout = axes ? { axes } : {
      fields: +form.fields.value, components: +form.components.value,
      depth: +form.depth.value, height: +form.height.value, width: +form.width.value,
    };
    const name = form.card_name.value.trim();
    const card = {
      id: slug(name), name, description: form.description.value.trim(), source, layout,
      variable: arrayRow.hidden ? null : arraySelect.value || null,
      field_names: listOf(form.field_names.value),
      eval: SPLITS[form.eval.value],
    };
    try {
      await api.addCard('datasets', card);
      dialog.close();
      toast(`Saved dataset ${name}`);
      onSaved('datasets', card.id);
    } catch (e) {
      error.textContent = e.message;
    }
  });
}

/** Guess the axis letters from an array's shape. Small sizes (≤ 3) are usually fields. */
function suggestAxes(shape) {
  const [n, last] = [shape.length, shape[shape.length - 1]];
  if (n === 3) return 'NTW';
  if (n === 4) return shape[2] <= 3 && shape[3] > 8 ? 'NTFW' : 'NTHW';
  if (n === 5) return last <= 3 ? 'NTHWF' : shape[2] <= 3 ? 'NTFHW' : 'NTDHW';
  if (n === 6) return last <= 3 ? 'NTDHWF' : 'NTFDHW';
  return '';
}
