// "Add model" and "Add dataset" dialogs. Both save a card (YAML) on the server.
import { api } from './api.js';

const DEFAULTS = {
  morph: { size: 'Ti', max_ar: 1, ar_order: 1, normalization: 'trajectory', lora_r_attn: 0, lora_r_mlp: 0, batch_size: 8 },
  pytorch: { ar_order: 1, normalization: 'none', batch_size: 8 },
};
const ADAPTER_LABEL = {
  morph: 'MORPH checkpoint (.pth)',
  pytorch: 'PyTorch file: torch.export (.pt2) or TorchScript (.pt)',
  persistence: 'Persistence baseline',
  linear_extrapolation: 'Linear extrapolation baseline',
};

export const slug = (s) => s.toLowerCase().normalize('NFKD').replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 60) || 'item';

function describe(src) {
  if (!src) return '';
  switch (src.type) {
    case 'huggingface': return `Hugging Face ${src.repo_type === 'dataset' ? 'dataset' : 'model'} repo ${src.repo_id}, file ${src.subfolder ? src.subfolder + '/' : ''}${src.filename}${src.revision ? ` (revision ${src.revision})` : ''}`;
    case 'github': return `GitHub repo ${src.repo_id}, branch ${src.revision || 'default'}, file ${src.filename}`;
    case 'url': return `Web address ${src.url}`;
    case 'local': return `File on the server: ${src.path}`;
    default: return '';
  }
}

async function parseInto(input, out, holder) {
  holder.source = null;
  out.textContent = '';
  const link = input.value.trim();
  if (!link) return null;
  try {
    holder.source = await api.parseLink(link);
    out.textContent = `Reads as: ${describe(holder.source)}`;
  } catch (e) {
    out.textContent = `Can't read that link: ${e.message}`;
  }
  return holder.source;
}

export function initForms({ getAdapters, onSaved, toast }) {
  // ---------------- model
  const dm = document.getElementById('dlg-model');
  const fm = document.getElementById('form-model');
  const adapterSel = document.getElementById('dm-adapter');
  const help = document.getElementById('dm-adapter-help');
  const srcBox = document.getElementById('dm-source');
  const mErr = document.getElementById('dm-error');
  const mHolder = {};

  const syncAdapter = () => {
    const a = adapterSel.value;
    const meta = getAdapters()[a] || {};
    help.textContent = meta.doc || '';
    srcBox.hidden = meta.needs_weights === false;
    fm.config.value = JSON.stringify(DEFAULTS[a] || {}, null, 2);
    fm.requires.value = a === 'morph' ? 'morph_pde' : '';
  };
  adapterSel.addEventListener('change', syncAdapter);
  fm.link.addEventListener('change', () => parseInto(fm.link, document.getElementById('dm-parsed'), mHolder));

  document.getElementById('add-model').addEventListener('click', () => {
    fm.reset();
    mErr.textContent = '';
    document.getElementById('dm-parsed').textContent = '';
    const adapters = getAdapters();
    adapterSel.replaceChildren(...Object.keys(adapters).map((a) => new Option(ADAPTER_LABEL[a] || a, a)));
    adapterSel.value = adapters.morph ? 'morph' : Object.keys(adapters)[0];
    syncAdapter();
    dm.showModal();
  });

  fm.addEventListener('submit', async (ev) => {
    if (ev.submitter?.value !== 'save') return;
    ev.preventDefault();
    mErr.textContent = '';
    let config;
    try { config = JSON.parse(fm.config.value || '{}'); } catch { mErr.textContent = 'Settings must be valid JSON, e.g. {"size": "Ti"}.'; return; }
    const adapter = adapterSel.value;
    const meta = getAdapters()[adapter] || {};
    let source = { type: 'none' };
    if (meta.needs_weights !== false) {
      source = mHolder.source || (await parseInto(fm.link, document.getElementById('dm-parsed'), mHolder));
      if (!source) { mErr.textContent = 'Add a link or path to the weights file.'; return; }
    }
    const card = {
      id: slug(fm.card_name.value), name: fm.card_name.value.trim(), adapter, source, config,
      description: fm.description.value.trim(),
      requires: fm.requires.value.split(',').map((s) => s.trim()).filter(Boolean),
    };
    if (adapter === 'morph') card.install_hint = 'pip install "git+https://github.com/lanl/MORPH.git@package-morph"';
    try {
      await api.addCard('models', card);
      dm.close();
      toast(`Saved model ${card.name}`);
      onSaved('models', card.id);
    } catch (e) { mErr.textContent = e.message; }
  });

  // ---------------- dataset
  const dd = document.getElementById('dlg-dataset');
  const fd = document.getElementById('form-dataset');
  const dErr = document.getElementById('dd-error');
  const varSel = document.getElementById('dd-var');
  const varWrap = document.getElementById('dd-var-wrap');
  const dHolder = {};
  let variables = [];

  fd.link.addEventListener('change', () => { varWrap.hidden = true; parseInto(fd.link, document.getElementById('dd-parsed'), dHolder); });

  const suggestAxes = (shape) => {
    const n = shape.length;
    const last = shape[n - 1];
    if (n === 3) return 'NTW';
    if (n === 4) return shape[2] <= 3 && shape[3] > 8 ? 'NTFW' : 'NTHW';
    if (n === 5) return last <= 3 ? 'NTHWF' : (shape[2] <= 3 ? 'NTFHW' : 'NTDHW');
    if (n === 6) return last <= 3 ? 'NTDHWF' : 'NTFDHW';
    return '';
  };
  const showVar = () => {
    const v = variables.find((x) => x.name === varSel.value);
    if (!v) return;
    const s = suggestAxes(v.shape);
    if (s && !fd.axes.value) fd.axes.value = s;
    document.getElementById('dd-parsed').textContent =
      `Array "${v.name}" has shape (${v.shape.join(', ')}), ${v.dtype}. ${s ? `Suggested axis letters: ${s}. Check that they match your data.` : ''}`;
  };
  varSel.addEventListener('change', showVar);

  document.getElementById('dd-inspect').addEventListener('click', async (ev) => {
    const btn = ev.currentTarget;
    dErr.textContent = '';
    const source = dHolder.source || (await parseInto(fd.link, document.getElementById('dd-parsed'), dHolder));
    if (!source) { dErr.textContent = 'Add a link or path first.'; return; }
    btn.disabled = true;
    btn.textContent = 'Looking inside…';
    try {
      const res = await api.inspect(source);
      variables = res.variables;
      varSel.replaceChildren(...variables.map((v) => new Option(`${v.name}  (${v.shape.join('×')})`, v.name)));
      varWrap.hidden = false;
      fd.axes.value = '';
      showVar();
    } catch (e) {
      dErr.textContent = e.message;
    } finally {
      btn.disabled = false;
      btn.textContent = 'Look inside the file';
    }
  });

  document.getElementById('add-dataset').addEventListener('click', () => {
    fd.reset();
    dErr.textContent = '';
    varWrap.hidden = true;
    variables = [];
    document.getElementById('dd-parsed').textContent = '';
    dd.showModal();
  });

  fd.addEventListener('submit', async (ev) => {
    if (ev.submitter?.value !== 'save') return;
    ev.preventDefault();
    dErr.textContent = '';
    const source = dHolder.source || (await parseInto(fd.link, document.getElementById('dd-parsed'), dHolder));
    if (!source) { dErr.textContent = 'Add a link or path to the data file.'; return; }
    const ev2 = fd.eval.value;
    const layout = fd.axes.value.trim()
      ? { axes: fd.axes.value.trim() }
      : { fields: +fd.fields.value, components: +fd.components.value, depth: +fd.depth.value, height: +fd.height.value, width: +fd.width.value };
    const card = {
      id: slug(fd.card_name.value), name: fd.card_name.value.trim(), description: fd.description.value.trim(), source, layout,
      variable: varWrap.hidden ? null : varSel.value || null,
      field_names: fd.field_names.value.split(',').map((s) => s.trim()).filter(Boolean),
      eval: ev2 === 'last_fraction' ? { mode: ev2, fraction: 0.1 } : ev2 === 'first_n' ? { mode: ev2, n: 10 } : { mode: 'all' },
    };
    try {
      await api.addCard('datasets', card);
      dd.close();
      toast(`Saved dataset ${card.name}`);
      onSaved('datasets', card.id);
    } catch (e) { dErr.textContent = e.message; }
  });
}
