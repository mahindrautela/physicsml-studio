// The page's shared state. Each module reads it; only the module named next to a field changes it.
export const state = {
  system: null,            // app.js:     PyTorch version, devices, GPUs (from /api/system)
  catalog: null,           // catalog.js: model and dataset cards (from /api/catalog)
  selected: {              // catalog.js: ticked cards
    models: new Set(),
    datasets: new Set(),
  },
  openCards: new Set(),    // catalog.js: cards whose details are expanded, as "models:<id>"
  run: null,               // runs.js:    the run being shown (from /api/runs/<id>)
};
