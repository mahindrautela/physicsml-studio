# How the code is organized

PhysicsML Studio is a small Python web server (FastAPI) and one web page (plain HTML, CSS and
JavaScript, with no build step). The server does all the computing. The page only displays
what the server sends.

```
studio.py               start the app (creates the demo data on first start)
requirements.txt
catalog/
  models/*.yaml         model cards
  datasets/*.yaml       dataset cards
app/                    the server
  main.py               web API routes: thin, they call the modules below
  config.py             folders and environment variables
  catalog.py            card formats; read, save and delete cards
  readiness.py          is a card ready to run? (the status dot)
  sources.py            download files from Hugging Face, GitHub or a URL
  datasets.py           read .npy/.nc/.h5 into the 7-D layout (N, T, F, C, D, H, W)
  runs.py               run records: start, list, read, stop, delete, and where files go
  runner.py             what a run does: one-step and rollout predictions, metrics, saving
  frames.py             slices of saved predictions for the field viewer
  metrics.py            MSE, RMSE, MAE, NRMSE, VRMSE; CSV export
  devices.py            CPU or GPU
  adapters/             one file per model type
    base.py             the interface every model type implements
    morph.py            MORPH
    pytorch.py          any exported PyTorch model (.pt2 / .pt)
    baselines.py        persistence and linear extrapolation
    template.py         copy this to add a model type
    _revin.py, _torch.py  shared helpers
static/                 the page
  index.html            layout and dialogs
  css/app.css           all styling
  js/app.js             startup
  js/catalog.js         model and dataset lists (left column)
  js/runs.js            start runs, Results menu, progress
  js/viewer.js          field viewer and timeline
  js/metrics.js         metrics table and charts
  js/forms.js           Add model / Add dataset dialogs
  js/api.js             calls to the server
  js/state.js           shared page state
  js/colors.js          color palettes and chart styling
  js/util.js            small helpers
  vendor/, fonts/       Plotly (charts) and the Atkinson Hyperlegible font, served locally
tools/make_demo_data.py simulate the demo datasets
tests/                  python -m pytest -q
```

## What happens when you select Run

1. `js/runs.js` posts the selected cards and settings to `POST /api/runs`.
2. `runs.py` validates them, writes `workspace/runs/<id>/run.json`, and queues the run. One run
   executes at a time, on a background thread.
3. `runner.py` loads each model once (`adapters/`, `sources.py`) and reads each dataset once
   (`datasets.py`). For every model × dataset it predicts, scores (`metrics.py`), and saves the
   predictions as `.npy` files. It updates `run.json` as it goes.
4. The page polls `GET /api/runs/<id>` every second and redraws. The viewer asks
   `GET /api/runs/<id>/frame` (`frames.py`) for one time step at a time.

## Where to make common changes

| To change | Edit |
|---|---|
| support a new kind of model | add a file in `app/adapters/` ([Adding models](adding-models.md)) |
| how a data file is read | `app/datasets.py` |
| a metric, or add one | `app/metrics.py`, then the table columns in `static/js/metrics.js` |
| what a run computes | `app/runner.py` |
| a setting in the Settings panel | `RunSettings` in `app/runs.py`, the field in `static/index.html`, `readSettings()` in `static/js/runs.js` |
| an API route | `app/main.py` (keep it thin) and `static/js/api.js` |
| how the page looks | `static/css/app.css`. Colors used in charts are in `static/js/colors.js`. |

## API

| Method and path | Returns |
|---|---|
| `GET /api/system` | Python and PyTorch versions, devices, GPUs |
| `GET /api/catalog` | all cards with their status, and the model types |
| `POST /api/models`, `POST /api/datasets` | save a new card |
| `DELETE /api/models/<id>`, `DELETE /api/datasets/<id>` | delete a card |
| `POST /api/parse-link` | a card source from a pasted link |
| `POST /api/inspect` | the arrays inside a data file |
| `POST /api/runs` | start a run |
| `GET /api/runs`, `GET /api/runs/<id>` | run summaries, or one run record |
| `POST /api/runs/<id>/cancel`, `DELETE /api/runs/<id>` | stop or delete a run |
| `GET /api/runs/<id>/frame` | one viewer frame |
| `GET /api/runs/<id>/metrics.csv` | all metrics of a run |

FastAPI also serves interactive API docs at http://127.0.0.1:8000/docs.

## Checking a change

```bash
python -m pytest -q        # backend tests, about 5 seconds, no downloads
python studio.py           # then try what you changed in the browser
```

The tests use a temporary catalog and workspace, so they never touch your own files. The MORPH
test uses random weights and runs only when MORPH's code is installed.
