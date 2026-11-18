# Adding models

A model is described by a **card**: a small YAML file in `catalog/models/`. The card says what
the model is, where its weights file is, and which **model type** runs it.

## With the form

Select **Add model**, choose the model type, and paste a link or path to the weights file:

- Hugging Face: `https://huggingface.co/<owner>/<repo>/blob/main/path/to/model.pth`
- GitHub: `https://github.com/<owner>/<repo>/blob/<branch>/path/to/model.pt2`
- any web address: `https://example.org/model.pt2`
- a file on the machine running the app: `/data/models/model.pth`

The form writes `catalog/models/<id>.yaml`, which you can edit later. New or edited cards show
up when you reload the page.

## As a YAML file

```yaml
# catalog/models/morph-ti-mydata.yaml
id: morph-ti-mydata               # lowercase letters, digits, - _ .
name: MORPH-Ti fine-tuned on my data
adapter: morph                    # the model type
source:
  type: huggingface               # huggingface | github | url | local
  repo_id: my-group/my-models
  filename: ft_morph-Ti-mydata.pth
  subfolder: checkpoints          # optional
  revision: main                  # optional: branch, tag or commit
config:                           # settings for the model type (see below)
  size: Ti
  lora_r_attn: 16
  lora_r_mlp: 12
requires: [morph_pde]             # Python packages the model needs
description: Level-1 fine-tune on our dataset.
```

The other source types use `repo_id` + `filename` (+ `revision`) for GitHub, `url` for a web
address, and `path` for a local file. A local path may start with `$STUDIO_HOME`.

## Model types

| `adapter` | Use it for | Weights file |
|---|---|---|
| `morph` | MORPH checkpoints, pretrained or fine-tuned (see [MORPH notes](morph.md)) | `.pth` |
| `pytorch` | any PyTorch model, without writing code for the app | `.pt2` or `.pt` |
| `persistence` | baseline: predicts no change | none |
| `linear_extrapolation` | baseline: next = 2 × current − previous | none |

The settings each type accepts are listed at the top of its file in `app/adapters/`.

### Any PyTorch model (`pytorch`)

Export the model so it takes one tensor shaped `(B, k, F, C, D, H, W)` and returns the next
frame, shaped `(B, F, C, D, H, W)`. The letters are explained in [Adding datasets](adding-datasets.md).

```python
import torch

model.eval()
example = torch.randn(8, 1, 2, 1, 1, 128, 128)   # 8 samples, 1 input frame, 2 fields, 128×128 grid
torch.export.save(torch.export.export(model, (example,)), "my_model.pt2")
```

- Export with a batch of 8, or set `batch_size` in the card to your batch size.
- An exported model only accepts the grid it was exported with. On other datasets it's marked
  *skipped*.
- Settings: `ar_order` (input frames, default 1), `normalization` (`none`, `trajectory` or
  `context`, default `none`) and `batch_size`.
- Older TorchScript files (`torch.jit.trace(...).save("model.pt")`) work too.

### A model that needs its own code

Use this when a model needs a package from GitHub or its own preprocessing:

1. Copy `app/adapters/template.py` to `app/adapters/<name>.py`.
2. Set `@register("<name>")` and fill in `load()` (build the model, load weights) and `predict()`
   (frames in, next frame out, both in the data's units).
3. Add the package to `requirements.txt`, and add a card with `adapter: <name>` and
   `requires: [<package>]`.
4. Restart `python studio.py`.

`app/adapters/morph.py` is a complete example.
