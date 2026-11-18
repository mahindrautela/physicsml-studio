# MORPH notes

[MORPH](https://github.com/lanl/MORPH/tree/package-morph) is a PDE foundation model from LANL
([paper](https://arxiv.org/abs/2509.21670)). The catalog has cards for the four pretrained
sizes: Ti, S, M and L. Their weights download from Hugging Face
([mahindrautela/MORPH](https://huggingface.co/mahindrautela/MORPH)) the first time you run them.
MORPH-L is large, so use a GPU for it.

## How it runs here

The adapter (`app/adapters/morph.py`) follows MORPH's own scripts
(`scripts/finetune_MORPH_general.py` with `--n_epochs 0`):

1. Data in the seven-axis layout (N, T, F, C, D, H, W), which MORPH calls UPTF-7.
2. Each trajectory is normalized by its own mean and variance per field (RevIN).
3. MORPH predicts the next frame.
4. The normalization is undone, and the metrics are computed in the data's units.

The pretrained cards are zero-shot: they use the checkpoints with no fine-tuning.

## Differences from MORPH's scripts

- **Normalization:** MORPH's scripts compute each test trajectory's statistics from **all** of
  its time steps. The cards keep that (`normalization: trajectory`) so the numbers match. For
  a stricter test that uses only the frames given to the model, set
  `normalization: context` in the card's `config`.
- **Test trajectories:** MORPH's scripts pick them with a random 80/10/10 split. The FNS-KF
  card uses the last 10% of the file instead.

## Limits

- At most 3 fields and 3 components.
- Grids up to 128 × 128 × 128. Every grid side longer than 1 must be divisible by 8.

The app checks these before running and marks combinations that don't fit as *skipped*.

## Your own fine-tuned checkpoints

Add a card with `adapter: morph`, pointing at your `.pth` file. If you fine-tuned with LoRA
(levels 1 to 3 in MORPH's script), set the same ranks in the card's `config`. MORPH's script
defaults are `lora_r_attn: 16` and `lora_r_mlp: 12`. See [Adding models](adding-models.md).
