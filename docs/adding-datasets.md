# Adding datasets

A dataset is a `.npy`, `.nc` (NetCDF) or `.h5` (HDF5) file holding many simulation trajectories.
It's described by a card in `catalog/datasets/`.

## With the form

Select **Add dataset** and paste a link or path to the file (Hugging Face, GitHub, any web
address, or a local path). Then select **Look inside the file**. The app downloads the file
if needed, lists the arrays in it, and suggests how the array is laid out.

## How data is laid out

Every dataset is turned into seven axes, **(N, T, F, C, D, H, W)**:

| Letter | Meaning |
|---|---|
| N | trajectories (separate simulations) |
| T | time steps |
| F | fields, e.g. velocity and pressure |
| C | components of a field, e.g. x, y, z of a velocity |
| D, H, W | the grid: depth, height, width. 1D data has D = H = 1; 2D data has D = 1 |

Describe your file in one of two ways:

- **Axis letters** (recommended): the order of the axes in the stored array, starting with `NT`.
  Missing letters become size 1.

  | Your array | Axis letters |
  |---|---|
  | 1D, one field: (traj, time, x) | `NTW` |
  | 2D, fields first: (traj, time, field, y, x) | `NTFHW` |
  | 2D, channels last: (traj, time, y, x, field) | `NTHWF` |
  | 3D, channels last | `NTDHWF` |

- **Sizes:** `fields`, `components`, `depth`, `height`, `width`. The array is reshaped to that
  layout, exactly like MORPH's `--dataset_specs F C D H W`.

## As a YAML file

```yaml
# catalog/datasets/my-sw2d.yaml
id: my-sw2d
name: Shallow water 2D (ours)
source:
  type: github
  repo_id: my-group/sim-data
  revision: main
  filename: sw2d/test.h5          # files stored with Git LFS work too
variable: data                    # array inside a .nc/.h5 file; leave out to use the largest
layout: {axes: NTHWF}
field_names: [h]
eval: {mode: last_fraction, fraction: 0.1}
```

## Which trajectories are evaluated

| `eval` | Trajectories used |
|---|---|
| `{mode: last_fraction, fraction: 0.1}` | the last 10% of the file |
| `{mode: all}` | all of them |
| `{mode: first_n, n: 20}` | the first 20 |
| `{mode: indices, indices: [3, 7]}` | exactly these |

The **Trajectories per dataset** setting then caps how many of those a run uses. Only those
trajectories are read from disk, so large files are fine.
