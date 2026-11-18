# Using the app

![Metrics](images/metrics.png)

## Run models

1. **Models:** tick one or more (up to 8, so each one keeps its own color). Select **Card** to
   see where a model comes from, its settings and its links.
2. **Datasets:** tick one or more.
3. **Settings:**

   | Setting | Meaning |
   |---|---|
   | Trajectories per dataset | how many trajectories each model is evaluated on. Start small on a CPU. |
   | Rollout steps | how far the models predict on their own (0 = as far as the data goes) |
   | Start rollout after step | the last true frame the models are given |
   | Trajectories saved for viewing | how many trajectories you can look at in the viewer |
   | Device | *Automatic* uses a GPU if PyTorch finds one |

4. Select **Run**. Every model runs on every dataset, one combination after another. Progress
   shows for each combination, and results appear as soon as each one finishes.

Past runs stay available from the **Results** menu at the top of the page. **Download metrics
(CSV)** saves a table of all the numbers in a run.

## Two kinds of prediction

- **One-step:** every frame is predicted from the true frame(s) just before it. This measures
  accuracy without errors building up.
- **Rollout:** the model gets the true frame(s) once, then feeds its own predictions back in.
  This shows how errors grow, which matters if you want to use the model as a simulator.

Every model in a run is scored on the same time steps. A model that needs more input frames
than the others moves the start for all of them.

## Looking at predicted fields

Choose the dataset, trajectory and field at the top of **Predicted fields**. For 3D data, also
choose a slice direction and position. There are two views:

**All rollout steps** (the default) shows one model's whole rollout at a glance:

```
            t=1   t=2   t=3   ...   t=20
Truth      [  ]  [  ]  [  ]  ...   [  ]
MORPH-Ti   [  ]  [  ]  [  ]  ...   [  ]
```

- Pick the model with the **Model** menu.
- Truth and prediction share one color scale, so differences show up directly.
- When there are many steps, **Steps shown** picks every 2nd, 4th, ... step; the last step is
  always shown. Choose *all* to see every step and scroll sideways.
- Hover over a tile to read the truth, prediction and error at that point.
- Click a tile to open that step in the other view.
- 1D data shows as two pictures, truth and prediction, with the grid across and time down.

**One step at a time** compares every model at a single time step:

- **Truth** sits on the left. Each model gets a column with its prediction above its error.
  The error scale also stays fixed over time.
- Switch between **Rollout** and **One-step** predictions.
- The **timeline** under the panels plots each model's error at every step. Drag the slider,
  click the chart, or press play.
- 1D data shows as line charts.

## Metrics

All metrics compare a prediction with the truth, in the data's own units.

| Metric | Definition |
|---|---|
| MSE, RMSE, MAE | mean squared error, its square root, mean absolute error |
| NRMSE | sqrt( mean((pred − true)²) / mean(true²) ) |
| VRMSE | sqrt( mean((pred − true)²) / var(true) ) |

NRMSE and VRMSE are computed per trajectory, then averaged, with the same definitions as MORPH's
evaluation code. They have no units, so you can compare them across datasets. Lower is better.
In the table, the best value per dataset is in bold.
