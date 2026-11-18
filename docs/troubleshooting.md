# Troubleshooting

The **Run log** at the bottom of a run has the full error messages.

| You see | What to do |
|---|---|
| "Couldn't download … 401" or "… 403" | Check the repo and file names. Check that this machine can reach huggingface.co or github.com (proxy or firewall). For private repos, set `HF_TOKEN` or `GITHUB_TOKEN` before starting. |
| A model says **Needs a package** | Open its card: it shows the `pip install` command. Install the package and reload the page. |
| A model says **Problem** | Open its card. Usually the local weights file doesn't exist at the path given. |
| "can't be reshaped" | The layout on the dataset card doesn't match the file. In **Add dataset**, use **Look inside the file** and set the axis letters ([Adding datasets](adding-datasets.md)). |
| **Skipped: …** | That model can't run on that dataset: too many fields, a grid not divisible by 8 for MORPH, or a different shape than an exported model expects. The message says which. |
| Out of memory, or very slow | Lower **Trajectories per dataset**, or lower the model's `batch_size` in its card. Use a GPU for MORPH-M and MORPH-L. |
| "Interrupted (the server restarted)" | The server stopped during a run. Start the run again. |
| "Address already in use" | Another program uses port 8000. Run `python studio.py --port 8080`. |
| The page is blank or stale after an update | Reload it while bypassing the cache (Ctrl+Shift+R, or Cmd+Shift+R on Mac). |
