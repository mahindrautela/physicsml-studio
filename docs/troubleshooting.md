# Troubleshooting

## Starting the app

| You see | What to do |
|---|---|
| "Some packages are missing" or `No module named 'fastapi'` | Run `pip install -r requirements.txt` in the environment you start the app from. The message shows which Python is running. |
| `pip install` fails on the `git+https://github.com/lanl/MORPH...` line | git isn't installed, or GitHub is blocked. Delete that line from `requirements.txt` and install again. Everything except MORPH works without it. |
| The window seems stuck while simulating the demo data | It shows a percentage as it goes and can take up to a minute. On Windows, clicking inside the window pauses the program; if the title starts with "Select", press Esc. To skip this step, run `python studio.py --skip-demo`. |
| "Address already in use" | Another program uses port 8000. Run `python studio.py --port 8080`. |

## The page doesn't open

The page only exists while `python studio.py` is running and its window shows
`PhysicsML Studio is running at http://127.0.0.1:8000`. That address means "this computer",
so it works in a browser on the same computer that runs the app.

**The app and the browser are on the same computer:**

1. Open http://127.0.0.1:8000 yourself if no browser tab opened.
2. A work proxy may be blocking local addresses. Let the browser skip the proxy for them:
   - Firefox: Settings → Network Settings → add `localhost, 127.0.0.1` to "No proxy for".
   - Chrome on Windows: Internet Options → Connections → LAN settings → tick "Bypass proxy
     server for local addresses".
   - Chrome on Mac: System Settings → Network → your connection → Details → Proxies → add
     `localhost, 127.0.0.1` to the bypass list.
3. The page is blank or old after an update: reload while bypassing the cache (Ctrl+Shift+R,
   or Cmd+Shift+R on Mac).

**The app runs on a server or cluster, and the browser is on your laptop:** the firewall blocks
the port, so go through SSH. Leave the app running on the server, and on your laptop run:

```bash
ssh -L 8000:localhost:8000 you@server
```

Then open http://127.0.0.1:8000 on the laptop. If the app runs on a compute node behind a login
node, use `ssh -J you@login-node -L 8000:localhost:8000 you@compute-node`. Don't use
`--host 0.0.0.0` on a shared machine: the app has no login.

## During a run

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
