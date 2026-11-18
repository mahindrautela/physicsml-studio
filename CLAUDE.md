# Working on PhysicsML Studio

## Guidelines from the repository owner

1. Keep the code **clean, clear, and easy to understand and update**. Prefer small modules
   with one job each, plain names, and docstrings that explain why. Avoid clever tricks.
2. Keep **README.md super short**: what the app is, a screenshot, quick start, and links.
   Details go in `docs/`, linked from the README.

## How changes are made

- Work on a branch and open a pull request. Don't push to `main`.
- Before opening a PR: run `python -m pytest -q`, start `python studio.py`, and check any
  page you changed in a browser.
- When behavior changes, update the matching guide in `docs/`. Keep the README links valid.

## Map

`docs/architecture.md` lists every file, what a run does step by step, and where to make
common changes. Model types live in `app/adapters/` (start from `template.py`). Cards are YAML
files in `catalog/`.
