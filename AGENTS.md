# AGENTS.md

## Cursor Cloud specific instructions

### Repository overview

This is a **static catalog monorepo** (632 LLNL open-source projects) with no top-level application server or unified build system. The "product" is the organizational structure and machine-readable index files (`INDEX.json`, `TAGS.json`). There is no top-level `package.json`, `Makefile`, `setup.py`, or `docker-compose.yml`.

### Development tools

- **Python 3** (stdlib only) — all CI validation and index-generation scripts are inline Python in `.github/workflows/`. No external Python packages are required for the repo-level scripts.
- **Linting tools** (`ruff`, `black`, `isort`) — installed via pip for linting Python files across sub-projects. These are in `~/.local/bin`.

### Key commands

| Task | Command |
|------|---------|
| Validate INDEX.json | `python3 -c "import json; json.load(open('INDEX.json'))"` |
| Validate TAGS.json | `python3 -c "import json; json.load(open('TAGS.json'))"` |
| Validate domain structure | See inline script in `.github/workflows/ci-integration.yml` (`validate-structure` job) |
| Lint Python files | `ruff check --select=E,F,W --ignore=E501 <domain>/<project>/` |
| Regenerate INDEX.json | See inline script in `.github/workflows/update-index.yml` |
| Dependency analysis | See inline script in `.github/workflows/analyze-dependencies.yml` |

### Gotchas

- The CI workflows use Python 3.11, but Python 3.12 is available in the Cloud Agent VM and works fine for all repo-level scripts.
- Linting runs with `continue-on-error: true` in CI — existing sub-projects have many lint findings. This is expected and not a blocker.
- The index generation script counts may differ slightly from the committed `INDEX.json` if sub-projects were added/removed without re-running the workflow.
- Individual sub-projects have their own independent build systems and dependencies. Working on a specific sub-project requires reading that project's own README and installing its specific dependencies.
