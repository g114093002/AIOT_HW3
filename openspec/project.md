# Project Context

## Purpose
This repository contains homework and small projects for the "物聯網與數據分析" (IoT and Data Analysis) course. The goals are:
- Collect, process, and analyze IoT sensor data
- Build reproducible data pipelines and experiments
- Document requirements and proposed changes using OpenSpec

## Tech Stack
- Python 3.10+ for data processing and analysis (pandas, numpy, scikit-learn)
- Jupyter / JupyterLab for interactive notebooks and exploration
- Node.js (LTS) used for developer tooling (OpenSpec CLI, npm scripts)
- MQTT (e.g., Mosquitto) for local IoT message brokering during experiments
- SQLite or lightweight time-series files (CSV/Parquet) for local storage
- Git + GitHub (or similar) for source control and PR-based collaboration

## Project Conventions

### Code Style
- Python: follow PEP8; use black for formatting and isort for imports
- Notebooks: keep exploratory work in `notebooks/`, minimal long-running notebooks in `reports/`
- Config and secrets: keep in `config/` and never commit secrets; use environment variables or a `.env` for local runs

### Architecture Patterns
- Small modular scripts and notebooks that can be composed into pipelines
- Separation between data ingestion (MQTT, file loads), processing (ETL), and analysis/modeling
- Keep experiments reproducible by pinning dependencies in a `requirements.txt` or `environment.yml`

### Testing Strategy
- Unit tests for core transformation functions using `pytest`
- Lightweight integration tests for end-to-end data flow where feasible (small datasets)
- Linting and static checks run locally or in CI (black, flake8/pylint)

### Git Workflow
- Branch-per-feature or branch-per-homework (e.g., `feature/xyz` or `hw3/solution`)
- Use PRs for review; include updated `openspec/changes/` proposal when proposing behavior changes
- Commits: short and descriptive; reference issue or change-id when relevant

## Domain Context
- Data originates from small IoT sensors (temperature, humidity, light, etc.) sampled at modest rates (e.g., 1s–1m)
- Units and sampling rates must be clearly documented per dataset
- Typical dataset sizes for homework: tens of MB to low hundreds of MB

## Important Constraints
- Work primarily developed and tested on Windows (PowerShell / cmd), so be mindful of path and shell differences
- PowerShell execution policy may block `npm`/`npx` wrappers — use the `.cmd` shims or run in `cmd.exe` when needed
- No heavy cloud infra required for homework; keep solutions runnable locally

## External Dependencies
- Local MQTT broker (Mosquitto) for message simulation
- Python packages: pandas, numpy, scikit-learn, matplotlib/plotly
- Node: OpenSpec CLI (`@fission-ai/openspec`) for spec workflows

## Contacts / Owners
- Primary: repository owner (update with actual name/email)
- For OpenSpec/QA: use the change author and reviewers listed in the proposal files

## OpenSpec workflow & local validation
- Use the `openspec/changes/README.md` (created in this repo) for quick local validation commands and CI guidance.
- On Windows prefer running OpenSpec from `cmd.exe` or call the `.cmd` shims when using PowerShell to avoid execution policy issues. Example:

```powershell
# from PowerShell run via cmd to avoid npm .ps1 wrapper issues
cmd /c "openspec validate <change-id> --strict"
```

If you prefer to relax PowerShell restrictions, update the current user execution policy (understand security implications):

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
```

---

If any of the above assumptions are incorrect (for example you prefer another language or have cloud infra), tell me which parts to change and I will update this file.
