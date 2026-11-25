# End of the World: Global Stress Index

Yearly Global Stress Index for 1959-2024 from four domains (climate, mortality, disasters, pandemic),
projected to 2100. Woosong University, Introduction to Machine Learning, Fall 2026.

Report and guide: see `docs/` (published with GitHub Pages).

## Who works where

Each member works on **their own branch**, created in advance. Do not commit to `main`.

| Member | Domain | Module file | Branch |
|---|---|---|---|
| Integrator | disasters, pandemic | `modules/disasters_pandemics.py` | `disasters-pandemics` |
| Climate | climate | `modules/climate.py` | `climate` |
| Birth and death rates | mortality | `modules/demographics.py` | `mortality` |

(If the branch names on GitHub differ from this table, use the names on GitHub and fix the table.)

## What to upload to your branch

```bash
git fetch origin
git checkout <your-branch>
# ...work...
git add modules/<your_module>.py data/processed/<domain>.csv data/processed/<domain>_meta.json outputs/<your_module>/
git commit -m "Describe what you did"
git push origin <your-branch>
```

Upload:
* your module code in `modules/`
* your two hand-off files in `data/processed/`: `<domain>.csv` and `<domain>_meta.json`
* your results folder `outputs/<your_module>/` (tables and figures for your report section)

Do not upload raw downloads (`data/raw/` is ignored), virtual environments or zip files.
Do not edit other people's files or the shared files (`config.py`, `common/`, `combine.py`);
ask the integrator if something there needs to change.

If your files are not in the hand-off format, upload them anyway. The integrator converts them
with `tools/adopt.py` and records every choice.

## Folders

```
docs/          website: report and team guide
common/        shared code
modules/       one module per member
tools/         integration and report-page tools
tests/         automated checks
data/raw/      raw downloads (not uploaded)
data/processed/  hand-off files, one pair per domain
outputs/       each module's results
```

Shared code and the full setup steps are added to `main` by the integrator.
