# Build-Failure Prediction

Predicts whether a CI build will fail from commit and history features, with the
whole workflow automated through GitHub Actions (lint, test, train, report, weekly retrain).

## Quick start
```bash
pip install -r requirements.txt
python -m src.train        # uses synthetic data if data/builds.csv is missing
```

## Use real data
```bash
export GITHUB_TOKEN=<your token>
python -m src.collect_data owner/repo another/repo --max-runs 300
python -m src.train
```

## Pipeline
1. `src/collect_data.py` pulls completed workflow runs and commit stats from the GitHub API
2. `src/features.py` cleans data and builds leak-free history features
3. `src/train.py` does a time-based split, trains Logistic Regression and Random Forest, and writes metrics and plots to `outputs/`
4. `.github/workflows/ci.yml` runs lint, tests and training on every push, and retrains weekly

## Features
lines changed, files changed, churn ratio, hour, weekday, main branch, pull request,
previous build failed, recent failure rate (last 5), author failure rate.
