"""Generate a synthetic CI-build dataset with the same schema as collect_data.py.

Used so the pipeline and the GitHub Actions workflow run without API access.
"""
import numpy as np
import pandas as pd


def make_synthetic(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-01")
    created = start + pd.to_timedelta(np.sort(rng.integers(0, 60 * 24 * 300, n)), unit="m")
    additions = rng.gamma(1.5, 40, n).astype(int)
    deletions = rng.gamma(1.2, 25, n).astype(int)
    files = np.clip((additions + deletions) // 40 + rng.integers(1, 4, n), 1, None)
    authors = rng.choice([f"dev{i}" for i in range(15)], n)
    author_risk = {a: rng.uniform(0.0, 0.15) for a in set(authors)}
    branch = rng.choice(["main", "feature", "fix"], n, p=[0.3, 0.5, 0.2])
    event = rng.choice(["push", "pull_request"], n, p=[0.6, 0.4])

    logit = (-3.2 + 0.004 * (additions + deletions) + 0.05 * files
             + np.array([author_risk[a] * 6 for a in authors])
             + np.where(branch == "feature", 0.4, 0.0))
    p = 1 / (1 + np.exp(-logit))
    # failures cluster: a failed build raises the chance the next one fails
    fail = np.zeros(n, dtype=bool)
    for i in range(n):
        boost = 0.25 if i > 0 and fail[i - 1] else 0.0
        fail[i] = rng.random() < min(p[i] + boost, 0.95)

    return pd.DataFrame({
        "repo": "demo/repo", "run_id": np.arange(n), "created_at": created,
        "author": authors, "branch": branch, "event": event,
        "additions": additions, "deletions": deletions, "files_changed": files,
        "conclusion": np.where(fail, "failure", "success"),
    })


if __name__ == "__main__":
    df = make_synthetic()
    df.to_csv("data/builds.csv", index=False)
    print(f"Wrote data/builds.csv: {len(df)} rows, failure rate {(df.conclusion == 'failure').mean():.2%}")
