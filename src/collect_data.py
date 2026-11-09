"""Collect real CI build data from the GitHub Actions API.

Usage:
    export GITHUB_TOKEN=ghp_xxx          # avoids tight rate limits
    python -m src.collect_data owner/repo [owner/repo ...] --max-runs 300
"""
import argparse
import os
import time

import pandas as pd
import requests

API = "https://api.github.com"


def _get(url, params=None):
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def collect_repo(repo, max_runs=300):
    rows, page = [], 1
    while len(rows) < max_runs:
        data = _get(f"{API}/repos/{repo}/actions/runs",
                    {"per_page": 100, "page": page, "status": "completed"})
        runs = data.get("workflow_runs", [])
        if not runs:
            break
        for run in runs:
            if run["conclusion"] not in ("success", "failure"):
                continue
            try:
                commit = _get(f"{API}/repos/{repo}/commits/{run['head_sha']}")
                stats = commit.get("stats", {})
                files = len(commit.get("files", []))
            except requests.HTTPError:
                continue
            rows.append({
                "repo": repo, "run_id": run["id"], "created_at": run["created_at"],
                "author": (run.get("actor") or {}).get("login", "unknown"),
                "branch": run["head_branch"], "event": run["event"],
                "additions": stats.get("additions", 0),
                "deletions": stats.get("deletions", 0),
                "files_changed": files, "conclusion": run["conclusion"],
            })
            time.sleep(0.1)
            if len(rows) >= max_runs:
                break
        page += 1
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="+")
    ap.add_argument("--max-runs", type=int, default=300)
    args = ap.parse_args()
    df = pd.concat([collect_repo(r, args.max_runs) for r in args.repos], ignore_index=True)
    df.to_csv("data/builds.csv", index=False)
    print(f"Saved {len(df)} runs to data/builds.csv")
