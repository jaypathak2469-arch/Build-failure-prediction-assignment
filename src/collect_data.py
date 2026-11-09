"""Collect real CI build data from the GitHub Actions API.

Usage:
    export GITHUB_TOKEN=ghp_xxx          # avoids tight rate limits
    python -m src.collect_data owner/repo [owner/repo ...] --max-runs 300

Network errors are retried with backoff. Results are saved after every repo,
and if a repo fails part-way the runs collected so far are kept.
"""
import argparse
import os

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API = "https://api.github.com"


def make_session():
    session = requests.Session()
    retry = Retry(total=6, backoff_factor=1.5, status_forcelist=[500, 502, 503, 504],
                  allowed_methods=["GET"])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["Accept"] = "application/vnd.github+json"
    token = os.getenv("GITHUB_TOKEN")
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
    return session


def get_json(session, url, params=None):
    r = session.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def commit_stats(session, repo, sha, cache):
    """Size of a commit, fetched once per commit (many runs share one commit)."""
    if sha not in cache:
        try:
            c = get_json(session, f"{API}/repos/{repo}/commits/{sha}")
            s = c.get("stats", {})
            cache[sha] = (s.get("additions", 0), s.get("deletions", 0), len(c.get("files", [])))
        except requests.HTTPError:
            cache[sha] = None
    return cache[sha]


def collect_repo(session, repo, max_runs=300):
    rows, cache, page = [], {}, 1
    try:
        while len(rows) < max_runs:
            data = get_json(session, f"{API}/repos/{repo}/actions/runs",
                            {"per_page": 100, "page": page, "status": "completed"})
            runs = data.get("workflow_runs", [])
            if not runs:
                break
            for run in runs:
                if run["conclusion"] not in ("success", "failure"):
                    continue
                stats = commit_stats(session, repo, run["head_sha"], cache)
                if stats is None:
                    continue
                rows.append({
                    "repo": repo, "run_id": run["id"], "created_at": run["created_at"],
                    "author": (run.get("actor") or {}).get("login", "unknown"),
                    "branch": run["head_branch"], "event": run["event"],
                    "additions": stats[0], "deletions": stats[1],
                    "files_changed": stats[2], "conclusion": run["conclusion"],
                })
                if len(rows) % 10 == 0:
                    print(f"{repo}: {len(rows)} runs collected", flush=True)
                if len(rows) >= max_runs:
                    break
            page += 1
    except requests.RequestException as e:
        print(f"WARNING: {repo} stopped early after {len(rows)} runs ({type(e).__name__})")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="+")
    ap.add_argument("--max-runs", type=int, default=300)
    args = ap.parse_args()
    session = make_session()
    frames = []
    for repo in args.repos:
        frames.append(collect_repo(session, repo, args.max_runs))
        df = pd.concat(frames, ignore_index=True)
        df.to_csv("data/builds.csv", index=False)
        print(f"Saved {len(df)} runs so far to data/builds.csv", flush=True)
