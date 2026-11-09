"""Clean raw build data and engineer features.

Every history-based feature uses shift(1) so a build never sees its own outcome.
"""
import pandas as pd

FEATURES = [
    "lines_changed", "files_changed", "churn_ratio", "hour", "weekday",
    "is_main", "is_pr", "prev_failed", "recent_fail_rate", "author_fail_rate",
]


def build_features(df):
    df = df.drop_duplicates(subset=["repo", "run_id"]).copy()
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df = df.sort_values(["created_at"]).reset_index(drop=True)
    cols = ["additions", "deletions", "files_changed"]
    df[cols] = df[cols].fillna(0)

    df["failed"] = (df["conclusion"] == "failure").astype(int)
    df["lines_changed"] = df["additions"] + df["deletions"]
    df["churn_ratio"] = df["additions"] / (df["lines_changed"] + 1)
    df["hour"] = df["created_at"].dt.hour
    df["weekday"] = df["created_at"].dt.weekday
    df["is_main"] = df["branch"].isin(["main", "master"]).astype(int)
    df["is_pr"] = (df["event"] == "pull_request").astype(int)

    by_repo = df.groupby("repo")["failed"]
    df["prev_failed"] = by_repo.shift(1).fillna(0)
    df["recent_fail_rate"] = by_repo.transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()).fillna(0)
    df["author_fail_rate"] = df.groupby(["repo", "author"])["failed"].transform(
        lambda s: s.shift(1).expanding().mean()).fillna(0)
    return df
