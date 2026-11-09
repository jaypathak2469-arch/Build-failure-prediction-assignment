import pandas as pd

from src.features import FEATURES, build_features
from src.make_synthetic import make_synthetic
from src.train import time_split


def two_repos():
    a = make_synthetic(n=300, seed=1)
    b = make_synthetic(n=300, seed=2)
    b["repo"] = "demo/other"
    b["run_id"] = b["run_id"] + 10_000
    b["created_at"] = b["created_at"] + pd.Timedelta(days=200)
    return pd.concat([a, b], ignore_index=True)


def test_features_have_no_nans():
    df = build_features(make_synthetic(n=300))
    assert not df[FEATURES].isna().any().any()
    assert set(df["failed"].unique()) <= {0, 1}


def test_first_build_has_no_history_leak():
    df = build_features(make_synthetic(n=300))
    assert df.loc[0, "prev_failed"] == 0


def test_split_is_chronological_within_each_repo():
    df = build_features(two_repos())
    tr, te = time_split(df)
    for repo in df["repo"].unique():
        assert tr[tr.repo == repo]["created_at"].max() <= te[te.repo == repo]["created_at"].min()


def test_every_repo_appears_in_test_set():
    df = build_features(two_repos())
    _, te = time_split(df)
    assert set(te["repo"]) == set(df["repo"])
