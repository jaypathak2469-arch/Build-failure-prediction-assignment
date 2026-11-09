from src.features import FEATURES, build_features
from src.make_synthetic import make_synthetic
from src.train import time_split


def test_features_have_no_nans():
    df = build_features(make_synthetic(n=300))
    assert not df[FEATURES].isna().any().any()
    assert set(df["failed"].unique()) <= {0, 1}


def test_first_build_has_no_history_leak():
    df = build_features(make_synthetic(n=300))
    assert df.loc[0, "prev_failed"] == 0


def test_time_split_is_chronological():
    df = build_features(make_synthetic(n=300))
    tr, te = time_split(df)
    assert tr["created_at"].max() <= te["created_at"].min()
