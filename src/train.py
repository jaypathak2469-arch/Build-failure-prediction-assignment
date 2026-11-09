"""Train and evaluate build-failure models with a time-based split."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,  # noqa: E402
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.features import FEATURES, build_features  # noqa: E402
from src.make_synthetic import make_synthetic  # noqa: E402

DATA = "data/builds.csv"
OUT = "outputs"


def load_data():
    if os.path.exists(DATA):
        print(f"Using {DATA}")
        return pd.read_csv(DATA)
    print("data/builds.csv not found, using synthetic data")
    return make_synthetic()


def time_split(df, test_frac=0.2):
    """Split each repo chronologically so every repo appears in both train and test."""
    train_parts, test_parts = [], []
    for _, g in df.groupby("repo"):
        g = g.sort_values("created_at")
        cut = int(len(g) * (1 - test_frac))
        train_parts.append(g.iloc[:cut])
        test_parts.append(g.iloc[cut:])
    return pd.concat(train_parts), pd.concat(test_parts)


def scores_at(y_true, pred):
    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    df = build_features(load_data())
    train, test = time_split(df)
    print(df.groupby('repo')['failed'].agg(builds='count', failure_rate='mean').round(3))
    print('Test failures per repo:', test.groupby('repo')['failed'].sum().to_dict())
    X_tr, y_tr = train[FEATURES], train["failed"]
    X_te, y_te = test[FEATURES], test["failed"]
    train_rate = y_tr.mean()
    print(f"Train {len(train)} / Test {len(test)} | failure rate {df['failed'].mean():.2%} "
          f"| failures in test set: {int(y_te.sum())}")

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", min_samples_leaf=5, random_state=42),
    }
    metrics = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        # Rule 1: default cutoff of 0.5
        pred_default = (proba >= 0.5).astype(int)
        # Rule 2: flag the riskiest builds, as many as the training failure rate
        cutoff = np.quantile(proba, 1 - train_rate)
        pred_topk = (proba >= cutoff).astype(int)
        metrics[name] = {
            "roc_auc": roc_auc_score(y_te, proba),
            "pr_auc": average_precision_score(y_te, proba),
            "at_threshold_0.5": scores_at(y_te, pred_default),
            "flag_top_risk_builds": scores_at(y_te, pred_topk),
        }
        ConfusionMatrixDisplay.from_predictions(
            y_te, pred_topk, display_labels=["pass", "fail"])
        plt.title(f"{name} (top-risk rule)")
        plt.savefig(f"{OUT}/confusion_{name}.png", bbox_inches="tight")
        plt.close()
        print(name, json.dumps(metrics[name], indent=1))

    rf = models["random_forest"]
    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
    imp.plot.barh(title="Random Forest feature importance")
    plt.savefig(f"{OUT}/feature_importance.png", bbox_inches="tight")
    plt.close()

    with open(f"{OUT}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
