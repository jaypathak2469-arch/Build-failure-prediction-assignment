"""Train and evaluate build-failure models with a time-based split."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, f1_score,  # noqa: E402
                             precision_score, recall_score, roc_auc_score)
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
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut], df.iloc[cut:]


def main():
    os.makedirs(OUT, exist_ok=True)
    df = build_features(load_data())
    train, test = time_split(df)
    X_tr, y_tr = train[FEATURES], train["failed"]
    X_te, y_te = test[FEATURES], test["failed"]
    print(f"Train {len(train)} / Test {len(test)} | failure rate {df['failed'].mean():.2%}")

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", min_samples_leaf=5, random_state=42),
    }
    metrics = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred, proba = model.predict(X_te), model.predict_proba(X_te)[:, 1]
        metrics[name] = {
            "precision": precision_score(y_te, pred, zero_division=0),
            "recall": recall_score(y_te, pred, zero_division=0),
            "f1": f1_score(y_te, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_te, proba),
        }
        ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels=["pass", "fail"])
        plt.title(name)
        plt.savefig(f"{OUT}/confusion_{name}.png", bbox_inches="tight")
        plt.close()
        print(name, {k: round(v, 3) for k, v in metrics[name].items()})

    rf = models["random_forest"]
    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
    imp.plot.barh(title="Random Forest feature importance")
    plt.savefig(f"{OUT}/feature_importance.png", bbox_inches="tight")
    plt.close()

    with open(f"{OUT}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
