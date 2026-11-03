"""Train the screening model on NHANES 2015-2016 and save it with its metrics.

Usage:  python -m ml.fetch_data && python -m ml.train

Writes ml/artifacts/model.joblib and ml/artifacts/metrics.json.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.features import FEATURES, TABLES, TARGET, build_dataset

ML_DIR = Path(__file__).parent
DATA_DIR = ML_DIR / "data"
ARTIFACT_DIR = ML_DIR / "artifacts"
SEED = 42


def load_tables(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    tables = {}
    for name in TABLES:
        path = data_dir / f"{name}.xpt"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run `python -m ml.fetch_data` first")
        tables[name] = pd.read_sas(path, format="xport")
    return tables


def make_logistic() -> Pipeline:
    # class_weight="balanced": only ~8% of adults report heart disease, and a
    # screener that misses most of them is useless, so we trade precision for recall.
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )


def make_forest() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300, min_samples_leaf=20, class_weight="balanced", random_state=SEED
                ),
            ),
        ]
    )


def evaluate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    scores = pipeline.predict_proba(X)[:, 1]
    predicted = (scores >= 0.5).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y, scores)), 3),
        "recall": round(float(recall_score(y, predicted)), 3),
        "precision": round(float(precision_score(y, predicted, zero_division=0)), 3),
    }


def train(dataset: pd.DataFrame, out_dir: Path = ARTIFACT_DIR) -> dict:
    """Train on 80% of the data, report metrics on the held-out 20%, save the model.

    The served model is logistic regression, because its per-feature
    contributions can be shown to the user. The random forest is trained only
    as a baseline to check we aren't leaving much accuracy on the table.
    """
    X, y = dataset[FEATURES], dataset[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    logistic = make_logistic().fit(X_train, y_train)
    forest = make_forest().fit(X_train, y_train)

    metrics = {
        "dataset": "NHANES 2015-2016, adults 20+",
        "rows": int(len(dataset)),
        "positive_rate": round(float(y.mean()), 3),
        "test_rows": int(len(y_test)),
        "served_model": "logistic_regression",
        "logistic_regression": evaluate(logistic, X_test, y_test),
        "random_forest_baseline": evaluate(forest, X_test, y_test),
        "features": FEATURES,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    # Refit the served model on all labelled rows before saving.
    final = make_logistic().fit(X, y)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final, "features": FEATURES, "metrics": metrics}, out_dir / "model.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> int:
    dataset = build_dataset(load_tables())
    metrics = train(dataset)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
