import json

import joblib

from ml.features import FEATURES


def test_training_writes_model_and_metrics(trained_model):
    model_path, metrics = trained_model
    bundle = joblib.load(model_path)
    assert bundle["features"] == FEATURES
    saved = json.loads((model_path.parent / "metrics.json").read_text())
    assert saved == metrics


def test_model_finds_the_signal_in_the_data(trained_model):
    _, metrics = trained_model
    # The fake data has a real relationship between risk factors and heart
    # disease, so a working pipeline should rank clearly better than chance.
    assert metrics["logistic_regression"]["roc_auc"] > 0.75
    assert metrics["logistic_regression"]["recall"] > 0.6
    assert set(metrics["random_forest_baseline"]) == {"roc_auc", "recall", "precision"}
