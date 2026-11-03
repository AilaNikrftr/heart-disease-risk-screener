"""Load the trained model and turn one person's answers into a screening result."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

FEATURE_LABELS = {
    "age": "Age",
    "sex_male": "Sex (male)",
    "bmi": "BMI",
    "systolic_bp": "Systolic blood pressure",
    "diastolic_bp": "Diastolic blood pressure",
    "total_cholesterol": "Total cholesterol",
    "hdl_cholesterol": "HDL cholesterol",
    "current_smoker": "Current smoker",
    "diabetes": "Diabetes",
}


@dataclass
class Prediction:
    score: float
    band: str
    flagged: bool
    top_factors: list[dict]


def band_for(score: float) -> str:
    # The model is trained with balanced class weights, so the score ranks
    # people by risk but is not a calibrated probability. Bands are relative.
    if score < 0.35:
        return "low"
    if score < 0.65:
        return "moderate"
    return "high"


class Predictor:
    def __init__(self, model_path: Path):
        bundle = joblib.load(model_path)
        self.pipeline = bundle["pipeline"]
        self.features: list[str] = bundle["features"]
        self.metrics: dict = bundle.get("metrics", {})

    def predict(self, answers: dict[str, float]) -> Prediction:
        row = pd.DataFrame([[answers[f] for f in self.features]], columns=self.features)
        score = float(self.pipeline.predict_proba(row)[0, 1])
        return Prediction(
            score=round(score, 3),
            band=band_for(score),
            flagged=score >= 0.5,
            top_factors=self._top_factors(row),
        )

    def _top_factors(self, row: pd.DataFrame, k: int = 3) -> list[dict]:
        """Which answers pushed this person's score up the most.

        For logistic regression, each feature adds coefficient * (scaled value)
        to the log-odds, so those products explain the score directly.
        """
        steps = self.pipeline.named_steps
        if "scale" not in steps:
            return []
        scaled = steps["scale"].transform(steps["impute"].transform(row))[0]
        contributions = steps["model"].coef_[0] * scaled
        order = np.argsort(contributions)[::-1]
        return [
            {
                "feature": self.features[i],
                "label": FEATURE_LABELS.get(self.features[i], self.features[i]),
                "contribution": round(float(contributions[i]), 3),
            }
            for i in order[:k]
            if contributions[i] > 0
        ]
