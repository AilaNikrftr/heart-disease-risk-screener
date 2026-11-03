"""Shared fixtures. Tests never download NHANES: they build small tables with the
same columns and codes, with heart disease made more likely by age, blood
pressure, smoking and diabetes so a trained model has a real signal to find."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from ml.features import build_dataset
from ml.train import train


def fake_nhanes(n: int = 1500, seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    seqn = np.arange(83732, 83732 + n)
    age = rng.integers(8, 81, n)
    sex = rng.integers(1, 3, n)
    sys_bp = rng.normal(125, 18, n).round()
    smoker_now = rng.choice([1, 2, 3], n, p=[0.15, 0.05, 0.8])
    diabetes = rng.choice([1, 2, 3], n, p=[0.12, 0.83, 0.05])

    logit = -6 + 0.07 * age + 0.02 * (sys_bp - 125) + 0.9 * (smoker_now == 1) + 1.0 * (diabetes == 1)
    chd = rng.random(n) < 1 / (1 + np.exp(-logit))

    def answer(flag):
        codes = np.where(flag, 1, 2).astype(float)
        codes[rng.random(n) < 0.02] = 9  # some "don't know" answers
        return codes

    return {
        "DEMO_I": pd.DataFrame({"SEQN": seqn, "RIDAGEYR": age, "RIAGENDR": sex}),
        "BMX_I": pd.DataFrame({"SEQN": seqn, "BMXBMI": rng.normal(29, 6, n).round(1)}),
        "BPX_I": pd.DataFrame(
            {
                "SEQN": seqn,
                "BPXSY1": sys_bp,
                "BPXSY2": sys_bp + rng.integers(-4, 5, n),
                "BPXSY3": np.nan,
                "BPXDI1": rng.normal(72, 10, n).round(),
                "BPXDI2": 0,  # NHANES records some diastolic readings as 0
                "BPXDI3": np.nan,
            }
        ),
        "TCHOL_I": pd.DataFrame({"SEQN": seqn, "LBXTC": rng.normal(190, 35, n).round()}),
        "HDL_I": pd.DataFrame({"SEQN": seqn, "LBDHDD": rng.normal(53, 14, n).round()}),
        "SMQ_I": pd.DataFrame(
            {
                "SEQN": seqn,
                "SMQ020": np.where(smoker_now == 3, rng.choice([1, 2], n), 1),
                "SMQ040": np.where(smoker_now == 3, 3, smoker_now),
            }
        ),
        "DIQ_I": pd.DataFrame({"SEQN": seqn, "DIQ010": diabetes}),
        "MCQ_I": pd.DataFrame(
            {
                "SEQN": seqn,
                "MCQ160C": answer(chd),
                "MCQ160D": answer(np.zeros(n, bool)),
                "MCQ160E": answer(chd & (rng.random(n) < 0.5)),
            }
        ),
    }


@pytest.fixture(scope="session")
def tables() -> dict[str, pd.DataFrame]:
    return fake_nhanes()


@pytest.fixture(scope="session")
def trained_model(tmp_path_factory, tables):
    out = tmp_path_factory.mktemp("artifacts")
    metrics = train(build_dataset(tables), out_dir=out)
    return out / "model.joblib", metrics


@pytest.fixture
def client(tmp_path, trained_model):
    model_path, _ = trained_model
    app = create_app(model_path=model_path, database_url=f"sqlite:///{tmp_path / 'test.db'}")
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_without_model(tmp_path):
    app = create_app(model_path=tmp_path / "missing.joblib", database_url=f"sqlite:///{tmp_path / 'test.db'}")
    with TestClient(app) as c:
        yield c


@pytest.fixture
def healthy_answers() -> dict:
    return {
        "age": 30,
        "sex": "female",
        "bmi": 23.0,
        "systolic_bp": 112,
        "diastolic_bp": 70,
        "total_cholesterol": 170,
        "hdl_cholesterol": 60,
        "current_smoker": False,
        "diabetes": False,
    }


@pytest.fixture
def risky_answers() -> dict:
    return {
        "age": 74,
        "sex": "male",
        "bmi": 33.0,
        "systolic_bp": 165,
        "diastolic_bp": 92,
        "total_cholesterol": 250,
        "hdl_cholesterol": 35,
        "current_smoker": True,
        "diabetes": True,
    }
