"""Turn raw NHANES 2015-2016 tables into one row per adult with model features.

NHANES ships each topic as its own table keyed by SEQN (the participant id).
This module joins the tables we need and recodes the survey answers into
plain numbers, so the rest of the code never sees NHANES codes like 7 or 9.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Order matters: the model is trained and served with columns in this order.
FEATURES: list[str] = [
    "age",
    "sex_male",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "total_cholesterol",
    "hdl_cholesterol",
    "current_smoker",
    "diabetes",
]
TARGET = "chd"

# NHANES table name -> columns we read from it.
TABLES: dict[str, list[str]] = {
    "DEMO_I": ["SEQN", "RIDAGEYR", "RIAGENDR"],
    "BMX_I": ["SEQN", "BMXBMI"],
    "BPX_I": ["SEQN", "BPXSY1", "BPXSY2", "BPXSY3", "BPXDI1", "BPXDI2", "BPXDI3"],
    "TCHOL_I": ["SEQN", "LBXTC"],
    "HDL_I": ["SEQN", "LBDHDD"],
    "SMQ_I": ["SEQN", "SMQ020", "SMQ040"],
    "DIQ_I": ["SEQN", "DIQ010"],
    "MCQ_I": ["SEQN", "MCQ160C", "MCQ160D", "MCQ160E"],
}

YES, NO = 1, 2  # NHANES questionnaire codes; 7 = refused, 9 = don't know


def _yes_no(series: pd.Series) -> pd.Series:
    """Map NHANES yes/no answers to 1/0 and everything else to NaN."""
    return series.map({YES: 1.0, NO: 0.0})


def _mean_reading(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Average the repeated blood pressure readings, ignoring missing ones.

    A diastolic reading of 0 is a recording artifact in NHANES, not a real value.
    """
    readings = df[cols].replace(0, np.nan)
    return readings.mean(axis=1, skipna=True)


def build_dataset(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join the NHANES tables and return adults (20+) with FEATURES and TARGET.

    Rows whose heart disease answers are missing, refused or "don't know"
    are dropped, because we can't label them. Missing feature values are
    kept as NaN; the training pipeline imputes them.
    """
    missing = set(TABLES) - set(tables)
    if missing:
        raise ValueError(f"missing NHANES tables: {sorted(missing)}")

    df = tables["DEMO_I"][TABLES["DEMO_I"]]
    for name, cols in TABLES.items():
        if name != "DEMO_I":
            df = df.merge(tables[name][cols], on="SEQN", how="left")

    df = df[df["RIDAGEYR"] >= 20]  # the heart disease questions are only asked of adults

    out = pd.DataFrame(index=df.index)
    out["age"] = df["RIDAGEYR"].astype(float)
    out["sex_male"] = (df["RIAGENDR"] == 1).astype(float)
    out["bmi"] = df["BMXBMI"]
    out["systolic_bp"] = _mean_reading(df, ["BPXSY1", "BPXSY2", "BPXSY3"])
    out["diastolic_bp"] = _mean_reading(df, ["BPXDI1", "BPXDI2", "BPXDI3"])
    out["total_cholesterol"] = df["LBXTC"]
    out["hdl_cholesterol"] = df["LBDHDD"]

    # Never smoked 100 cigarettes -> not a current smoker. Otherwise SMQ040
    # says whether they smoke now (1 every day, 2 some days, 3 not at all).
    smoker = pd.Series(np.nan, index=df.index)
    smoker[df["SMQ020"] == NO] = 0.0
    smoker[df["SMQ040"].isin([1, 2])] = 1.0
    smoker[df["SMQ040"] == 3] = 0.0
    out["current_smoker"] = smoker

    # DIQ010: 1 yes, 2 no, 3 borderline. Borderline counts as no diagnosis.
    out["diabetes"] = df["DIQ010"].map({1: 1.0, 2: 0.0, 3: 0.0})

    # Label: ever told they had coronary heart disease, angina, or a heart attack.
    answers = pd.concat(
        [_yes_no(df[c]) for c in ["MCQ160C", "MCQ160D", "MCQ160E"]], axis=1
    )
    labelled = answers.notna().all(axis=1)
    out[TARGET] = answers.max(axis=1)
    out = out[labelled]
    out[TARGET] = out[TARGET].astype(int)

    return out[FEATURES + [TARGET]].reset_index(drop=True)
