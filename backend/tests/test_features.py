import numpy as np
import pandas as pd
import pytest

from ml.features import FEATURES, TARGET, build_dataset


def one_person(**overrides) -> dict[str, pd.DataFrame]:
    """NHANES tables for a single participant, with chosen fields overridden."""
    row = {
        "RIDAGEYR": 50, "RIAGENDR": 1, "BMXBMI": 27.5,
        "BPXSY1": 120, "BPXSY2": 124, "BPXSY3": np.nan,
        "BPXDI1": 80, "BPXDI2": 0, "BPXDI3": 76,
        "LBXTC": 200, "LBDHDD": 50, "SMQ020": 2, "SMQ040": np.nan, "DIQ010": 2,
        "MCQ160C": 2, "MCQ160D": 2, "MCQ160E": 2,
    }
    row.update(overrides)
    table = lambda *cols: pd.DataFrame([{"SEQN": 1, **{c: row[c] for c in cols}}])
    return {
        "DEMO_I": table("RIDAGEYR", "RIAGENDR"),
        "BMX_I": table("BMXBMI"),
        "BPX_I": table("BPXSY1", "BPXSY2", "BPXSY3", "BPXDI1", "BPXDI2", "BPXDI3"),
        "TCHOL_I": table("LBXTC"),
        "HDL_I": table("LBDHDD"),
        "SMQ_I": table("SMQ020", "SMQ040"),
        "DIQ_I": table("DIQ010"),
        "MCQ_I": table("MCQ160C", "MCQ160D", "MCQ160E"),
    }


def test_columns_and_basic_recoding():
    df = build_dataset(one_person())
    assert list(df.columns) == FEATURES + [TARGET]
    row = df.iloc[0]
    assert row["sex_male"] == 1
    assert row["systolic_bp"] == 122  # mean of the two readings taken
    assert row["diastolic_bp"] == 78  # the 0 reading is ignored
    assert row["current_smoker"] == 0  # never smoked 100 cigarettes
    assert row[TARGET] == 0


@pytest.mark.parametrize("smq020,smq040,expected", [(1, 1, 1), (1, 2, 1), (1, 3, 0), (2, np.nan, 0)])
def test_current_smoker(smq020, smq040, expected):
    assert build_dataset(one_person(SMQ020=smq020, SMQ040=smq040)).iloc[0]["current_smoker"] == expected


@pytest.mark.parametrize("code,expected", [(1, 1.0), (2, 0.0), (3, 0.0)])
def test_diabetes_borderline_counts_as_no(code, expected):
    assert build_dataset(one_person(DIQ010=code)).iloc[0]["diabetes"] == expected


@pytest.mark.parametrize("question", ["MCQ160C", "MCQ160D", "MCQ160E"])
def test_any_heart_condition_is_positive(question):
    assert build_dataset(one_person(**{question: 1})).iloc[0][TARGET] == 1


@pytest.mark.parametrize("code", [7, 9, np.nan])
def test_unlabelled_rows_are_dropped(code):
    assert build_dataset(one_person(MCQ160C=code)).empty


def test_children_are_excluded():
    assert build_dataset(one_person(RIDAGEYR=15)).empty


def test_missing_lab_values_stay_missing_for_the_imputer():
    assert np.isnan(build_dataset(one_person(LBXTC=np.nan)).iloc[0]["total_cholesterol"])


def test_missing_table_is_reported():
    tables = one_person()
    del tables["HDL_I"]
    with pytest.raises(ValueError, match="HDL_I"):
        build_dataset(tables)
