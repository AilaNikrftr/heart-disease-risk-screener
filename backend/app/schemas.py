"""Request and response shapes. The ranges reject obvious typos (a BMI of 250)
without trying to be a medical reference."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ScreeningInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: int = Field(ge=20, le=100, description="Years; the model was trained on adults 20+")
    sex: Literal["male", "female"]
    bmi: float = Field(ge=12, le=80)
    systolic_bp: float = Field(ge=70, le=260, description="mmHg")
    diastolic_bp: float = Field(ge=30, le=160, description="mmHg")
    total_cholesterol: float = Field(ge=70, le=500, description="mg/dL")
    hdl_cholesterol: float = Field(ge=10, le=200, description="mg/dL")
    current_smoker: bool
    diabetes: bool

    @model_validator(mode="after")
    def check_blood_pressure(self) -> "ScreeningInput":
        if self.diastolic_bp >= self.systolic_bp:
            raise ValueError("diastolic blood pressure must be lower than systolic")
        if self.hdl_cholesterol >= self.total_cholesterol:
            raise ValueError("HDL cholesterol must be lower than total cholesterol")
        return self

    def to_features(self) -> dict[str, float]:
        return {
            "age": float(self.age),
            "sex_male": 1.0 if self.sex == "male" else 0.0,
            "bmi": self.bmi,
            "systolic_bp": self.systolic_bp,
            "diastolic_bp": self.diastolic_bp,
            "total_cholesterol": self.total_cholesterol,
            "hdl_cholesterol": self.hdl_cholesterol,
            "current_smoker": float(self.current_smoker),
            "diabetes": float(self.diabetes),
        }


class Factor(BaseModel):
    feature: str
    label: str
    contribution: float


class ScreeningResult(BaseModel):
    id: int
    created_at: datetime
    answers: dict
    score: float
    band: Literal["low", "moderate", "high"]
    flagged: bool
    top_factors: list[Factor]
    disclaimer: str = (
        "Educational screening tool, not a diagnosis. Talk to a doctor about your heart health."
    )


class Health(BaseModel):
    status: str
    model_loaded: bool
    metrics: dict | None = None
