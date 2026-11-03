"""FastAPI app: score a screening, store it in SQLite, list and delete past ones."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import Screening, make_sessionmaker
from app.predictor import Predictor
from app.schemas import Health, ScreeningInput, ScreeningResult

BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = BACKEND_DIR / "ml" / "artifacts" / "model.joblib"


def to_result(row: Screening) -> ScreeningResult:
    return ScreeningResult(
        id=row.id,
        created_at=row.created_at,
        answers=row.answers,
        score=row.score,
        band=row.band,
        flagged=row.score >= 0.5,
        top_factors=row.top_factors,
    )


def create_app(model_path: Path | None = None, database_url: str | None = None) -> FastAPI:
    model_path = Path(model_path or os.getenv("MODEL_PATH", DEFAULT_MODEL))
    database_url = database_url or os.getenv("DATABASE_URL", "sqlite:///./screenings.db")

    app = FastAPI(title="Heart Disease Risk Screener", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )

    # A missing model shouldn't stop the API from starting; /api/health reports it.
    app.state.predictor = Predictor(model_path) if model_path.exists() else None
    _, SessionLocal = make_sessionmaker(database_url)

    def get_db() -> Iterator[Session]:
        with SessionLocal() as session:
            yield session

    def get_predictor(request: Request) -> Predictor:
        predictor = request.app.state.predictor
        if predictor is None:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "Model not trained yet. Run `python -m ml.fetch_data && python -m ml.train`.",
            )
        return predictor

    @app.get("/api/health", response_model=Health)
    def health(request: Request) -> Health:
        predictor = request.app.state.predictor
        return Health(
            status="ok",
            model_loaded=predictor is not None,
            metrics=predictor.metrics if predictor else None,
        )

    @app.post("/api/screenings", response_model=ScreeningResult, status_code=status.HTTP_201_CREATED)
    def create_screening(
        body: ScreeningInput,
        db: Session = Depends(get_db),
        predictor: Predictor = Depends(get_predictor),
    ) -> ScreeningResult:
        prediction = predictor.predict(body.to_features())
        row = Screening(
            answers=body.model_dump(),
            score=prediction.score,
            band=prediction.band,
            top_factors=prediction.top_factors,
        )
        db.add(row)
        db.commit()
        return to_result(row)

    @app.get("/api/screenings", response_model=list[ScreeningResult])
    def list_screenings(
        limit: int = Query(20, ge=1, le=100), db: Session = Depends(get_db)
    ) -> list[ScreeningResult]:
        rows = db.scalars(
            select(Screening).order_by(Screening.created_at.desc(), Screening.id.desc()).limit(limit)
        )
        return [to_result(r) for r in rows]

    @app.delete("/api/screenings/{screening_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_screening(screening_id: int, db: Session = Depends(get_db)) -> None:
        row = db.get(Screening, screening_id)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Screening not found")
        db.delete(row)
        db.commit()

    return app
