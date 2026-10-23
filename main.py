from pathlib import Path
import logging
import sqlite3
from datetime import datetime, timezone
import json
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi import Query
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "classifier.joblib"
DB_PATH = BASE_DIR / "audit.db"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Prediction API",
    description="A demonstration API with SQLite audit logging.",
    version="1.0.0",
)

model = joblib.load(MODEL_PATH)


class PredictionInput(BaseModel):
    features: list[float] = Field(
        min_length=6,
        max_length=6,
        description="Exactly six numeric model features.",
    )


def initialize_database():
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS request_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                prediction TEXT,
                probabilities TEXT,
                error TEXT
            )
            """
        )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
):
    error_details = json.dumps(exc.errors(), default=str)

    log_request(
        status="failed",
        error=f"Request validation failed: {error_details}",
    )

    return JSONResponse(
        status_code=422,
        content={"detail": json.loads(error_details)},
    )

def log_request(
    status: str,
    prediction=None,
    probabilities=None,
    error=None,
):
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            INSERT INTO request_logs
            (timestamp, status, prediction, probabilities, error)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                status,
                str(prediction) if prediction is not None else None,
                json.dumps(probabilities) if probabilities is not None else None,
                error,
            ),
        )


initialize_database()

@app.get("/")
def home():
    return {"message": "ML Prediction API is running"}

@app.post("/predict")
def predict(data: PredictionInput):
    try:
        features = np.asarray(data.features, dtype=float).reshape(1, -1)

        if not np.isfinite(features).all():
            raise ValueError("All features must be finite numbers.")

        prediction = int(model.predict(features)[0])
        probabilities_array = model.predict_proba(features)[0]

        probabilities = {
            str(label): round(float(probability), 6)
            for label, probability in zip(
                model.classes_, probabilities_array
            )
        }

        log_request(
            status="success",
            prediction=prediction,
            probabilities=probabilities,
        )

        return {
            "prediction": prediction,
            "probabilities": probabilities,
        }

    except ValueError as exc:
        log_request(status="failed", error=str(exc))
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except Exception as exc:
        logger.exception("Prediction request failed")
        log_request(status="failed", error="Internal prediction error")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal error.",
        ) from exc

@app.get("/audit")
def get_audit_logs(limit: int = Query(default=20, ge=1, le=100)):
    try:
        with sqlite3.connect(DB_PATH) as connection:
            connection.row_factory = sqlite3.Row

            rows = connection.execute(
                """
                SELECT id, timestamp, status, prediction,
                probabilities, error
                FROM request_logs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        records = []

        for row in rows:
            record = dict(row)

            # Parse valid JSON while retaining older records
            # written using Python dictionary formatting.
            if record["probabilities"]:
                try:
                    record["probabilities"] = json.loads(
                        record["probabilities"]
                    )
                except (json.JSONDecodeError, TypeError):
                    pass

            records.append(record)

        return {
            "count": len(records),
            "records": records,
        }

    except sqlite3.Error:
        logger.exception("Unable to retrieve audit logs")
        raise HTTPException(
            status_code=500,
            detail="Unable to retrieve audit logs.",
        )
