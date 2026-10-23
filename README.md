# ML Model Serving with FastAPI and SQLite Audit Logging

## Overview

A local machine-learning application that serves a trained scikit-learn classification model through FastAPI. The API validates input, returns predictions and class probabilities, and records successful and failed prediction requests in SQLite. A Streamlit interface provides a graphical frontend.

**Purpose:** Educational demonstration of ML model serving, REST APIs, validation, error handling, and audit logging.

## Architecture

- **scikit-learn:** Trains a logistic regression classifier.
- **FastAPI:** Exposes prediction and audit endpoints.
- **Pydantic:** Validates incoming JSON.
- **Uvicorn:** Runs the API server.
- **SQLite:** Stores request outcomes and timestamps.
- **Streamlit:** Provides the user interface.
- **Requests:** Sends HTTP requests from Streamlit to FastAPI.
- **Joblib:** Saves and loads the trained model pipeline.

## Project structure

```text
ml_model_fastapi_logging/
├── models/
│   └── classifier.joblib
├── train_model.py
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── audit.db                 # Generated at runtime
```

## Setup

Requires a compatible Python installation and pip.

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Train the model and create the model artifact:

```bash
python train_model.py
```

Start the FastAPI backend:

```bash
python -m uvicorn main:app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

In a second terminal, activate the same virtual environment and start Streamlit:

```bash
python -m streamlit run app.py
```

Open the local Streamlit URL printed in the terminal, usually http://localhost:8501.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Check that the API is running |
| POST | `/predict` | Validate features and return a prediction |
| GET | `/audit` | Retrieve recent audit records |
| GET | `/docs` | Interactive Swagger documentation |

### Example prediction request

```json
{
  "features": [0.5, -1.2, 0.8, 1.5, -0.3, 2.0]
}
```

Send this JSON to `POST /predict` with the `Content-Type: application/json` header.

### Example successful response

```json
{
  "prediction": 1,
  "probabilities": {
    "0": 0.3356,
    "1": 0.6644
  }
}
```

The exact output depends on the trained model and input features.

### Example validation failure

Submitting fewer than six features should return HTTP `422 Unprocessable Entity`. Non-finite feature values are also rejected.

## Automatic logging and audit trail

The SQLite database, `audit.db`, is created automatically.

The `request_logs` table stores:

- Request ID and UTC timestamp
- Outcome status (`success` or `failed`)
- Prediction and probabilities when available
- Error details for failed requests

Successful predictions and failed validation attempts are recorded. The audit endpoint returns recent records, newest first:

```text
GET http://127.0.0.1:8000/audit
GET http://127.0.0.1:8000/audit?limit=5
```

To inspect the database directly:

```python
import sqlite3

with sqlite3.connect("audit.db") as connection:
    rows = connection.execute(
        """
        SELECT id, timestamp, status, prediction, error
        FROM request_logs
        ORDER BY id DESC
        """
    ).fetchall()

    for row in rows:
        print(row)
```

## Verification

The following behaviors have been tested during development:

- Model training and artifact creation
- FastAPI startup and Swagger access
- Successful prediction with class probabilities
- Invalid feature count rejected with HTTP `422`
- Non-finite input rejected with HTTP `422`
- Successful and failed requests recorded in SQLite

The successful test returned prediction `1`, with probabilities of `0.3356` for class `0` and `0.6644` for class `1`.

The synthetic model achieved 0.675 test accuracy on a held-out set of 200 samples. This is a baseline result, not evidence of suitability for a real-world application.

## Limitations and safety

- The dataset and features are synthetic and have no clinical meaning.
- The model is a demonstration classifier, not a validated decision system.
- Probabilities are model estimates, not guarantees of correctness or calibrated confidence.
- SQLite is intended for local demonstration, not high-concurrency production workloads.
- The API has no authentication, authorization, rate limiting, or production-grade security.
- The audit endpoint is unauthenticated and should not be publicly exposed.
- Error logs can contain validation details; production systems should minimize sensitive data and restrict access.
- Joblib model files must be loaded only from trusted sources.
- Persistent logging, database failure handling, and deployment security require additional production hardening.

**Status:** Core API prediction and SQLite logging paths have been tested. The Streamlit interface was observed to display a previous result after changing inputs.
