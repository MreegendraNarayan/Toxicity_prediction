"""FastAPI REST API for true GNN-based toxicity prediction."""

import io
import logging
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent / "src"))

from toxicity.inference import (  # noqa: E402
    DEFAULT_MODEL_NAME,
    MODEL_SPECS,
    TOX21_TASKS,
    InvalidSmilesError,
    ModelUnavailableError,
    ToxicityPredictor,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Toxicity Prediction API",
    description="RDKit + Torch Geometric GNN API for 12 Tox21 toxicity endpoints",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor_service = ToxicityPredictor()


class PredictRequest(BaseModel):
    """JSON body for single-molecule predictions."""

    smiles: str = Field(..., description="SMILES string representation of the molecule")
    model: str = Field(DEFAULT_MODEL_NAME, description="Enhanced-GCN or Enhanced-GIN")


def timestamp() -> str:
    return datetime.now().isoformat()


def clean_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert pandas row values into JSON-safe primitives."""
    cleaned = {}
    for key, value in row.items():
        if pd.isna(value):
            cleaned[key] = None
        else:
            cleaned[key] = value.item() if hasattr(value, "item") else value
    return cleaned


def prediction_response(prediction: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "success": True,
        **prediction,
        "timestamp": timestamp(),
    }


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Toxicity Prediction API",
        "version": app.version,
        "timestamp": timestamp(),
    }


@app.get("/models")
def list_models() -> Dict[str, Any]:
    """List supported models and checkpoint compatibility status."""
    return {
        "default_model": DEFAULT_MODEL_NAME,
        "supported_models": list(MODEL_SPECS.keys()),
        "models": [asdict(status) for status in predictor_service.available_models()],
        "timestamp": timestamp(),
    }


@app.post("/predict")
async def predict(
    body: Optional[PredictRequest] = Body(default=None),
    smiles: Optional[str] = Query(default=None),
    model: str = Query(DEFAULT_MODEL_NAME),
) -> JSONResponse:
    """Predict toxicity for a single molecule.

    Backward-compatible query parameters are supported:
    `/predict?smiles=...&model=Enhanced-GIN`

    JSON bodies are also supported:
    `{"smiles": "...", "model": "Enhanced-GIN"}`
    """
    selected_smiles = smiles
    selected_model = model

    if body is not None:
        selected_smiles = selected_smiles or body.smiles
        selected_model = selected_model or body.model
        if model == DEFAULT_MODEL_NAME and body.model:
            selected_model = body.model

    if not selected_smiles:
        raise HTTPException(status_code=400, detail="SMILES cannot be empty")

    try:
        prediction = predictor_service.predict_one(selected_smiles, selected_model)
        return JSONResponse(status_code=200, content=prediction_response(prediction))
    except InvalidSmilesError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/batch")
async def batch_predict(
    file: UploadFile = File(...),
    model: str = Query(DEFAULT_MODEL_NAME),
) -> JSONResponse:
    """Batch predict toxicity from a CSV file with a `smiles` column."""
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be CSV format")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV file: {exc}") from exc

    if "smiles" not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must have a 'smiles' column")

    rows = []
    error_details = []

    for idx, row in df.iterrows():
        row_data = clean_row(row.to_dict())
        smiles_value = row_data.get("smiles")
        try:
            prediction = predictor_service.predict_one(str(smiles_value), model)
            prediction_values = prediction["predictions"]
            rows.append(
                {
                    **row_data,
                    "model": prediction["model"],
                    **prediction_values,
                    "average_toxicity": prediction["average_toxicity"],
                    "risk_label": prediction["risk_label"],
                }
            )
        except InvalidSmilesError as exc:
            error_details.append({"row": int(idx), "smiles": smiles_value, "error": str(exc)})
        except ModelUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Batch prediction failed for row %s", idx)
            error_details.append({"row": int(idx), "smiles": smiles_value, "error": str(exc)})

    if not rows:
        raise HTTPException(status_code=400, detail="No valid SMILES strings found in CSV")

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "model": model,
            "processed": len(rows),
            "errors": len(error_details),
            "rows": rows,
            "error_details": error_details,
            "timestamp": timestamp(),
        },
    )


@app.get("/info")
def get_info() -> Dict[str, Any]:
    """Get API information and usage examples."""
    return {
        "name": "Toxicity Prediction API",
        "version": app.version,
        "description": "Graph Neural Network toxicity prediction for the 12 real Tox21 tasks",
        "default_model": DEFAULT_MODEL_NAME,
        "available_models": list(MODEL_SPECS.keys()),
        "toxicity_endpoints": TOX21_TASKS,
        "endpoints": {
            "GET /health": "Health check",
            "GET /models": "List supported models and checkpoint compatibility",
            "POST /predict": "Predict one SMILES string",
            "POST /batch": "Batch predict from a CSV file with a smiles column",
            "GET /info": "API metadata and usage examples",
        },
        "usage_examples": {
            "query_prediction": "/predict?smiles=CC(=O)Oc1ccccc1C(=O)O&model=Enhanced-GIN",
            "json_prediction": {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "model": "Enhanced-GIN"},
            "train_checkpoints": "python train_api_checkpoints.py --models all",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
