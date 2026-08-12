import json
from pathlib import Path
from typing import Literal
 
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
 
ARTIFACT_DIR = Path(__file__).parent
 
app = FastAPI(
    title="Crop Yield Advisory API",
    description="Predicts expected Yield, Fertilizer rate, and Pesticide rate for a given crop, season, farm area, and rainfall.",
    version="1.0.0",
)
 
# ---- Load model + metadata once at startup, not per-request ----
# Supports two artifact shapes so it works with whichever one you have:
#   - yield_model.joblib (from train_model.py)         -> a plain sklearn Pipeline
#   - final_model_from_notebook.joblib (from the notebook) -> a plain dict bundle
# Either way, no custom classes are ever pickled, so this always loads safely.
_MODEL_CANDIDATES = ["yield_model.joblib", "final_model_from_notebook.joblib", "final_model.joblib"]
_artifact = None
for _name in _MODEL_CANDIDATES:
    _path = ARTIFACT_DIR / _name
    if _path.exists():
        _artifact = joblib.load(_path)
        print(f"Loaded model artifact: {_name}")
        break
if _artifact is None:
    raise FileNotFoundError(
        f"No model artifact found in {ARTIFACT_DIR}. Run train_model.py first, "
        f"or copy final_model.joblib here."
    )
 
metadata = json.loads((ARTIFACT_DIR / "model_metadata.json").read_text())
VALID_CROPS = set(metadata["valid_crops"])
VALID_SEASONS = metadata["valid_seasons"]  # e.g. ["Dry Season", "Rainy Season", "Year-Round"]
TARGETS = metadata["targets"]
 
 
def predict_row(row: pd.DataFrame):
    """Handles both artifact shapes uniformly -- always returns a (1, 3) array in TARGETS order."""
    if isinstance(_artifact, dict):
        # dict-bundle format from the notebook (section 8)
        shared_preds = _artifact["shared_model"].predict(row)
        cols = []
        for i, t in enumerate(_artifact["targets"]):
            if _artifact["winners"][t] == "Separate":
                cols.append(_artifact["per_target_models"][t].predict(row))
            else:
                cols.append(shared_preds[:, i])
        import numpy as np
        return np.column_stack(cols)
    else:
        # plain sklearn Pipeline (train_model.py's yield_model.joblib)
        return _artifact.predict(row)
 
 
class FarmInput(BaseModel):
    crop: str = Field(..., description="Crop name, e.g. 'Rice', 'Wheat', 'Sugarcane'")
    season: Literal["Rainy Season", "Dry Season", "Year-Round"] = Field(
        ..., description="Mapped season category"
    )
    area: float = Field(..., gt=0, description="Farm area in hectares")
    annual_rainfall: float = Field(..., ge=0, description="Region's annual rainfall in mm")
 
    class Config:
        json_schema_extra = {
            "example": {
                "crop": "Rice",
                "season": "Rainy Season",
                "area": 50,
                "annual_rainfall": 1200,
            }
        }
 
 
class FarmPrediction(BaseModel):
    predicted_yield_per_hectare: float
    predicted_fertilizer_per_hectare: float
    predicted_pesticide_per_hectare: float
    predicted_total_fertilizer: float
    predicted_total_pesticide: float
    predicted_total_yield: float
 
 
@app.get("/")
def root():
    return {
        "message": "Crop Yield Advisory API",
        "valid_seasons": VALID_SEASONS,
        "num_valid_crops": len(VALID_CROPS),
        "docs": "/docs",
    }
 
 
@app.get("/health")
def health():
    return {"status": "ok"}
 
 
@app.get("/crops")
def list_crops():
    return {"valid_crops": sorted(VALID_CROPS)}
 
 
@app.post("/predict", response_model=FarmPrediction)
def predict(farm: FarmInput):
    if farm.crop not in VALID_CROPS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown crop '{farm.crop}'. See GET /crops for the list this model was trained on.",
        )
 
    row = pd.DataFrame([{
        "Crop": farm.crop,
        "Season_Mapped": farm.season,
        "Area": farm.area,
        "Annual_Rainfall": farm.annual_rainfall,
    }])
 
    yield_pred, fert_pred, pest_pred = predict_row(row)[0]
 
    return FarmPrediction(
        predicted_yield_per_hectare=round(float(yield_pred), 3),
        predicted_fertilizer_per_hectare=round(float(fert_pred), 3),
        predicted_pesticide_per_hectare=round(float(pest_pred), 3),
        predicted_total_yield=round(float(yield_pred) * farm.area, 3),
        predicted_total_fertilizer=round(float(fert_pred) * farm.area, 3),
        predicted_total_pesticide=round(float(pest_pred) * farm.area, 3),
    )