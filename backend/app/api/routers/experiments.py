import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.app.core.config import settings

router = APIRouter(prefix="/experiments", tags=["Research Experiments & Metrics"])


@router.get("/classifier")
def get_classifier_metrics():
    path = settings.EXPERIMENTS_DIR / "results" / "classifier_results.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Classifier results not found. Run training first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/policies")
def get_policy_metrics():
    path = settings.EXPERIMENTS_DIR / "results" / "policy_evaluation_results.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Policy evaluation results not found. Run evaluation first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/ablations")
def get_ablation_metrics():
    path = settings.EXPERIMENTS_DIR / "results" / "ablation_results.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Ablation results not found. Run ablations first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/export-csv")
def export_results_csv():
    csv_path = settings.EXPERIMENTS_DIR / "results" / "ablation_summary.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Ablation CSV not found.")
    return FileResponse(csv_path, media_type="text/csv", filename="asgp_ablation_summary.csv")
