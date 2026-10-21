import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks
from backend.app.core.config import settings
from backend.app.services.data_importer import ESCOImporter
from backend.app.database.init_db import init_and_seed_database

router = APIRouter(prefix="/admin", tags=["Administration & Data Quality"])


@router.get("/quality-report")
def get_data_quality_report():
    report_path = settings.DATA_PROCESSED_DIR / "data_quality_report.json"
    if not report_path.exists():
        # Fallback check
        importer = ESCOImporter()
        report = importer.run_import(force_fallback=True)
        return report

    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


@router.post("/import-dataset")
def import_dataset(force_fallback: bool = True):
    importer = ESCOImporter()
    report = importer.run_import(force_fallback=force_fallback)
    return {
        "status": "success",
        "message": "Dataset imported and validated.",
        "report": report
    }


@router.post("/reseed-database")
def reseed_database():
    try:
        init_and_seed_database(force_reseed=True)
        return {"status": "success", "message": "Database reseeded successfully from active dataset."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
