import pytest
from pathlib import Path
import pandas as pd
from backend.app.services.data_importer import ESCOImporter


def test_synthetic_fallback_generation(tmp_path):
    importer = ESCOImporter(raw_dir=tmp_path / "raw", processed_dir=tmp_path / "processed")
    report = importer.run_import(force_fallback=True)

    assert report["status"] == "completed"
    assert report["source"] == "synthetic_fallback"
    assert report["metrics"]["total_occupations"] == 10
    assert report["metrics"]["total_skills"] == 60
    assert report["metrics"]["total_occupation_skill_edges"] > 0
    assert (tmp_path / "processed" / "occupations.csv").exists()
    assert (tmp_path / "processed" / "skills.csv").exists()
    assert (tmp_path / "processed" / "occupation_skill_edges.csv").exists()
    assert (tmp_path / "processed" / "data_quality_report.json").exists()


def test_esco_importer_column_resolution(tmp_path):
    importer = ESCOImporter(processed_dir=tmp_path)
    df = pd.DataFrame({"conceptUri": ["http://data.europa.eu/esco/1"], "preferredLabel": ["Software Engineer"]})
    col = importer._resolve_column(df, ["conceptUri", "uri", "id"])
    assert col == "conceptUri"
