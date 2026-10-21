import logging
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
import joblib
import numpy as np

from backend.app.database.session import get_db
from backend.app.database.models import DBEmployee
from backend.app.schemas.schemas import EmployeeGapsResponse, SkillGapItem
from backend.app.api.routers.occupations import get_shared_kg
from backend.app.services.gap_analyzer import SkillGapAnalyzer, UnassessedPolicy
from backend.app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/gaps", tags=["Skill Gaps & Criticality"])

_classifier_cache = None


def get_trained_classifier():
    global _classifier_cache
    if _classifier_cache is None:
        model_path = settings.MODELS_DIR / "random_forest_improved.joblib"
        if not model_path.exists():
            model_path = settings.MODELS_DIR / "random_forest_paper.joblib"
        if model_path.exists():
            try:
                _classifier_cache = joblib.load(model_path)
            except Exception as e:
                logger.warning("Could not load ML model: %s", e)
    return _classifier_cache


@router.get("/{employee_id}", response_model=EmployeeGapsResponse)
def get_employee_skill_gaps(
    employee_id: str,
    unassessed_policy: str = Query("pessimistic", pattern="^(pessimistic|optimistic|in_progress_as_gap)$"),
    db: Session = Depends(get_db)
):
    emp = db.query(DBEmployee).filter_by(employee_id=employee_id).first()
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    kg = get_shared_kg()
    analyzer = SkillGapAnalyzer(kg)

    skill_recs = [
        {"skill_id": s.skill_id, "state": s.state, "assessment_score": s.assessment_score}
        for s in emp.skills
    ]

    readiness = analyzer.compute_readiness(emp.occupation_id, skill_recs)
    policy_enum = UnassessedPolicy(unassessed_policy)
    gaps = analyzer.compute_employee_gaps(emp.employee_id, emp.occupation_id, skill_recs, unassessed_policy=policy_enum)

    # ML Classifier scoring
    model_obj = get_trained_classifier()
    model = model_obj.get("model") if isinstance(model_obj, dict) else model_obj

    gap_items = []
    for g in gaps:
        # Features: relevance_weight, is_generic, possessed_skills_count, cluster_coverage
        feats = np.array([[
            g["relevance_weight"],
            g["is_generic"],
            g["possessed_skills_count"],
            g["cluster_coverage"]
        ]])

        if model is not None:
            pred_crit = int(model.predict(feats)[0])
            prob_crit = float(model.predict_proba(feats)[0][1]) if hasattr(model, "predict_proba") else 0.5
            crit_source = "RandomForest (Trained ML Classifier)"
        else:
            # Deterministic threshold rule fallback
            pred_crit = 1 if g["relevance_weight"] >= 0.75 else 0
            prob_crit = 0.90 if pred_crit == 1 else 0.20
            crit_source = "Deterministic Relevance Rule (>= 0.75)"

        rationale = (
            f"Gap importance is driven by occupation relevance weight of {g['relevance_weight']:.2f}. "
            f"Employee currently possesses {int(g['cluster_coverage']*100)}% coverage of cluster '{g['cluster_name']}'."
        )

        gap_items.append(SkillGapItem(
            skill_id=g["skill_id"],
            skill_name=g["skill_name"],
            cluster_id=g["cluster_id"],
            cluster_name=g["cluster_name"],
            relevance_weight=g["relevance_weight"],
            is_generic=g["is_generic"],
            possessed_skills_count=g["possessed_skills_count"],
            cluster_coverage=g["cluster_coverage"],
            current_state=g["current_state"],
            current_score=g["current_score"],
            predicted_criticality=pred_crit,
            criticality_probability=round(prob_crit, 3),
            criticality_source=crit_source,
            rationale=rationale
        ))

    req_skills_map = kg.get_required_skills(emp.occupation_id)
    mastered_count = sum(1 for s in emp.skills if s.state == "mastered")

    return EmployeeGapsResponse(
        employee_id=emp.employee_id,
        occupation_id=emp.occupation_id,
        occupation_title=emp.occupation.title if emp.occupation else emp.occupation_id,
        current_readiness=readiness,
        target_readiness=0.80,
        total_required_skills=len(req_skills_map),
        mastered_skills_count=mastered_count,
        gaps_count=len(gap_items),
        gaps=gap_items
    )
