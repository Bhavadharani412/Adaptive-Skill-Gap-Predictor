from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
import numpy as np

from backend.app.database.session import get_db
from backend.app.database.models import (
    DBEmployee, DBEmployeeSkill, DBTrainingModule, DBRecommendationLog
)
from backend.app.schemas.schemas import (
    RecommendationResponse, RecommendationItem, OverrideRequest,
    CompleteModuleRequest, TrainingModuleResponse
)
from backend.app.api.routers.occupations import get_shared_kg
from backend.app.services.recommendation_engine import (
    RecommendationEngine, RecommendationPolicy
)
from backend.app.services.gap_analyzer import SkillGapAnalyzer

router = APIRouter(prefix="/recommendations", tags=["Recommendations & Training"])


@router.get("/modules", response_model=List[TrainingModuleResponse])
def list_training_modules(limit: int = 100, db: Session = Depends(get_db)):
    return db.query(DBTrainingModule).limit(limit).all()


@router.get("/{employee_id}", response_model=RecommendationResponse)
def get_recommendations(
    employee_id: str,
    policy: str = Query("adaptive", pattern="^(random|static|adaptive)$"),
    db: Session = Depends(get_db)
):
    emp = db.query(DBEmployee).filter_by(employee_id=employee_id).first()
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    kg = get_shared_kg()
    analyzer = SkillGapAnalyzer(kg)

    # Load module metadata into engine
    modules_db = db.query(DBTrainingModule).all()
    mod_map = {
        m.target_skill_id: {
            "module_id": m.module_id,
            "title": m.title,
            "duration_hours": m.duration_hours,
            "difficulty_level": m.difficulty_level,
            "learning_cost": m.learning_cost,
            "target_skill_id": m.target_skill_id
        }
        for m in modules_db
    }

    engine = RecommendationEngine(kg, modules_df_or_map=mod_map)
    policy_enum = RecommendationPolicy(policy)

    current_skills = [
        {"skill_id": s.skill_id, "state": s.state, "assessment_score": s.assessment_score}
        for s in emp.skills
    ]
    readiness = analyzer.compute_readiness(emp.occupation_id, current_skills)

    # Get primary recommendation
    mod_info, explanation = engine.recommend_next(
        employee_id=emp.employee_id,
        occupation_id=emp.occupation_id,
        current_skills=current_skills,
        policy=policy_enum
    )

    recommended_item = None
    if mod_info and explanation:
        recommended_item = RecommendationItem(
            module_id=mod_info["module_id"],
            target_skill_id=explanation.skill_id,
            skill_name=explanation.skill_name,
            occupation_id=explanation.occupation_id,
            relevance_weight=explanation.relevance_weight,
            cluster_name=explanation.cluster_name,
            cluster_transfer_bonus=explanation.cluster_transfer_bonus,
            prerequisite_fulfilled=explanation.prerequisite_fulfilled,
            is_generic=explanation.is_generic,
            predicted_criticality=explanation.predicted_criticality,
            composite_score=explanation.composite_score,
            duration_hours=mod_info.get("duration_hours", 10),
            difficulty_level=mod_info.get("difficulty_level", "intermediate"),
            policy_used=explanation.policy_used,
            rationale=explanation.rationale
        )

    # Get alternative recommendations from remaining gaps
    gaps = analyzer.compute_employee_gaps(emp.employee_id, emp.occupation_id, current_skills)
    alt_items = []
    for g in gaps[:5]:
        s_id = g["skill_id"]
        if recommended_item and s_id == recommended_item.target_skill_id:
            continue
        m_meta = mod_map.get(s_id, {
            "module_id": f"MOD-{s_id}",
            "duration_hours": 10,
            "difficulty_level": "intermediate"
        })
        alt_items.append(RecommendationItem(
            module_id=m_meta["module_id"],
            target_skill_id=s_id,
            skill_name=g["skill_name"],
            occupation_id=emp.occupation_id,
            relevance_weight=g["relevance_weight"],
            cluster_name=g["cluster_name"],
            cluster_transfer_bonus=g["cluster_coverage"],
            prerequisite_fulfilled=True,
            is_generic=bool(g["is_generic"]),
            predicted_criticality=1 if g["relevance_weight"] >= 0.75 else 0,
            composite_score=g["relevance_weight"],
            duration_hours=m_meta.get("duration_hours", 10),
            difficulty_level=m_meta.get("difficulty_level", "intermediate"),
            policy_used=policy,
            rationale=f"Alternative eligible module in cluster '{g['cluster_name']}'."
        ))

    return RecommendationResponse(
        employee_id=emp.employee_id,
        policy=policy,
        current_readiness=readiness,
        recommended_module=recommended_item,
        alternative_modules=alt_items
    )


@router.post("/{employee_id}/override")
def override_recommendation(
    employee_id: str,
    req: OverrideRequest,
    db: Session = Depends(get_db)
):
    emp = db.query(DBEmployee).filter_by(employee_id=employee_id).first()
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    mod = db.query(DBTrainingModule).filter_by(module_id=req.selected_module_id).first()
    if not mod:
        raise HTTPException(status_code=404, detail="Training module not found")

    log_entry = DBRecommendationLog(
        employee_id=employee_id,
        module_id=req.selected_module_id,
        target_skill_id=mod.target_skill_id,
        policy_used="human_override",
        composite_score=None,
        rationale=f"Human Manager Override: {req.reason}",
        is_overridden=True,
        override_reason=req.reason
    )
    db.add(log_entry)
    db.commit()

    return {
        "status": "success",
        "message": f"Recommendation successfully overridden to {mod.title}",
        "override_reason": req.reason
    }


@router.post("/{employee_id}/complete-module")
def complete_module_and_assess(
    employee_id: str,
    req: CompleteModuleRequest,
    db: Session = Depends(get_db)
):
    emp = db.query(DBEmployee).filter_by(employee_id=employee_id).first()
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    mod = db.query(DBTrainingModule).filter_by(module_id=req.module_id).first()
    if not mod:
        raise HTTPException(status_code=404, detail="Training module not found")

    target_skill_id = mod.target_skill_id

    # Determine assessment score & pass status
    if req.score is not None:
        score = req.score
        passed = req.manual_passed if req.manual_passed is not None else (score >= mod.assessment_pass_threshold)
    else:
        # Simulate assessment score with high probability of passing
        score = round(float(np.random.uniform(0.78, 0.96)), 2)
        passed = True

    new_state = "mastered" if passed else "in_progress"

    # Update or insert skill in database
    emp_skill = db.query(DBEmployeeSkill).filter_by(
        employee_id=employee_id, skill_id=target_skill_id
    ).first()

    if emp_skill:
        emp_skill.state = new_state
        emp_skill.assessment_score = score
        emp_skill.updated_at = datetime.utcnow()
    else:
        emp_skill = DBEmployeeSkill(
            employee_id=employee_id,
            skill_id=target_skill_id,
            state=new_state,
            assessment_score=score
        )
        db.add(emp_skill)

    # Log completion in recommendation log
    log = DBRecommendationLog(
        employee_id=employee_id,
        module_id=req.module_id,
        target_skill_id=target_skill_id,
        policy_used="completed_assessment",
        composite_score=score,
        rationale=f"Module completed with score {score:.2f} ({'PASSED - Mastered' if passed else 'FAILED - In Progress'})",
        is_overridden=False
    )
    db.add(log)
    db.commit()

    # Recompute updated readiness
    kg = get_shared_kg()
    analyzer = SkillGapAnalyzer(kg)
    updated_skills = [
        {"skill_id": s.skill_id, "state": s.state, "assessment_score": s.assessment_score}
        for s in emp.skills
    ]
    new_readiness = analyzer.compute_readiness(emp.occupation_id, updated_skills)
    remaining_gaps = analyzer.compute_employee_gaps(emp.employee_id, emp.occupation_id, updated_skills)

    return {
        "status": "success",
        "employee_id": employee_id,
        "module_completed": mod.title,
        "assessment_score": score,
        "passed": passed,
        "new_skill_state": new_state,
        "updated_readiness": new_readiness,
        "remaining_gaps_count": len(remaining_gaps)
    }
