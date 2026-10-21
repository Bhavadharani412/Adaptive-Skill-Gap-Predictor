from typing import Dict, List, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import pandas as pd

from backend.app.database.session import get_db
from backend.app.database.models import DBEmployee, DBTrainingModule
from backend.app.schemas.schemas import (
    SimulationEpisodeRequest, SimulationEpisodeResponse,
    CohortSimulationRequest
)
from backend.app.api.routers.occupations import get_shared_kg
from backend.app.services.recommendation_engine import (
    RecommendationEngine, RecommendationPolicy
)
from backend.app.services.policy_evaluator import PolicyEvaluator

router = APIRouter(prefix="/simulator", tags=["Training Simulator"])


@router.post("/run-episode", response_model=SimulationEpisodeResponse)
def simulate_single_episode(
    req: SimulationEpisodeRequest,
    db: Session = Depends(get_db)
):
    emp = db.query(DBEmployee).filter_by(employee_id=req.employee_id).first()
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    kg = get_shared_kg()
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
    evaluator = PolicyEvaluator(
        kg=kg,
        engine=engine,
        target_readiness=req.target_readiness,
        max_budget=req.max_budget
    )

    initial_skills = [
        {"skill_id": s.skill_id, "state": s.state, "assessment_score": s.assessment_score}
        for s in emp.skills
    ]

    policy_enum = RecommendationPolicy(req.policy)
    res = evaluator.simulate_episode(
        employee_id=emp.employee_id,
        occupation_id=emp.occupation_id,
        initial_skills=initial_skills,
        policy=policy_enum,
        seed=req.seed
    )

    return SimulationEpisodeResponse(
        employee_id=res["employee_id"],
        policy=res["policy"],
        initial_readiness=res["initial_readiness"],
        final_readiness=res["final_readiness"],
        target_readiness=res["target_readiness"],
        steps_to_target=res["steps_to_target"],
        reached_target=res["reached_target"],
        censored=res["censored"],
        total_learning_hours=res["total_learning_hours"],
        history=res["history"]
    )


@router.post("/compare-cohort")
def compare_policies_cohort(
    req: CohortSimulationRequest,
    db: Session = Depends(get_db)
):
    kg = get_shared_kg()
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
    evaluator = PolicyEvaluator(
        kg=kg,
        engine=engine,
        target_readiness=req.target_readiness,
        max_budget=req.max_budget
    )

    # Fetch sample employees from DB
    emps = db.query(DBEmployee).limit(req.cohort_size).all()
    if not emps:
        raise HTTPException(status_code=400, detail="No employees found in database")

    emp_records = [{"employee_id": e.employee_id, "occupation_id": e.occupation_id} for e in emps]
    emp_df = pd.DataFrame(emp_records)

    skill_records = []
    for e in emps:
        for s in e.skills:
            skill_records.append({
                "employee_id": e.employee_id,
                "skill_id": s.skill_id,
                "state": s.state,
                "assessment_score": s.assessment_score
            })
    emp_skills_df = pd.DataFrame(skill_records)

    results = evaluator.evaluate_cohort(
        employees_df=emp_df,
        emp_skills_df=emp_skills_df,
        sample_size=req.cohort_size,
        random_seed=req.seed
    )

    return results
