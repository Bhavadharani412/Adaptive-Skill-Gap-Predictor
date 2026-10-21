from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.database.models import DBEmployee, DBEmployeeSkill, DBOccupation, DBSkill
from backend.app.schemas.schemas import (
    EmployeeResponse, EmployeeDetailResponse, EmployeeCreate,
    EmployeeUpdate, EmployeeSkillRecord
)
from backend.app.api.routers.occupations import get_shared_kg
from backend.app.services.gap_analyzer import SkillGapAnalyzer

router = APIRouter(prefix="/employees", tags=["Employees"])


@router.get("", response_model=List[EmployeeResponse])
def list_employees(
    occupation_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    kg = get_shared_kg()
    gap_analyzer = SkillGapAnalyzer(kg)

    query = db.query(DBEmployee)
    if occupation_id:
        query = query.filter_by(occupation_id=occupation_id)

    emps = query.offset(offset).limit(limit).all()
    results = []

    for e in emps:
        skill_recs = [
            {"skill_id": s.skill_id, "state": s.state, "assessment_score": s.assessment_score}
            for s in e.skills
        ]
        readiness = gap_analyzer.compute_readiness(e.occupation_id, skill_recs)
        mastered = sum(1 for s in e.skills if s.state == "mastered")
        gaps = gap_analyzer.compute_employee_gaps(e.employee_id, e.occupation_id, skill_recs)

        results.append(EmployeeResponse(
            employee_id=e.employee_id,
            name=e.name,
            email=e.email,
            occupation_id=e.occupation_id,
            occupation_title=e.occupation.title if e.occupation else e.occupation_id,
            department=e.department,
            hired_date=e.hired_date,
            readiness=readiness,
            mastered_count=mastered,
            total_gaps_count=len(gaps)
        ))

    return results


@router.get("/{employee_id}", response_model=EmployeeDetailResponse)
def get_employee(employee_id: str, db: Session = Depends(get_db)):
    e = db.query(DBEmployee).filter_by(employee_id=employee_id).first()
    if not e:
        raise HTTPException(status_code=404, detail="Employee not found")

    kg = get_shared_kg()
    gap_analyzer = SkillGapAnalyzer(kg)

    skill_recs = [
        {"skill_id": s.skill_id, "state": s.state, "assessment_score": s.assessment_score}
        for s in e.skills
    ]
    readiness = gap_analyzer.compute_readiness(e.occupation_id, skill_recs)
    mastered = sum(1 for s in e.skills if s.state == "mastered")
    gaps = gap_analyzer.compute_employee_gaps(e.employee_id, e.occupation_id, skill_recs)

    # Resolve skill names
    skill_items = []
    for s in e.skills:
        node_data = kg.graph.nodes.get(s.skill_id, {})
        s_name = node_data.get("name", s.skill_id)
        skill_items.append(EmployeeSkillRecord(
            skill_id=s.skill_id,
            skill_name=s_name,
            state=s.state,
            assessment_score=s.assessment_score
        ))

    return EmployeeDetailResponse(
        employee_id=e.employee_id,
        name=e.name,
        email=e.email,
        occupation_id=e.occupation_id,
        occupation_title=e.occupation.title if e.occupation else e.occupation_id,
        department=e.department,
        hired_date=e.hired_date,
        readiness=readiness,
        mastered_count=mastered,
        total_gaps_count=len(gaps),
        skills=skill_items
    )


@router.post("", response_model=EmployeeResponse)
def create_employee(req: EmployeeCreate, db: Session = Depends(get_db)):
    occ = db.query(DBOccupation).filter_by(occupation_id=req.occupation_id).first()
    if not occ:
        raise HTTPException(status_code=400, detail="Invalid occupation_id")

    existing_email = db.query(DBEmployee).filter_by(email=req.email).first()
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already registered")

    emp_count = db.query(DBEmployee).count()
    new_id = f"EMP-{emp_count + 1:04d}"

    emp = DBEmployee(
        employee_id=new_id,
        name=req.name,
        email=req.email,
        occupation_id=req.occupation_id,
        department=req.department or "Engineering",
        hired_date=datetime.utcnow().strftime("%Y-%m-%d")
    )
    db.add(emp)
    db.commit()

    # Initialize skill states from required occupation edges as not_assessed
    kg = get_shared_kg()
    req_skills = kg.get_required_skills(req.occupation_id)
    for s_id in req_skills.keys():
        db.add(DBEmployeeSkill(
            employee_id=new_id,
            skill_id=s_id,
            state="not_assessed",
            assessment_score=None
        ))
    db.commit()

    return EmployeeResponse(
        employee_id=emp.employee_id,
        name=emp.name,
        email=emp.email,
        occupation_id=emp.occupation_id,
        occupation_title=occ.title,
        department=emp.department,
        hired_date=emp.hired_date,
        readiness=0.0,
        mastered_count=0,
        total_gaps_count=len(req_skills)
    )


@router.put("/{employee_id}/skills/{skill_id}")
def update_employee_skill(
    employee_id: str,
    skill_id: str,
    state: str = Query(..., pattern="^(not_assessed|not_mastered|in_progress|mastered)$"),
    score: Optional[float] = Query(None, ge=0.0, le=1.0),
    db: Session = Depends(get_db)
):
    emp = db.query(DBEmployee).filter_by(employee_id=employee_id).first()
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    rec = db.query(DBEmployeeSkill).filter_by(employee_id=employee_id, skill_id=skill_id).first()
    if not rec:
        rec = DBEmployeeSkill(employee_id=employee_id, skill_id=skill_id, state=state, assessment_score=score)
        db.add(rec)
    else:
        rec.state = state
        rec.assessment_score = score
        rec.updated_at = datetime.utcnow()

    db.commit()
    return {"status": "success", "employee_id": employee_id, "skill_id": skill_id, "new_state": state, "score": score}
