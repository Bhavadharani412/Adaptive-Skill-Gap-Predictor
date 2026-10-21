from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.app.database.session import get_db
from backend.app.database.models import DBOccupation, DBSkill, DBOccupationSkillEdge
from backend.app.schemas.schemas import OccupationResponse, SkillResponse
from backend.app.services.knowledge_graph import OccupationSkillKnowledgeGraph
from backend.app.core.config import settings

router = APIRouter(tags=["Occupations & Skills"])

# Shared in-memory Knowledge Graph cache
_kg_instance: Optional[OccupationSkillKnowledgeGraph] = None


def get_shared_kg() -> OccupationSkillKnowledgeGraph:
    global _kg_instance
    if _kg_instance is None:
        _kg_instance = OccupationSkillKnowledgeGraph()
        data_dir = settings.DATA_SYNTHETIC_DIR
        if not (data_dir / "occupations.csv").exists():
            data_dir = settings.DATA_PROCESSED_DIR
        _kg_instance.build_from_directory(data_dir)
    return _kg_instance


@router.get("/occupations", response_model=List[OccupationResponse])
def list_occupations(db: Session = Depends(get_db)):
    occs = db.query(DBOccupation).all()
    results = []
    for o in occs:
        req_count = db.query(DBOccupationSkillEdge).filter_by(occupation_id=o.occupation_id).count()
        results.append(OccupationResponse(
            occupation_id=o.occupation_id,
            title=o.title,
            description=o.description,
            primary_cluster_id=o.primary_cluster_id,
            primary_cluster_name=o.primary_cluster_name,
            secondary_cluster_id=o.secondary_cluster_id,
            secondary_cluster_name=o.secondary_cluster_name,
            source=o.source,
            required_skills_count=req_count
        ))
    return results


@router.get("/occupations/{occupation_id}", response_model=OccupationResponse)
def get_occupation(occupation_id: str, db: Session = Depends(get_db)):
    occ = db.query(DBOccupation).filter_by(occupation_id=occupation_id).first()
    if not occ:
        raise HTTPException(status_code=404, detail="Occupation not found")
    req_count = db.query(DBOccupationSkillEdge).filter_by(occupation_id=occ.occupation_id).count()
    return OccupationResponse(
        occupation_id=occ.occupation_id,
        title=occ.title,
        description=occ.description,
        primary_cluster_id=occ.primary_cluster_id,
        primary_cluster_name=occ.primary_cluster_name,
        secondary_cluster_id=occ.secondary_cluster_id,
        secondary_cluster_name=occ.secondary_cluster_name,
        source=occ.source,
        required_skills_count=req_count
    )


@router.get("/skills", response_model=List[SkillResponse])
def list_skills(
    cluster_id: Optional[int] = None,
    is_generic: Optional[bool] = None,
    limit: int = 250,
    db: Session = Depends(get_db)
):
    query = db.query(DBSkill)
    if cluster_id is not None:
        query = query.filter_by(cluster_id=cluster_id)
    if is_generic is not None:
        query = query.filter_by(is_generic=is_generic)
    return query.limit(limit).all()


@router.get("/graph/metrics")
def get_graph_metrics():
    kg = get_shared_kg()
    return kg.compute_metrics()


@router.get("/graph/elements")
def get_graph_elements(max_skills: int = Query(60, ge=10, le=220)):
    kg = get_shared_kg()
    return kg.to_cytoscape_elements(max_skills=max_skills)
