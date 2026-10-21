import logging
from pathlib import Path
import pandas as pd
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.database.session import engine, Base, SessionLocal
from backend.app.database.models import (
    DBOccupation, DBSkill, DBOccupationSkillEdge,
    DBEmployee, DBEmployeeSkill, DBTrainingModule
)

logger = logging.getLogger(__name__)


def init_and_seed_database(force_reseed: bool = False):
    """
    Initializes tables and seeds the SQLite database from synthetic or processed data.
    """
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        occ_count = db.query(DBOccupation).count()
        if occ_count > 0 and not force_reseed:
            logger.info("Database already contains %d occupations. Skipping seed.", occ_count)
            return

        data_dir = settings.DATA_SYNTHETIC_DIR
        if not (data_dir / "occupations.csv").exists():
            data_dir = settings.DATA_PROCESSED_DIR

        if not (data_dir / "occupations.csv").exists():
            logger.warning("No data files found to seed database.")
            return

        logger.info("Seeding database from %s ...", data_dir)

        # 1. Occupations
        occ_df = pd.read_csv(data_dir / "occupations.csv")
        for _, r in occ_df.iterrows():
            db.merge(DBOccupation(
                occupation_id=str(r["occupation_id"]),
                title=str(r["title"]),
                description=str(r.get("description", "")),
                primary_cluster_id=int(r.get("primary_cluster_id", 0)),
                primary_cluster_name=str(r.get("primary_cluster_name", "")),
                secondary_cluster_id=int(r.get("secondary_cluster_id", 0)),
                secondary_cluster_name=str(r.get("secondary_cluster_name", "")),
                source=str(r.get("source", "synthetic"))
            ))
        db.commit()

        # 2. Skills
        skill_df = pd.read_csv(data_dir / "skills.csv")
        for _, r in skill_df.iterrows():
            db.merge(DBSkill(
                skill_id=str(r["skill_id"]),
                name=str(r["name"]),
                skill_type=str(r.get("skill_type", "skill/competence")),
                cluster_id=int(r.get("cluster_id", 0)),
                cluster_name=str(r.get("cluster_name", "")),
                is_generic=bool(r.get("is_generic", False)),
                complexity=float(r.get("complexity", 0.5)),
                source=str(r.get("source", "synthetic"))
            ))
        db.commit()

        # 3. Edges
        edge_df = pd.read_csv(data_dir / "occupation_skill_edges.csv")
        for _, r in edge_df.iterrows():
            db.add(DBOccupationSkillEdge(
                occupation_id=str(r["occupation_id"]),
                skill_id=str(r["skill_id"]),
                relation_type=str(r.get("relation_type", "essential")),
                relevance_weight=float(r.get("relevance_weight", 0.7)),
                tier=str(r.get("tier", "domain")),
                source=str(r.get("source", "synthetic"))
            ))
        db.commit()

        # 4. Modules
        mod_file = data_dir / "training_modules.csv"
        if mod_file.exists():
            mod_df = pd.read_csv(mod_file)
            for _, r in mod_df.iterrows():
                db.merge(DBTrainingModule(
                    module_id=str(r["module_id"]),
                    target_skill_id=str(r["target_skill_id"]),
                    title=str(r["title"]),
                    description=str(r.get("description", "")),
                    duration_hours=int(r.get("duration_hours", 10)),
                    difficulty_level=str(r.get("difficulty_level", "intermediate")),
                    learning_cost=float(r.get("learning_cost", 15.0)),
                    assessment_pass_threshold=float(r.get("assessment_pass_threshold", 0.75))
                ))
            db.commit()

        # 5. Employees & Skills
        emp_file = data_dir / "employees.csv"
        emp_skill_file = data_dir / "employee_skills.csv"

        if emp_file.exists() and emp_skill_file.exists():
            emp_df = pd.read_csv(emp_file)
            for _, r in emp_df.iterrows():
                db.merge(DBEmployee(
                    employee_id=str(r["employee_id"]),
                    name=str(r["name"]),
                    email=str(r["email"]),
                    occupation_id=str(r["occupation_id"]),
                    department=str(r.get("department", "Engineering")),
                    hired_date=str(r.get("hired_date", "2026-01-15"))
                ))
            db.commit()

            emp_skills_df = pd.read_csv(emp_skill_file)
            # Batch insert in chunks
            records = []
            for _, r in emp_skills_df.iterrows():
                score = r.get("assessment_score")
                score_val = float(score) if pd.notnull(score) else None
                records.append(DBEmployeeSkill(
                    employee_id=str(r["employee_id"]),
                    skill_id=str(r["skill_id"]),
                    state=str(r.get("state", "not_assessed")),
                    assessment_score=score_val
                ))
                if len(records) >= 1000:
                    db.bulk_save_objects(records)
                    db.commit()
                    records = []
            if records:
                db.bulk_save_objects(records)
                db.commit()

        logger.info("Database seeding successfully completed!")

    except Exception as e:
        db.rollback()
        logger.error("Failed to seed database: %s", e)
        raise e
    finally:
        db.close()


if __name__ == "__main__":
    init_and_seed_database()
