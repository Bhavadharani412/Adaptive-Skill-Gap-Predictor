from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship

from backend.app.database.session import Base


class DBOccupation(Base):
    __tablename__ = "occupations"

    occupation_id = Column(String(50), primary_key=True, index=True)
    title = Column(String(200), nullable=False, index=True)
    description = Column(Text, nullable=True)
    primary_cluster_id = Column(Integer, default=0)
    primary_cluster_name = Column(String(100), nullable=True)
    secondary_cluster_id = Column(Integer, default=0)
    secondary_cluster_name = Column(String(100), nullable=True)
    source = Column(String(50), default="synthetic")

    employees = relationship("DBEmployee", back_populates="occupation")
    required_skills = relationship("DBOccupationSkillEdge", back_populates="occupation")


class DBSkill(Base):
    __tablename__ = "skills"

    skill_id = Column(String(50), primary_key=True, index=True)
    name = Column(String(200), nullable=False, index=True)
    skill_type = Column(String(50), default="skill/competence")
    cluster_id = Column(Integer, default=0, index=True)
    cluster_name = Column(String(100), nullable=True)
    is_generic = Column(Boolean, default=False)
    complexity = Column(Float, default=0.5)
    source = Column(String(50), default="synthetic")

    training_module = relationship("DBTrainingModule", back_populates="target_skill", uselist=False)


class DBOccupationSkillEdge(Base):
    __tablename__ = "occupation_skill_edges"

    id = Column(Integer, primary_key=True, autoincrement=True)
    occupation_id = Column(String(50), ForeignKey("occupations.occupation_id"), index=True, nullable=False)
    skill_id = Column(String(50), ForeignKey("skills.skill_id"), index=True, nullable=False)
    relation_type = Column(String(50), default="essential")
    relevance_weight = Column(Float, nullable=False)
    tier = Column(String(50), default="primary_domain")
    source = Column(String(50), default="synthetic")

    occupation = relationship("DBOccupation", back_populates="required_skills")


class DBEmployee(Base):
    __tablename__ = "employees"

    employee_id = Column(String(50), primary_key=True, index=True)
    name = Column(String(150), nullable=False)
    email = Column(String(150), unique=True, index=True)
    occupation_id = Column(String(50), ForeignKey("occupations.occupation_id"), nullable=False)
    department = Column(String(100), default="Engineering")
    hired_date = Column(String(50), default="2026-01-15")
    created_at = Column(DateTime, default=datetime.utcnow)

    occupation = relationship("DBOccupation", back_populates="employees")
    skills = relationship("DBEmployeeSkill", back_populates="employee", cascade="all, delete-orphan")
    recommendations = relationship("DBRecommendationLog", back_populates="employee", cascade="all, delete-orphan")


class DBEmployeeSkill(Base):
    __tablename__ = "employee_skills"

    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(String(50), ForeignKey("employees.employee_id"), index=True, nullable=False)
    skill_id = Column(String(50), ForeignKey("skills.skill_id"), index=True, nullable=False)
    state = Column(String(30), default="not_assessed")  # not_assessed, not_mastered, in_progress, mastered
    assessment_score = Column(Float, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    employee = relationship("DBEmployee", back_populates="skills")


class DBTrainingModule(Base):
    __tablename__ = "training_modules"

    module_id = Column(String(50), primary_key=True, index=True)
    target_skill_id = Column(String(50), ForeignKey("skills.skill_id"), unique=True, nullable=False)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    duration_hours = Column(Integer, default=10)
    difficulty_level = Column(String(30), default="intermediate")
    learning_cost = Column(Float, default=15.0)
    assessment_pass_threshold = Column(Float, default=0.75)

    target_skill = relationship("DBSkill", back_populates="training_module")


class DBRecommendationLog(Base):
    __tablename__ = "recommendation_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(String(50), ForeignKey("employees.employee_id"), index=True, nullable=False)
    module_id = Column(String(50), ForeignKey("training_modules.module_id"), nullable=False)
    target_skill_id = Column(String(50), nullable=False)
    policy_used = Column(String(50), default="adaptive")
    composite_score = Column(Float, nullable=True)
    rationale = Column(Text, nullable=True)
    is_overridden = Column(Boolean, default=False)
    override_reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    employee = relationship("DBEmployee", back_populates="recommendations")
