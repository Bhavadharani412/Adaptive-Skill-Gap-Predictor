from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


# --- Health & General ---
class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str = "1.0.0"
    timestamp: str


# --- Occupation & Skill ---
class OccupationResponse(BaseModel):
    occupation_id: str
    title: str
    description: Optional[str] = None
    primary_cluster_id: int
    primary_cluster_name: Optional[str] = None
    secondary_cluster_id: int
    secondary_cluster_name: Optional[str] = None
    source: str
    required_skills_count: Optional[int] = 0

    model_config = {"from_attributes": True}


class SkillResponse(BaseModel):
    skill_id: str
    name: str
    skill_type: str
    cluster_id: int
    cluster_name: Optional[str] = None
    is_generic: bool
    complexity: float
    source: str

    model_config = {"from_attributes": True}


# --- Employee ---
class EmployeeSkillRecord(BaseModel):
    skill_id: str
    skill_name: Optional[str] = None
    state: str = "not_assessed"  # not_assessed, not_mastered, in_progress, mastered
    assessment_score: Optional[float] = None


class EmployeeCreate(BaseModel):
    name: str
    email: str
    occupation_id: str
    department: Optional[str] = "Engineering"


class EmployeeUpdate(BaseModel):
    name: Optional[str] = None
    occupation_id: Optional[str] = None
    department: Optional[str] = None


class EmployeeResponse(BaseModel):
    employee_id: str
    name: str
    email: str
    occupation_id: str
    occupation_title: Optional[str] = None
    department: str
    hired_date: str
    readiness: Optional[float] = 0.0
    mastered_count: Optional[int] = 0
    total_gaps_count: Optional[int] = 0

    model_config = {"from_attributes": True}


class EmployeeDetailResponse(EmployeeResponse):
    skills: List[EmployeeSkillRecord] = []


# --- Skill Gap & Criticality ---
class SkillGapItem(BaseModel):
    skill_id: str
    skill_name: str
    cluster_id: int
    cluster_name: str
    relevance_weight: float
    is_generic: int
    possessed_skills_count: int
    cluster_coverage: float
    current_state: str
    current_score: Optional[float] = None
    predicted_criticality: int = 0
    criticality_probability: float = 0.0
    criticality_source: str = "model_prediction"
    rationale: str = ""


class EmployeeGapsResponse(BaseModel):
    employee_id: str
    occupation_id: str
    occupation_title: str
    current_readiness: float
    target_readiness: float = 0.80
    total_required_skills: int
    mastered_skills_count: int
    gaps_count: int
    gaps: List[SkillGapItem]


# --- Training & Recommendations ---
class TrainingModuleResponse(BaseModel):
    module_id: str
    target_skill_id: str
    title: str
    description: Optional[str] = None
    duration_hours: int
    difficulty_level: str
    learning_cost: float
    assessment_pass_threshold: float

    model_config = {"from_attributes": True}


class RecommendationItem(BaseModel):
    module_id: str
    target_skill_id: str
    skill_name: str
    occupation_id: str
    relevance_weight: float
    cluster_name: str
    cluster_transfer_bonus: float
    prerequisite_fulfilled: bool
    is_generic: bool
    predicted_criticality: Optional[int] = None
    composite_score: float
    duration_hours: int
    difficulty_level: str
    policy_used: str
    rationale: str


class RecommendationResponse(BaseModel):
    employee_id: str
    policy: str
    current_readiness: float
    recommended_module: Optional[RecommendationItem] = None
    alternative_modules: List[RecommendationItem] = []


class OverrideRequest(BaseModel):
    selected_module_id: str
    reason: str


class CompleteModuleRequest(BaseModel):
    module_id: str
    score: Optional[float] = None
    manual_passed: Optional[bool] = None


# --- Simulator ---
class SimulationEpisodeRequest(BaseModel):
    employee_id: str
    policy: str = "adaptive"  # random, static, adaptive
    target_readiness: float = 0.80
    max_budget: int = 50
    seed: int = 42


class SimulationStepLog(BaseModel):
    step: int
    module_id: str
    target_skill_id: str
    passed: bool
    score: float
    readiness_after_step: float
    rationale: str


class SimulationEpisodeResponse(BaseModel):
    employee_id: str
    policy: str
    initial_readiness: float
    final_readiness: float
    target_readiness: float
    steps_to_target: int
    reached_target: bool
    censored: bool
    total_learning_hours: float
    history: List[SimulationStepLog]


class CohortSimulationRequest(BaseModel):
    cohort_size: int = 50
    target_readiness: float = 0.80
    max_budget: int = 50
    seed: int = 42


# --- Admin & Data ---
class DataQualityResponse(BaseModel):
    source: str
    imported_at: str
    status: str
    warnings: List[str] = []
    errors: List[str] = []
    metrics: Dict[str, Any] = {}
