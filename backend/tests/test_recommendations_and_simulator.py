import pytest
import pandas as pd
from backend.app.core.config import settings
from backend.app.services.knowledge_graph import OccupationSkillKnowledgeGraph
from backend.app.services.recommendation_engine import (
    RecommendationEngine, RecommendationPolicy
)
from backend.app.services.policy_evaluator import PolicyEvaluator


@pytest.fixture(scope="module")
def setup_services():
    kg = OccupationSkillKnowledgeGraph()
    kg.build_from_directory(settings.DATA_SYNTHETIC_DIR)
    modules_df = pd.read_csv(settings.DATA_SYNTHETIC_DIR / "training_modules.csv")
    engine = RecommendationEngine(kg, modules_df=modules_df)
    evaluator = PolicyEvaluator(kg, engine, target_readiness=0.80, max_budget=20)
    return kg, engine, evaluator


def test_policies_recommend_modules(setup_services):
    kg, engine, evaluator = setup_services
    reqs = kg.get_required_skills("OCC-001")
    req_list = list(reqs.keys())

    # Employee has 3 mastered skills, rest missing
    current_skills = [{"skill_id": s, "state": "mastered"} for s in req_list[:3]]

    # 1. Random
    mod_a, exp_a = engine.recommend_next(
        "EMP-TEST", "OCC-001", current_skills, policy=RecommendationPolicy.RANDOM
    )
    assert mod_a is not None
    assert exp_a.policy_used == "random"

    # 2. Static
    mod_b, exp_b = engine.recommend_next(
        "EMP-TEST", "OCC-001", current_skills, policy=RecommendationPolicy.STATIC
    )
    assert mod_b is not None
    assert exp_b.policy_used == "static"

    # 3. Adaptive
    mod_c, exp_c = engine.recommend_next(
        "EMP-TEST", "OCC-001", current_skills, policy=RecommendationPolicy.ADAPTIVE
    )
    assert mod_c is not None
    assert exp_c.policy_used == "adaptive"
    assert exp_c.composite_score > 0
    assert len(exp_c.rationale) > 0


def test_simulation_episode(setup_services):
    kg, engine, evaluator = setup_services
    reqs = kg.get_required_skills("OCC-001")
    req_list = list(reqs.keys())
    current_skills = [{"skill_id": s, "state": "mastered"} for s in req_list[:5]]

    episode = evaluator.simulate_episode(
        employee_id="EMP-TEST",
        occupation_id="OCC-001",
        initial_skills=current_skills,
        policy=RecommendationPolicy.ADAPTIVE,
        seed=42
    )

    assert "steps_to_target" in episode
    assert "reached_target" in episode
    assert episode["steps_to_target"] >= 1
    assert len(episode["history"]) == episode["steps_to_target"]
    assert episode["final_readiness"] >= episode["initial_readiness"]
