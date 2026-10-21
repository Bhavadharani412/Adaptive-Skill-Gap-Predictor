import pytest
from backend.app.core.config import settings
from backend.app.services.knowledge_graph import OccupationSkillKnowledgeGraph
from backend.app.services.gap_analyzer import SkillGapAnalyzer, UnassessedPolicy


@pytest.fixture(scope="module")
def shared_kg():
    kg = OccupationSkillKnowledgeGraph()
    kg.build_from_directory(settings.DATA_SYNTHETIC_DIR)
    return kg


def test_gap_computation_and_features(shared_kg):
    analyzer = SkillGapAnalyzer(shared_kg)
    reqs = shared_kg.get_required_skills("OCC-001")
    req_skills_list = list(reqs.keys())

    # Employee masters half, has gaps on the other half
    mastered_skills = req_skills_list[:len(req_skills_list)//2]
    missing_skills = req_skills_list[len(req_skills_list)//2:]

    skill_records = (
        [{"skill_id": s, "state": "mastered", "assessment_score": 0.9} for s in mastered_skills] +
        [{"skill_id": s, "state": "not_mastered", "assessment_score": 0.2} for s in missing_skills]
    )

    gaps = analyzer.compute_employee_gaps(
        employee_id="EMP-TEST",
        occupation_id="OCC-001",
        skill_records=skill_records
    )

    assert len(gaps) == len(missing_skills)

    for g in gaps:
        assert "relevance_weight" in g
        assert "is_generic" in g
        assert "possessed_skills_count" in g
        assert "cluster_coverage" in g
        assert 0.0 <= g["relevance_weight"] <= 1.0
        assert g["is_generic"] in (0, 1)
        assert g["possessed_skills_count"] == len(mastered_skills)
        assert 0.0 <= g["cluster_coverage"] <= 1.0


def test_readiness_calculation(shared_kg):
    analyzer = SkillGapAnalyzer(shared_kg)
    reqs = shared_kg.get_required_skills("OCC-001")
    req_list = list(reqs.keys())

    # All mastered -> readiness == 1.0
    all_mastered = [{"skill_id": s, "state": "mastered"} for s in req_list]
    readiness_full = analyzer.compute_readiness("OCC-001", all_mastered)
    assert readiness_full == 1.0

    # None mastered -> readiness == 0.0
    none_mastered = [{"skill_id": s, "state": "not_mastered"} for s in req_list]
    readiness_zero = analyzer.compute_readiness("OCC-001", none_mastered)
    assert readiness_zero == 0.0
