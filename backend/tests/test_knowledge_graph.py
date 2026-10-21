import pytest
from pathlib import Path
from backend.app.core.config import settings
from backend.app.services.knowledge_graph import OccupationSkillKnowledgeGraph


def test_knowledge_graph_construction():
    kg = OccupationSkillKnowledgeGraph()
    data_dir = settings.DATA_SYNTHETIC_DIR
    metrics = kg.build_from_directory(data_dir)

    assert metrics["occupation_nodes_count"] == 40
    assert metrics["skill_nodes_count"] == 220
    assert metrics["generic_skills_count"] == 26
    assert metrics["num_clusters"] == 8
    assert metrics["total_edges"] > 0
    assert metrics["connected_components_count"] >= 1


def test_knowledge_graph_queries():
    kg = OccupationSkillKnowledgeGraph()
    kg.build_from_directory(settings.DATA_SYNTHETIC_DIR)

    reqs = kg.get_required_skills("OCC-001")
    assert isinstance(reqs, dict)
    assert len(reqs) > 0

    # Every weight must be bounded in [0.4, 1.0]
    for w in reqs.values():
        assert 0.4 <= w <= 1.0

    cluster_id = kg.get_skill_cluster("SKL-001")
    assert cluster_id is not None
    assert 0 <= cluster_id < 8

    elements = kg.to_cytoscape_elements(max_skills=20)
    assert len(elements) > 0
