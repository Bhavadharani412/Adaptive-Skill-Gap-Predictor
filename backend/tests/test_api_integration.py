import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.database.init_db import init_and_seed_database


@pytest.fixture(scope="module")
def client():
    init_and_seed_database(force_reseed=False)
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_occupations_and_skills(client):
    res = client.get("/api/occupations")
    assert res.status_code == 200
    occs = res.json()
    assert len(occs) >= 10

    res_skills = client.get("/api/skills?limit=10")
    assert res_skills.status_code == 200
    assert len(res_skills.json()) == 10


def test_end_to_end_employee_workflow(client):
    # 1. Create a new employee
    emp_payload = {
        "name": "Integration Test Engineer",
        "email": "integration.test@asgp.org",
        "occupation_id": "OCC-001",
        "department": "AI Research"
    }
    create_res = client.post("/api/employees", json=emp_payload)
    assert create_res.status_code in (200, 400)

    if create_res.status_code == 200:
        emp_id = create_res.json()["employee_id"]
    else:
        # If already exists from prior test run, fetch from list
        emp_list = client.get("/api/employees?limit=1").json()
        emp_id = emp_list[0]["employee_id"]

    # 2. Get Employee Gaps
    gaps_res = client.get(f"/api/gaps/{emp_id}")
    assert gaps_res.status_code == 200
    gaps_data = gaps_res.json()
    assert "gaps" in gaps_data
    initial_readiness = gaps_data["current_readiness"]
    initial_gaps_count = len(gaps_data["gaps"])
    assert initial_gaps_count > 0

    # 3. Get Recommendation
    rec_res = client.get(f"/api/recommendations/{emp_id}?policy=adaptive")
    assert rec_res.status_code == 200
    rec_data = rec_res.json()
    assert rec_data["recommended_module"] is not None
    recommended_mod = rec_data["recommended_module"]
    mod_id = recommended_mod["module_id"]
    assert len(recommended_mod["rationale"]) > 0

    # 4. Complete Module and Submit Assessment (passing score 0.90)
    complete_payload = {
        "module_id": mod_id,
        "score": 0.90,
        "manual_passed": True
    }
    complete_res = client.post(
        f"/api/recommendations/{emp_id}/complete-module",
        json=complete_payload
    )
    assert complete_res.status_code == 200
    completion_data = complete_res.json()
    assert completion_data["passed"] is True
    assert completion_data["updated_readiness"] >= initial_readiness

    # 5. Verify updated gaps reduced or profile reflects mastery
    gaps_after = client.get(f"/api/gaps/{emp_id}").json()
    assert gaps_after["current_readiness"] >= initial_readiness
