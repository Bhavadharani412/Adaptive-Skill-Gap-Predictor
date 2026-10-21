import logging
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
import pandas as pd

from backend.app.services.knowledge_graph import OccupationSkillKnowledgeGraph

logger = logging.getLogger(__name__)


class UnassessedPolicy(str, Enum):
    PESSIMISTIC = "pessimistic"    # Treat unassessed skills as missing gaps (standard HR safety)
    OPTIMISTIC = "optimistic"      # Assume unassessed skills are possessed until proven otherwise
    IN_PROGRESS_ONLY = "in_progress_as_gap"  # Treat in_progress and not_mastered as gaps, skip unassessed


class SkillGapAnalyzer:
    r"""
    Computes employee skill gaps: G_e = S_r \ P_e
    and extracts the 4 paper features for every identified gap:
      1. Occupation-skill relevance weight (w_{r,s})
      2. Generic skill indicator (g_s)
      3. Total skills already possessed by employee (|P_e|)
      4. Employee cluster coverage for the missing skill's cluster
    """

    FEATURE_NAMES = [
        "relevance_weight",
        "is_generic",
        "possessed_skills_count",
        "cluster_coverage"
    ]

    def __init__(self, knowledge_graph: OccupationSkillKnowledgeGraph):
        self.kg = knowledge_graph

    def compute_employee_gaps(
        self,
        employee_id: str,
        occupation_id: str,
        skill_records: List[Dict[str, Any]],
        unassessed_policy: UnassessedPolicy = UnassessedPolicy.PESSIMISTIC
    ) -> List[Dict[str, Any]]:
        """
        Calculates missing skills and the 4 paper features for an individual employee.
        """
        req_skills_map = self.kg.get_required_skills(occupation_id)
        if not req_skills_map:
            logger.warning("No required skills found for occupation %s", occupation_id)
            return []

        # Parse employee skills
        # Map skill_id -> {state: str, score: Optional[float]}
        emp_state_map = {}
        for rec in skill_records:
            s_id = str(rec["skill_id"])
            emp_state_map[s_id] = {
                "state": str(rec.get("state", "not_assessed")),
                "score": rec.get("assessment_score")
            }

        # Set of mastered skills P_e
        possessed_skill_ids: Set[str] = {
            s_id for s_id, data in emp_state_map.items()
            if data["state"] == "mastered"
        }
        total_possessed_count = len(possessed_skill_ids)

        gaps = []
        for s_id, rel_weight in req_skills_map.items():
            state_data = emp_state_map.get(s_id, {"state": "not_assessed", "score": None})
            state = state_data["state"]
            score = state_data["score"]

            # Determine whether skill is considered missing gap G_e
            is_gap = False
            if state == "not_mastered":
                is_gap = True
            elif state == "in_progress":
                is_gap = True
            elif state == "not_assessed":
                if unassessed_policy == UnassessedPolicy.PESSIMISTIC:
                    is_gap = True
                elif unassessed_policy == UnassessedPolicy.OPTIMISTIC:
                    is_gap = False
                elif unassessed_policy == UnassessedPolicy.IN_PROGRESS_ONLY:
                    is_gap = False
            elif state == "mastered":
                is_gap = False

            if not is_gap:
                continue

            # Compute 4 paper features
            f1_relevance = float(rel_weight)
            f2_is_generic = 1.0 if self.kg.is_generic_skill(s_id) else 0.0
            f3_possessed_count = float(total_possessed_count)

            # Feature 4: Cluster coverage
            c_id = self.kg.get_skill_cluster(s_id)
            if c_id is not None:
                # Coverage relative to required skills in this cluster
                cluster_req_skills = [
                    sk for sk in req_skills_map.keys()
                    if self.kg.get_skill_cluster(sk) == c_id
                ]
                if cluster_req_skills:
                    possessed_in_cluster = sum(1 for sk in cluster_req_skills if sk in possessed_skill_ids)
                    f4_cluster_coverage = float(possessed_in_cluster) / float(len(cluster_req_skills))
                else:
                    f4_cluster_coverage = 0.0
            else:
                f4_cluster_coverage = 0.0

            node_attrs = self.kg.graph.nodes.get(s_id, {})
            skill_name = node_attrs.get("name", s_id)
            cluster_name = node_attrs.get("cluster_name", f"Cluster {c_id}")

            gaps.append({
                "employee_id": employee_id,
                "occupation_id": occupation_id,
                "skill_id": s_id,
                "skill_name": skill_name,
                "cluster_id": c_id,
                "cluster_name": cluster_name,
                "relevance_weight": round(f1_relevance, 4),
                "is_generic": int(f2_is_generic),
                "possessed_skills_count": int(f3_possessed_count),
                "cluster_coverage": round(f4_cluster_coverage, 4),
                "current_state": state,
                "current_score": score
            })

        return gaps

    def compute_readiness(
        self,
        occupation_id: str,
        skill_records: List[Dict[str, Any]]
    ) -> float:
        """
        Paper formulation for readiness:
        Readiness(e) = (sum of relevance weights of mastered required skills) / (total relevance weight of all required skills)
        """
        req_skills_map = self.kg.get_required_skills(occupation_id)
        if not req_skills_map:
            return 0.0

        total_req_weight = sum(req_skills_map.values())
        if total_req_weight <= 0:
            return 0.0

        mastered_skill_ids = {
            str(r["skill_id"]) for r in skill_records
            if str(r.get("state")) == "mastered"
        }

        mastered_weight = sum(
            req_skills_map[s_id] for s_id in mastered_skill_ids
            if s_id in req_skills_map
        )

        return round(float(mastered_weight / total_req_weight), 4)

    def generate_dataset_gap_features(
        self,
        employees_df: pd.DataFrame,
        emp_skills_df: pd.DataFrame,
        unassessed_policy: UnassessedPolicy = UnassessedPolicy.PESSIMISTIC
    ) -> pd.DataFrame:
        """
        Batch extracts the 4 paper features for all employees across the dataset.
        """
        all_gaps = []
        skills_by_emp = emp_skills_df.groupby("employee_id")

        for _, emp in employees_df.iterrows():
            e_id = str(emp["employee_id"])
            o_id = str(emp["occupation_id"])

            if e_id in skills_by_emp.groups:
                e_records = skills_by_emp.get_group(e_id).to_dict("records")
            else:
                e_records = []

            emp_gaps = self.compute_employee_gaps(
                employee_id=e_id,
                occupation_id=o_id,
                skill_records=e_records,
                unassessed_policy=unassessed_policy
            )
            all_gaps.extend(emp_gaps)

        df = pd.DataFrame(all_gaps)
        logger.info(
            "Extracted %d gap instances across %d employees",
            len(df), len(employees_df)
        )
        return df
