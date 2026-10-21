import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np

from backend.app.services.knowledge_graph import OccupationSkillKnowledgeGraph
from backend.app.services.gap_analyzer import SkillGapAnalyzer, UnassessedPolicy

logger = logging.getLogger(__name__)


class RecommendationPolicy(str, Enum):
    RANDOM = "random"      # Policy A: Random baseline
    STATIC = "static"      # Policy B: Static initial relevance ranking
    ADAPTIVE = "adaptive"  # Policy C: Dynamic graph & cluster-transfer adaptive ranking


@dataclass
class RecommendationExplanation:
    skill_id: str
    skill_name: str
    occupation_id: str
    relevance_weight: float
    cluster_id: int
    cluster_name: str
    cluster_transfer_bonus: float
    prerequisite_fulfilled: bool
    is_generic: bool
    predicted_criticality: Optional[int]
    composite_score: float
    policy_used: str
    rationale: str


class RecommendationEngine:
    """
    Implements the three training policies:
    - Policy A: Random Baseline
    - Policy B: Static Relevance Baseline
    - Policy C: Graph-Informed Adaptive Recommendation
    """

    def __init__(
        self,
        kg: OccupationSkillKnowledgeGraph,
        modules_df_or_map: Any = None,
        modules_df: Any = None,
        alpha: float = 1.0,   # Weight of occupation-skill relevance
        beta: float = 0.35,   # Weight of cluster transfer bonus
        gamma: float = 0.25   # Weight of prerequisite readiness
    ):
        self.kg = kg
        self.gap_analyzer = SkillGapAnalyzer(kg)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.skill_to_module: Dict[str, Dict[str, Any]] = {}

        data = modules_df if modules_df is not None else modules_df_or_map
        if data is not None:
            self.load_modules(data)

    def load_modules(self, modules_data: Any):
        """Builds lookup index from target_skill_id to training module."""
        if hasattr(modules_data, "iterrows"):
            for _, row in modules_data.iterrows():
                s_id = str(row["target_skill_id"])
                self.skill_to_module[s_id] = {
                    "module_id": str(row["module_id"]),
                    "title": str(row["title"]),
                    "duration_hours": int(row.get("duration_hours", 10)),
                    "difficulty_level": str(row.get("difficulty_level", "intermediate")),
                    "learning_cost": float(row.get("learning_cost", 15.0)),
                    "target_skill_id": s_id
                }
        elif isinstance(modules_data, dict):
            self.skill_to_module = modules_data

    def recommend_next(
        self,
        employee_id: str,
        occupation_id: str,
        current_skills: List[Dict[str, Any]],
        policy: RecommendationPolicy = RecommendationPolicy.ADAPTIVE,
        rng: Optional[np.random.Generator] = None,
        unassessed_policy: UnassessedPolicy = UnassessedPolicy.PESSIMISTIC,
        static_rankings: Optional[List[str]] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[RecommendationExplanation]]:
        """
        Selects the next training module for the employee based on the specified policy.
        Returns (selected_module_dict, explanation_dataclass).
        """
        # 1. Identify current missing gaps G_e = S_r \ P_e
        gaps = self.gap_analyzer.compute_employee_gaps(
            employee_id=employee_id,
            occupation_id=occupation_id,
            skill_records=current_skills,
            unassessed_policy=unassessed_policy
        )

        if not gaps:
            return None, None

        missing_skill_ids = [g["skill_id"] for g in gaps]
        gap_lookup = {g["skill_id"]: g for g in gaps}

        # Set of mastered skills
        mastered_skill_ids: Set[str] = {
            str(r["skill_id"]) for r in current_skills
            if str(r.get("state")) == "mastered"
        }

        # Select target skill based on policy
        selected_skill_id: Optional[str] = None
        explanation: Optional[RecommendationExplanation] = None

        if policy == RecommendationPolicy.RANDOM:
            # Policy A: Uniform random choice among remaining gaps
            r_gen = rng if rng is not None else np.random.default_rng()
            selected_skill_id = str(r_gen.choice(missing_skill_ids))
            gap_info = gap_lookup[selected_skill_id]
            explanation = RecommendationExplanation(
                skill_id=selected_skill_id,
                skill_name=gap_info["skill_name"],
                occupation_id=occupation_id,
                relevance_weight=gap_info["relevance_weight"],
                cluster_id=gap_info["cluster_id"],
                cluster_name=gap_info["cluster_name"],
                cluster_transfer_bonus=0.0,
                prerequisite_fulfilled=True,
                is_generic=bool(gap_info["is_generic"]),
                predicted_criticality=None,
                composite_score=0.0,
                policy_used="random",
                rationale="Selected via Policy A (Random Baseline) from eligible outstanding skill gaps."
            )

        elif policy == RecommendationPolicy.STATIC:
            # Policy B: Static relevance ranking (fixed initial order of relevance weights)
            if static_rankings is not None:
                # Find first skill in static ranking that is still missing
                for s_id in static_rankings:
                    if s_id in gap_lookup:
                        selected_skill_id = s_id
                        break
            if selected_skill_id is None:
                # Fallback: Sort by relevance weight descending
                sorted_gaps = sorted(gaps, key=lambda x: x["relevance_weight"], reverse=True)
                selected_skill_id = sorted_gaps[0]["skill_id"]

            gap_info = gap_lookup[selected_skill_id]
            explanation = RecommendationExplanation(
                skill_id=selected_skill_id,
                skill_name=gap_info["skill_name"],
                occupation_id=occupation_id,
                relevance_weight=gap_info["relevance_weight"],
                cluster_id=gap_info["cluster_id"],
                cluster_name=gap_info["cluster_name"],
                cluster_transfer_bonus=0.0,
                prerequisite_fulfilled=True,
                is_generic=bool(gap_info["is_generic"]),
                predicted_criticality=None,
                composite_score=gap_info["relevance_weight"],
                policy_used="static",
                rationale=f"Selected via Policy B (Static Relevance): Highest fixed relevance weight ({gap_info['relevance_weight']:.2f}) for occupation."
            )

        elif policy == RecommendationPolicy.ADAPTIVE:
            # Policy C: Adaptive graph-informed recommendation
            # Score(s) = alpha * w_{r,s} + beta * ClusterTransferBonus + gamma * PrereqBonus
            scored_candidates = []

            for g in gaps:
                s_id = g["skill_id"]
                w_rs = g["relevance_weight"]
                c_id = g["cluster_id"]
                c_cov = g["cluster_coverage"]

                # Prerequisite satisfaction check
                prereqs = self.kg.get_prerequisites(s_id)
                all_prereqs_met = all(p in mastered_skill_ids for p in prereqs)
                prereq_score = 1.0 if all_prereqs_met else 0.20

                # Cluster transfer bonus: higher coverage in current cluster provides cognitive momentum
                cluster_bonus = c_cov

                composite_score = (
                    self.alpha * w_rs +
                    self.beta * cluster_bonus +
                    self.gamma * prereq_score
                )

                scored_candidates.append({
                    "skill_id": s_id,
                    "gap_info": g,
                    "score": composite_score,
                    "cluster_bonus": cluster_bonus,
                    "prereqs_met": all_prereqs_met
                })

            # Sort by composite adaptive score descending
            scored_candidates.sort(key=lambda x: x["score"], reverse=True)
            top = scored_candidates[0]
            selected_skill_id = top["skill_id"]
            gap_info = top["gap_info"]

            explanation = RecommendationExplanation(
                skill_id=selected_skill_id,
                skill_name=gap_info["skill_name"],
                occupation_id=occupation_id,
                relevance_weight=gap_info["relevance_weight"],
                cluster_id=gap_info["cluster_id"],
                cluster_name=gap_info["cluster_name"],
                cluster_transfer_bonus=round(top["cluster_bonus"], 3),
                prerequisite_fulfilled=top["prereqs_met"],
                is_generic=bool(gap_info["is_generic"]),
                predicted_criticality=1 if gap_info["relevance_weight"] >= 0.75 else 0,
                composite_score=round(top["score"], 4),
                policy_used="adaptive",
                rationale=(
                    f"Selected via Policy C (Adaptive Engine): Composite score {top['score']:.3f} "
                    f"[Relevance: {gap_info['relevance_weight']:.2f}, Cluster Momentum: {top['cluster_bonus']:.2f}, "
                    f"Prerequisites: {'Met' if top['prereqs_met'] else 'Pending'}]. Target cluster: {gap_info['cluster_name']}."
                )
            )

        # Retrieve module metadata
        module_info = self.skill_to_module.get(selected_skill_id, {
            "module_id": f"MOD-{selected_skill_id}",
            "title": f"Training on {gap_lookup[selected_skill_id]['skill_name']}",
            "duration_hours": 10,
            "difficulty_level": "intermediate",
            "learning_cost": 15.0,
            "target_skill_id": selected_skill_id
        })

        return module_info, explanation
