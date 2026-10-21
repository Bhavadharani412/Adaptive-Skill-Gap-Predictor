import copy
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats

from backend.app.services.knowledge_graph import OccupationSkillKnowledgeGraph
from backend.app.services.gap_analyzer import SkillGapAnalyzer, UnassessedPolicy
from backend.app.services.recommendation_engine import (
    RecommendationEngine,
    RecommendationPolicy,
    RecommendationExplanation
)

logger = logging.getLogger(__name__)


class PolicyEvaluator:
    """
    Evaluates Policies A (Random), B (Static), and C (Adaptive) under matched experimental conditions.
    Simulates employee training episodes until reaching the target readiness (default 0.80) or budget exhaustion.
    Performs paired statistical significance tests (Wilcoxon signed-rank test) and effect size calculations.
    """

    def __init__(
        self,
        kg: OccupationSkillKnowledgeGraph,
        engine: RecommendationEngine,
        target_readiness: float = 0.80,
        max_budget: int = 50
    ):
        self.kg = kg
        self.engine = engine
        self.gap_analyzer = SkillGapAnalyzer(kg)
        self.target_readiness = target_readiness
        self.max_budget = max_budget

    def simulate_episode(
        self,
        employee_id: str,
        occupation_id: str,
        initial_skills: List[Dict[str, Any]],
        policy: RecommendationPolicy,
        seed: int = 42,
        static_rankings: Optional[List[str]] = None,
        assessment_noise_std: float = 0.05
    ) -> Dict[str, Any]:
        """
        Simulates one full training episode for an employee under a specified policy.
        """
        rng = np.random.default_rng(seed)
        skills_state = [copy.deepcopy(s) for s in initial_skills]

        initial_readiness = self.gap_analyzer.compute_readiness(occupation_id, skills_state)
        current_readiness = initial_readiness

        history = []
        total_learning_hours = 0.0
        step = 0
        reached_target = current_readiness >= self.target_readiness

        while not reached_target and step < self.max_budget:
            step += 1

            # 1. Recommend next module
            module, explanation = self.engine.recommend_next(
                employee_id=employee_id,
                occupation_id=occupation_id,
                current_skills=skills_state,
                policy=policy,
                rng=rng,
                static_rankings=static_rankings
            )

            if module is None:
                # No more missing skills
                break

            target_skill_id = module["target_skill_id"]
            duration = module.get("duration_hours", 10)
            total_learning_hours += duration

            # 2. Simulate assessment outcome
            # Transfer learning model: pass probability boosted by cluster coverage
            gaps = self.gap_analyzer.compute_employee_gaps(
                employee_id, occupation_id, skills_state
            )
            gap_lookup = {g["skill_id"]: g for g in gaps}
            cov = gap_lookup.get(target_skill_id, {}).get("cluster_coverage", 0.0)

            # Base pass probability: 0.80 + 0.15 * coverage
            pass_prob = np.clip(0.78 + 0.15 * cov + rng.normal(0, assessment_noise_std), 0.50, 0.98)
            passed = rng.random() < pass_prob

            if passed:
                score = round(float(rng.uniform(0.78, 0.98)), 2)
                new_state = "mastered"
            else:
                score = round(float(rng.uniform(0.50, 0.72)), 2)
                new_state = "in_progress"

            # 3. Update skill state
            skill_found = False
            for s in skills_state:
                if str(s["skill_id"]) == target_skill_id:
                    s["state"] = new_state
                    s["assessment_score"] = score
                    skill_found = True
                    break
            if not skill_found:
                skills_state.append({
                    "skill_id": target_skill_id,
                    "state": new_state,
                    "assessment_score": score
                })

            # 4. Recalculate readiness
            current_readiness = self.gap_analyzer.compute_readiness(occupation_id, skills_state)
            reached_target = current_readiness >= self.target_readiness

            history.append({
                "step": step,
                "module_id": module["module_id"],
                "target_skill_id": target_skill_id,
                "passed": passed,
                "score": score,
                "readiness_after_step": current_readiness,
                "rationale": explanation.rationale if explanation else ""
            })

        return {
            "employee_id": employee_id,
            "occupation_id": occupation_id,
            "policy": policy.value,
            "seed": seed,
            "initial_readiness": initial_readiness,
            "final_readiness": current_readiness,
            "target_readiness": self.target_readiness,
            "steps_to_target": step,
            "reached_target": reached_target,
            "censored": not reached_target and (step >= self.max_budget),
            "total_learning_hours": total_learning_hours,
            "history": history
        }

    def evaluate_cohort(
        self,
        employees_df: pd.DataFrame,
        emp_skills_df: pd.DataFrame,
        sample_size: int = 100,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Runs matched cohort evaluation across all 3 policies on identical employee starting states.
        """
        skills_by_emp = emp_skills_df.groupby("employee_id")

        if len(employees_df) > sample_size:
            sample_emps = employees_df.sample(n=sample_size, random_state=random_seed)
        else:
            sample_emps = employees_df

        results = {
            "random": [],
            "static": [],
            "adaptive": []
        }

        for idx, (_, emp) in enumerate(sample_emps.iterrows()):
            e_id = str(emp["employee_id"])
            o_id = str(emp["occupation_id"])

            if e_id in skills_by_emp.groups:
                initial_skills = skills_by_emp.get_group(e_id).to_dict("records")
            else:
                initial_skills = []

            # Determine static ranking order once for Policy B
            initial_gaps = self.gap_analyzer.compute_employee_gaps(e_id, o_id, initial_skills)
            static_sorted = sorted(initial_gaps, key=lambda x: x["relevance_weight"], reverse=True)
            static_order = [g["skill_id"] for g in static_sorted]

            # Assign a matched, independent pseudo-random seed per employee across all policies
            emp_seed = (random_seed * 1000) + idx

            # 1. Policy A: Random
            res_a = self.simulate_episode(
                e_id, o_id, initial_skills, RecommendationPolicy.RANDOM,
                seed=emp_seed
            )
            # 2. Policy B: Static
            res_b = self.simulate_episode(
                e_id, o_id, initial_skills, RecommendationPolicy.STATIC,
                seed=emp_seed, static_rankings=static_order
            )
            # 3. Policy C: Adaptive
            res_c = self.simulate_episode(
                e_id, o_id, initial_skills, RecommendationPolicy.ADAPTIVE,
                seed=emp_seed
            )

            results["random"].append(res_a)
            results["static"].append(res_b)
            results["adaptive"].append(res_c)

        # Statistical analysis
        summary = self._summarize_cohort_results(results)
        return summary

    def _summarize_cohort_results(self, results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        policy_stats = {}
        for p_name, eps in results.items():
            steps = [e["steps_to_target"] for e in eps]
            reached = [1 if e["reached_target"] else 0 for e in eps]
            hours = [e["total_learning_hours"] for e in eps]
            final_read = [e["final_readiness"] for e in eps]

            policy_stats[p_name] = {
                "cohort_size": len(eps),
                "success_rate": round(float(np.mean(reached)), 4),
                "steps_mean": round(float(np.mean(steps)), 2),
                "steps_median": round(float(np.median(steps)), 2),
                "steps_std": round(float(np.std(steps)), 2),
                "steps_min": int(np.min(steps)),
                "steps_max": int(np.max(steps)),
                "mean_learning_hours": round(float(np.mean(hours)), 1),
                "mean_final_readiness": round(float(np.mean(final_read)), 4),
                "censored_count": int(sum(1 for e in eps if e["censored"]))
            }

        # Paired Wilcoxon Signed-Rank Tests
        steps_rand = np.array([e["steps_to_target"] for e in results["random"]])
        steps_stat = np.array([e["steps_to_target"] for e in results["static"]])
        steps_adapt = np.array([e["steps_to_target"] for e in results["adaptive"]])

        w_stat_vs_rand = stats.wilcoxon(steps_rand, steps_stat, alternative="two-sided")
        w_adapt_vs_rand = stats.wilcoxon(steps_rand, steps_adapt, alternative="two-sided")
        w_adapt_vs_stat = stats.wilcoxon(steps_stat, steps_adapt, alternative="two-sided")

        # Effect sizes (Cohen's d)
        diff_stat_rand = steps_rand - steps_stat
        d_stat_rand = np.mean(diff_stat_rand) / (np.std(diff_stat_rand) + 1e-8)

        diff_adapt_stat = steps_stat - steps_adapt
        d_adapt_stat = np.mean(diff_adapt_stat) / (np.std(diff_adapt_stat) + 1e-8)

        statistical_tests = {
            "static_vs_random": {
                "wilcoxon_stat": float(w_stat_vs_rand.statistic),
                "p_value": float(w_stat_vs_rand.pvalue),
                "cohens_d": round(float(d_stat_rand), 4),
                "significant_at_05": bool(w_stat_vs_rand.pvalue < 0.05)
            },
            "adaptive_vs_random": {
                "wilcoxon_stat": float(w_adapt_vs_rand.statistic),
                "p_value": float(w_adapt_vs_rand.pvalue),
                "cohens_d": round(float(np.mean(steps_rand - steps_adapt) / (np.std(steps_rand - steps_adapt) + 1e-8)), 4),
                "significant_at_05": bool(w_adapt_vs_rand.pvalue < 0.05)
            },
            "adaptive_vs_static": {
                "wilcoxon_stat": float(w_adapt_vs_stat.statistic),
                "p_value": float(w_adapt_vs_stat.pvalue),
                "cohens_d": round(float(d_adapt_stat), 4),
                "significant_at_05": bool(w_adapt_vs_stat.pvalue < 0.05)
            }
        }

        # Reference values from published paper
        paper_references = {
            "random_policy_mean_steps": 42.66,
            "static_policy_mean_steps": 21.75,
            "adaptive_policy_mean_steps": 24.24,
            "target_readiness": 0.80,
            "notes": (
                "Under greedy relevance-only metric definitions, static relevance reaches the 80% numerical "
                "threshold quickly because it targets the highest numerator weights first. However, adaptive recommendation "
                "fosters cluster transfer, reduces failure risk, and honors prerequisite sequences, providing pedagogical durability."
            )
        }

        return {
            "target_readiness": self.target_readiness,
            "max_budget": self.max_budget,
            "policies": policy_stats,
            "paired_statistical_tests": statistical_tests,
            "paper_references_comparison": paper_references
        }
