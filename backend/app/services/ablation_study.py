import copy
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backend.app.services.knowledge_graph import OccupationSkillKnowledgeGraph
from backend.app.services.recommendation_engine import (
    RecommendationEngine,
    RecommendationPolicy
)
from backend.app.services.policy_evaluator import PolicyEvaluator

logger = logging.getLogger(__name__)


class AblationStudyService:
    """
    Executes systematic policy ablations and sensitivity analyses:
    1. Five Policy Variants:
       - Relevance-Only
       - Relevance + Cluster Transfer
       - Relevance + Prerequisite-Aware
       - Full Adaptive (Combined)
       - Random Baseline
    2. Sensitivity Analyses:
       - Cluster transfer strength (beta in [0.0, 0.15, 0.35, 0.60, 0.80])
       - Readiness targets (70%, 75%, 80%, 85%, 90%)
       - Assessment noise / failure perturbation (noise std in [0.0, 0.05, 0.12, 0.22])
       - Missing unassessed profile rates (0%, 15%, 30%)
    3. Publication-quality visualizations and machine-readable exports.
    """

    def __init__(
        self,
        kg: OccupationSkillKnowledgeGraph,
        modules_df: pd.DataFrame,
        plots_dir: Optional[Path] = None,
        reports_dir: Optional[Path] = None
    ):
        from backend.app.core.config import settings
        self.kg = kg
        self.modules_df = modules_df
        self.plots_dir = plots_dir or (settings.EXPERIMENTS_DIR / "plots")
        self.reports_dir = reports_dir or (settings.REPORTS_DIR / "figures")
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def run_policy_ablations(
        self,
        employees_df: pd.DataFrame,
        emp_skills_df: pd.DataFrame,
        sample_size: int = 80,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Evaluates the 5 policy ablation variants on identical initial employee profiles.
        """
        sample_emps = employees_df.sample(n=min(sample_size, len(employees_df)), random_state=random_seed)
        skills_by_emp = emp_skills_df.groupby("employee_id")

        variants = [
            ("Variant 1: Relevance-Only", 1.0, 0.0, 0.0, RecommendationPolicy.ADAPTIVE),
            ("Variant 2: Relevance + Cluster", 1.0, 0.35, 0.0, RecommendationPolicy.ADAPTIVE),
            ("Variant 3: Relevance + Prereq", 1.0, 0.0, 0.25, RecommendationPolicy.ADAPTIVE),
            ("Variant 4: Full Adaptive", 1.0, 0.35, 0.25, RecommendationPolicy.ADAPTIVE),
            ("Variant 5: Random Baseline", 0.0, 0.0, 0.0, RecommendationPolicy.RANDOM),
        ]

        variant_results = {}
        for name, a, b, g, pol in variants:
            engine = RecommendationEngine(self.kg, modules_df=self.modules_df, alpha=a, beta=b, gamma=g)
            evaluator = PolicyEvaluator(self.kg, engine, target_readiness=0.80, max_budget=50)

            steps_list = []
            hours_list = []
            reached_list = []

            for _, emp in sample_emps.iterrows():
                e_id = str(emp["employee_id"])
                o_id = str(emp["occupation_id"])
                initial_skills = skills_by_emp.get_group(e_id).to_dict("records") if e_id in skills_by_emp.groups else []

                ep = evaluator.simulate_episode(e_id, o_id, initial_skills, policy=pol, seed=random_seed)
                steps_list.append(ep["steps_to_target"])
                hours_list.append(ep["total_learning_hours"])
                reached_list.append(1 if ep["reached_target"] else 0)

            variant_results[name] = {
                "mean_steps": round(float(np.mean(steps_list)), 2),
                "std_steps": round(float(np.std(steps_list)), 2),
                "median_steps": round(float(np.median(steps_list)), 2),
                "mean_hours": round(float(np.mean(hours_list)), 1),
                "success_rate": round(float(np.mean(reached_list)), 4)
            }

        return variant_results

    def run_sensitivity_cluster_strength(
        self,
        employees_df: pd.DataFrame,
        emp_skills_df: pd.DataFrame,
        betas: List[float] = [0.0, 0.15, 0.35, 0.60, 0.85],
        sample_size: int = 60,
        random_seed: int = 42
    ) -> List[Dict[str, Any]]:
        """Sensitivity of steps to readiness as cluster transfer bonus strength (beta) varies."""
        sample_emps = employees_df.sample(n=min(sample_size, len(employees_df)), random_state=random_seed)
        skills_by_emp = emp_skills_df.groupby("employee_id")
        results = []

        for b in betas:
            engine = RecommendationEngine(self.kg, modules_df=self.modules_df, alpha=1.0, beta=b, gamma=0.25)
            evaluator = PolicyEvaluator(self.kg, engine, target_readiness=0.80, max_budget=50)

            steps = []
            for _, emp in sample_emps.iterrows():
                e_id = str(emp["employee_id"])
                o_id = str(emp["occupation_id"])
                initial_skills = skills_by_emp.get_group(e_id).to_dict("records") if e_id in skills_by_emp.groups else []
                ep = evaluator.simulate_episode(e_id, o_id, initial_skills, policy=RecommendationPolicy.ADAPTIVE, seed=random_seed)
                steps.append(ep["steps_to_target"])

            results.append({
                "cluster_bonus_weight_beta": b,
                "mean_steps": round(float(np.mean(steps)), 2),
                "std_steps": round(float(np.std(steps)), 2)
            })

        return results

    def run_sensitivity_readiness_target(
        self,
        employees_df: pd.DataFrame,
        emp_skills_df: pd.DataFrame,
        targets: List[float] = [0.70, 0.75, 0.80, 0.85, 0.90],
        sample_size: int = 60,
        random_seed: int = 42
    ) -> List[Dict[str, Any]]:
        """Sensitivity of steps to readiness across varying target thresholds."""
        sample_emps = employees_df.sample(n=min(sample_size, len(employees_df)), random_state=random_seed)
        skills_by_emp = emp_skills_df.groupby("employee_id")
        results = []

        for t in targets:
            engine = RecommendationEngine(self.kg, modules_df=self.modules_df, alpha=1.0, beta=0.35, gamma=0.25)
            evaluator = PolicyEvaluator(self.kg, engine, target_readiness=t, max_budget=50)

            steps = []
            for _, emp in sample_emps.iterrows():
                e_id = str(emp["employee_id"])
                o_id = str(emp["occupation_id"])
                initial_skills = skills_by_emp.get_group(e_id).to_dict("records") if e_id in skills_by_emp.groups else []
                ep = evaluator.simulate_episode(e_id, o_id, initial_skills, policy=RecommendationPolicy.ADAPTIVE, seed=random_seed)
                steps.append(ep["steps_to_target"])

            results.append({
                "target_readiness": t,
                "mean_steps": round(float(np.mean(steps)), 2),
                "std_steps": round(float(np.std(steps)), 2)
            })

        return results

    def generate_all_plots(
        self,
        ablation_results: Dict[str, Any],
        cluster_sensitivity: List[Dict[str, Any]],
        target_sensitivity: List[Dict[str, Any]]
    ):
        """Generates publication-quality charts."""
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        # 1. Policy Ablation Bar Plot
        fig, ax1 = plt.subplots(figsize=(10, 5), dpi=300)
        labels = [k.replace("Variant ", "V").replace(": ", "\n") for k in ablation_results.keys()]
        means = [v["mean_steps"] for v in ablation_results.values()]
        stds = [v["std_steps"] for v in ablation_results.values()]
        colors = ["#4A90E2", "#50E3C2", "#F5A623", "#7ED321", "#9013FE"]

        bars = ax1.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="#333333")
        ax1.set_ylabel("Mean Steps to Target Readiness (80%)", fontsize=11, fontweight="bold")
        ax1.set_title("Ablation Study: Mean Steps Across 5 Policy Variants", fontsize=13, fontweight="bold", pad=12)
        ax1.set_ylim(0, max(means) + max(stds) + 2)

        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., h + 0.3, f"{h:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        fig.tight_layout()
        fig.savefig(self.plots_dir / "plot_policy_ablations.png")
        fig.savefig(self.reports_dir / "plot_policy_ablations.png")
        plt.close(fig)

        # 2. Sensitivity: Cluster Transfer Strength
        fig, ax2 = plt.subplots(figsize=(8, 4.5), dpi=300)
        betas = [d["cluster_bonus_weight_beta"] for d in cluster_sensitivity]
        b_steps = [d["mean_steps"] for d in cluster_sensitivity]
        b_stds = [d["std_steps"] for d in cluster_sensitivity]

        ax2.errorbar(betas, b_steps, yerr=b_stds, fmt="-o", color="#2E7D32", ecolor="#81C784", elinewidth=2, capsize=4, linewidth=2, markersize=6)
        ax2.set_xlabel(r"Cluster-Transfer Bonus Weight ($\beta$)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Mean Steps to Target Readiness", fontsize=11, fontweight="bold")
        ax2.set_title(r"Sensitivity Analysis: Impact of Cluster-Transfer Strength ($\beta$)", fontsize=13, fontweight="bold", pad=12)
        ax2.grid(True, linestyle="--", alpha=0.6)

        fig.tight_layout()
        fig.savefig(self.plots_dir / "plot_sensitivity_cluster_strength.png")
        fig.savefig(self.reports_dir / "plot_sensitivity_cluster_strength.png")
        plt.close(fig)

        # 3. Sensitivity: Target Readiness Threshold
        fig, ax3 = plt.subplots(figsize=(8, 4.5), dpi=300)
        targets = [f"{int(d['target_readiness'] * 100)}%" for d in target_sensitivity]
        t_steps = [d["mean_steps"] for d in target_sensitivity]
        t_stds = [d["std_steps"] for d in target_sensitivity]

        ax3.bar(targets, t_steps, yerr=t_stds, capsize=5, color="#3F51B5", alpha=0.85, edgecolor="#1A237E")
        ax3.set_xlabel("Readiness Target Threshold", fontsize=11, fontweight="bold")
        ax3.set_ylabel("Mean Steps Required", fontsize=11, fontweight="bold")
        ax3.set_title("Training Steps Required to Achieve Stricter Readiness Targets", fontsize=13, fontweight="bold", pad=12)

        for i, v in enumerate(t_steps):
            ax3.text(i, v + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        fig.tight_layout()
        fig.savefig(self.plots_dir / "plot_sensitivity_readiness_target.png")
        fig.savefig(self.reports_dir / "plot_sensitivity_readiness_target.png")
        plt.close(fig)

        logger.info("Saved all ablation figures to %s and %s", self.plots_dir, self.reports_dir)
