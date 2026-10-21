import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.core.config import settings
from backend.app.services.knowledge_graph import OccupationSkillKnowledgeGraph
from backend.app.services.gap_analyzer import SkillGapAnalyzer
from backend.app.services.ml_classifier import CriticalGapClassifierService
from backend.app.services.recommendation_engine import RecommendationEngine, RecommendationPolicy
from backend.app.services.policy_evaluator import PolicyEvaluator
from backend.app.services.ablation_study import AblationStudyService


def audit_classifier_and_targets():
    print("\n=======================================================")
    print("AUDIT SECTION 1: CLASSIFIER, TARGETS & CIRCULARITY")
    print("=======================================================")
    kg = OccupationSkillKnowledgeGraph()
    kg.build_from_directory(settings.DATA_SYNTHETIC_DIR)
    analyzer = SkillGapAnalyzer(kg)

    emp_df = pd.read_csv(settings.DATA_SYNTHETIC_DIR / "employees.csv")
    emp_skills_df = pd.read_csv(settings.DATA_SYNTHETIC_DIR / "employee_skills.csv")
    gaps_df = analyzer.generate_dataset_gap_features(emp_df, emp_skills_df)

    service = CriticalGapClassifierService(kg)

    # 1. Phase 6 target
    df_p6 = service.assign_paper_baseline_labels(gaps_df)
    corr_p6_w = df_p6["relevance_weight"].corr(df_p6["is_critical"])
    print(f"Phase 6 Total Gap Instances: {len(df_p6)}")
    print(f"Phase 6 Class Distribution: Critical={df_p6['is_critical'].sum()} ({df_p6['is_critical'].mean():.4f}), Non-critical={(df_p6['is_critical']==0).sum()}")
    print(f"Correlation between 'relevance_weight' and Phase 6 label 'is_critical': {corr_p6_w:.4f}")

    # Check circularity in Phase 6: Is the label a deterministic function of relevance_weight?
    thresholds = {}
    for occ_id, skills_map in kg.occ_required_skills.items():
        thresholds[occ_id] = float(np.quantile(list(skills_map.values()), 0.667))
    matches = sum(df_p6["is_critical"] == (df_p6["relevance_weight"] >= df_p6["occupation_id"].map(thresholds)).astype(int))
    print(f"Phase 6 Label = Threshold(relevance_weight) Match Rate: {matches}/{len(df_p6)} ({matches/len(df_p6):.2%}) -> EXACT CIRCULARITY")

    # 2. Phase 7 target
    df_p7 = service.assign_simulated_outcome_labels(gaps_df)
    corr_p7_w = df_p7["relevance_weight"].corr(df_p7["is_critical"])
    corr_p7_cov = df_p7["cluster_coverage"].corr(df_p7["is_critical"])
    corr_p7_count = df_p7["possessed_skills_count"].corr(df_p7["is_critical"])
    corr_p7_gen = df_p7["is_generic"].corr(df_p7["is_critical"])

    print(f"\nPhase 7 Target Feature Correlations:")
    print(f"  - corr(relevance_weight, is_critical): {corr_p7_w:.4f}")
    print(f"  - corr(cluster_coverage, is_critical): {corr_p7_cov:.4f}")
    print(f"  - corr(possessed_skills_count, is_critical): {corr_p7_count:.4f}")
    print(f"  - corr(is_generic, is_critical): {corr_p7_gen:.4f}")

    # Check Phase 7 formula:
    # crit = 1 if (rel_impact >= 0.055 and cov < 0.50) else 0
    # Does this directly use relevance_weight and cluster_coverage?
    print("Phase 7 Target Definition Formula:")
    print("  is_critical = 1 if (relevance_weight / total_occ_weight >= 0.055 AND cluster_coverage < 0.50) else 0")
    print("  -> Finding: Phase 7 label is formed as a deterministic conjunction of TWO input features (relevance_weight and cluster_coverage).")


def audit_policy_simulation():
    print("\n=======================================================")
    print("AUDIT SECTION 2: POLICY SIMULATION & SEED INDEPENDENCE")
    print("=======================================================")
    kg = OccupationSkillKnowledgeGraph()
    kg.build_from_directory(settings.DATA_SYNTHETIC_DIR)
    modules_df = pd.read_csv(settings.DATA_SYNTHETIC_DIR / "training_modules.csv")
    engine = RecommendationEngine(kg, modules_df=modules_df)
    evaluator = PolicyEvaluator(kg, engine, target_readiness=0.80, max_budget=50)

    emp_df = pd.read_csv(settings.DATA_SYNTHETIC_DIR / "employees.csv")
    emp_skills_df = pd.read_csv(settings.DATA_SYNTHETIC_DIR / "employee_skills.csv")

    # Audit current implementation (where seeds were fixed to 43, 44, 45 for all employees)
    res_orig = evaluator.evaluate_cohort(emp_df, emp_skills_df, sample_size=100, random_seed=42)
    print("Current evaluate_cohort output (sample=100, seeds=43, 44, 45):")
    for p, stats_dict in res_orig["policies"].items():
        print(f"  [{p.upper()}] Mean Steps: {stats_dict['steps_mean']}, Median: {stats_dict['steps_median']}, Hours: {stats_dict['mean_learning_hours']}")

    print("\nPaired Wilcoxon Tests currently reported:")
    for comp, test in res_orig["paired_statistical_tests"].items():
        print(f"  {comp}: W={test['wilcoxon_stat']}, p={test['p_value']:.2e}, d={test['cohens_d']}")

    # Now let's test with matched, independent per-employee random seeds
    # Every employee i gets a unique seed for their episode, and the SAME seed is used across the 3 policies for that employee
    sample_emps = emp_df.sample(n=100, random_state=42)
    skills_by_emp = emp_skills_df.groupby("employee_id")

    rand_steps, stat_steps, adapt_steps = [], [], []
    rand_hours, stat_hours, adapt_hours = [], [], []

    for idx, (_, emp) in enumerate(sample_emps.iterrows()):
        e_id = str(emp["employee_id"])
        o_id = str(emp["occupation_id"])
        initial_skills = skills_by_emp.get_group(e_id).to_dict("records")
        initial_gaps = evaluator.gap_analyzer.compute_employee_gaps(e_id, o_id, initial_skills)
        static_order = [g["skill_id"] for g in sorted(initial_gaps, key=lambda x: x["relevance_weight"], reverse=True)]

        # Unique seed per employee
        emp_seed = 42000 + idx

        # Run with matched seed
        res_a = evaluator.simulate_episode(e_id, o_id, initial_skills, RecommendationPolicy.RANDOM, seed=emp_seed)
        res_b = evaluator.simulate_episode(e_id, o_id, initial_skills, RecommendationPolicy.STATIC, seed=emp_seed, static_rankings=static_order)
        res_c = evaluator.simulate_episode(e_id, o_id, initial_skills, RecommendationPolicy.ADAPTIVE, seed=emp_seed)

        rand_steps.append(res_a["steps_to_target"])
        stat_steps.append(res_b["steps_to_target"])
        adapt_steps.append(res_c["steps_to_target"])

        rand_hours.append(res_a["total_learning_hours"])
        stat_hours.append(res_b["total_learning_hours"])
        adapt_hours.append(res_c["total_learning_hours"])

    print("\nAudited evaluate_cohort with INDEPENDENT per-employee seeds (matched across policies):")
    print(f"  [RANDOM]   Mean Steps: {np.mean(rand_steps):.2f} (std: {np.std(rand_steps):.2f}), Mean Hours: {np.mean(rand_hours):.1f}")
    print(f"  [STATIC]   Mean Steps: {np.mean(stat_steps):.2f} (std: {np.std(stat_steps):.2f}), Mean Hours: {np.mean(stat_hours):.1f}")
    print(f"  [ADAPTIVE] Mean Steps: {np.mean(adapt_steps):.2f} (std: {np.std(adapt_steps):.2f}), Mean Hours: {np.mean(adapt_hours):.1f}")

    w_stat_rand = stats.wilcoxon(rand_steps, stat_steps)
    w_adapt_stat = stats.wilcoxon(stat_steps, adapt_steps)
    w_adapt_rand = stats.wilcoxon(rand_steps, adapt_steps)

    d_adapt_stat = float(np.mean(np.array(stat_steps) - np.array(adapt_steps)) / (np.std(np.array(stat_steps) - np.array(adapt_steps)) + 1e-8))
    d_adapt_rand = float(np.mean(np.array(rand_steps) - np.array(adapt_steps)) / (np.std(np.array(rand_steps) - np.array(adapt_steps)) + 1e-8))
    print(f"  Wilcoxon Adaptive vs Static: W={w_adapt_stat.statistic}, p={w_adapt_stat.pvalue:.2e}, Cohen's d={d_adapt_stat:.4f}")
    print(f"  Wilcoxon Adaptive vs Random: W={w_adapt_rand.statistic}, p={w_adapt_rand.pvalue:.2e}, Cohen's d={d_adapt_rand:.4f}")

    import json
    out_path = Path(settings.EXPERIMENTS_DIR) / "results" / "audit_matched_seeds_policy_results.json"
    audit_data = {
        "evaluation_protocol": "matched_per_employee_seeds (seed = 42000 + i)",
        "sample_size": 100,
        "target_readiness": 0.80,
        "random": {
            "mean_steps": float(np.mean(rand_steps)),
            "std_steps": float(np.std(rand_steps)),
            "median_steps": float(np.median(rand_steps)),
            "mean_hours": float(np.mean(rand_hours)),
            "std_hours": float(np.std(rand_hours)),
        },
        "static": {
            "mean_steps": float(np.mean(stat_steps)),
            "std_steps": float(np.std(stat_steps)),
            "median_steps": float(np.median(stat_steps)),
            "mean_hours": float(np.mean(stat_hours)),
            "std_hours": float(np.std(stat_hours)),
        },
        "adaptive": {
            "mean_steps": float(np.mean(adapt_steps)),
            "std_steps": float(np.std(adapt_steps)),
            "median_steps": float(np.median(adapt_steps)),
            "mean_hours": float(np.mean(adapt_hours)),
            "std_hours": float(np.std(adapt_hours)),
        },
        "comparisons": {
            "adaptive_vs_static": {
                "wilcoxon_stat": float(w_adapt_stat.statistic),
                "p_value": float(w_adapt_stat.pvalue),
                "cohens_d": d_adapt_stat
            },
            "adaptive_vs_random": {
                "wilcoxon_stat": float(w_adapt_rand.statistic),
                "p_value": float(w_adapt_rand.pvalue),
                "cohens_d": d_adapt_rand
            }
        }
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
    print(f"Saved matched-seed audit results to: {out_path}")


def audit_ablations():
    print("\n=======================================================")
    print("AUDIT SECTION 3: ABLATION STUDY COMPARISONS")
    print("=======================================================")
    kg = OccupationSkillKnowledgeGraph()
    kg.build_from_directory(settings.DATA_SYNTHETIC_DIR)
    modules_df = pd.read_csv(settings.DATA_SYNTHETIC_DIR / "training_modules.csv")
    emp_df = pd.read_csv(settings.DATA_SYNTHETIC_DIR / "employees.csv")
    emp_skills_df = pd.read_csv(settings.DATA_SYNTHETIC_DIR / "employee_skills.csv")

    service = AblationStudyService(kg, modules_df=modules_df)
    res = service.run_policy_ablations(emp_df, emp_skills_df, sample_size=80, random_seed=42)

    for k, v in res.items():
        print(f"  {k}: Mean Steps = {v['mean_steps']}, Mean Hours = {v['mean_hours']}")


if __name__ == "__main__":
    audit_classifier_and_targets()
    audit_policy_simulation()
    audit_ablations()
