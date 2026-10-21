import argparse
import json
import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.core.config import settings
from backend.app.services.knowledge_graph import OccupationSkillKnowledgeGraph
from backend.app.services.recommendation_engine import RecommendationEngine
from backend.app.services.policy_evaluator import PolicyEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate Policies A (Random), B (Static), C (Adaptive) on matched cohort.")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing dataset CSVs")
    parser.add_argument("--cohort-size", type=int, default=150, help="Number of employees to evaluate (default: 150)")
    parser.add_argument("--target-readiness", type=float, default=settings.READINESS_TARGET, help="Target readiness (default: 0.80)")
    parser.add_argument("--budget", type=int, default=settings.DEFAULT_TRAINING_BUDGET, help="Max step budget (default: 50)")
    parser.add_argument("--output-results", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else settings.DATA_SYNTHETIC_DIR
    out_results = Path(args.output_results) if args.output_results else (settings.EXPERIMENTS_DIR / "results" / "policy_evaluation_results.json")
    out_results.parent.mkdir(parents=True, exist_ok=True)

    print("\n--- 1. Loading Knowledge Graph & Modules ---")
    kg = OccupationSkillKnowledgeGraph()
    kg.build_from_directory(data_dir)

    modules_df = pd.read_csv(data_dir / "training_modules.csv")
    engine = RecommendationEngine(kg, modules_df=modules_df)

    emp_df = pd.read_csv(data_dir / "employees.csv")
    emp_skills_df = pd.read_csv(data_dir / "employee_skills.csv")

    evaluator = PolicyEvaluator(
        kg=kg,
        engine=engine,
        target_readiness=args.target_readiness,
        max_budget=args.budget
    )

    print(f"\n--- 2. Simulating Matched Cohort (n={args.cohort_size}, Target={args.target_readiness:.0%}) ---")
    results = evaluator.evaluate_cohort(
        employees_df=emp_df,
        emp_skills_df=emp_skills_df,
        sample_size=args.cohort_size,
        random_seed=42
    )

    print("\n================ POLICY EVALUATION RESULTS ================")
    print(f"Target Readiness: {results['target_readiness']:.0%} | Max Budget: {results['max_budget']} steps\n")
    for p_name, p_stats in results["policies"].items():
        print(f"[{p_name.upper()} POLICY]")
        print(f"  - Mean Steps to Readiness : {p_stats['steps_mean']} (std: {p_stats['steps_std']}, median: {p_stats['steps_median']})")
        print(f"  - Success Rate (% reach) : {p_stats['success_rate']:.1%}")
        print(f"  - Mean Learning Hours    : {p_stats['mean_learning_hours']} hrs")
        print(f"  - Mean Final Readiness   : {p_stats['mean_final_readiness']:.1%}")
        print(f"  - Censored (unsuccessful): {p_stats['censored_count']}\n")

    print("--- Paired Wilcoxon Signed-Rank Significance Tests ---")
    for comp_name, test_data in results["paired_statistical_tests"].items():
        sig_str = "SIGNIFICANT (p < 0.05)" if test_data["significant_at_05"] else "NOT SIGNIFICANT"
        print(f"  * {comp_name.replace('_', ' ').title()}: p={test_data['p_value']:.2e}, Cohen's d={test_data['cohens_d']} [{sig_str}]")

    print("\n--- Paper Reference Comparison ---")
    for k, v in results["paper_references_comparison"].items():
        if k != "notes":
            print(f"  * {k}: {v}")

    with open(out_results, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {out_results}\n")


if __name__ == "__main__":
    main()
