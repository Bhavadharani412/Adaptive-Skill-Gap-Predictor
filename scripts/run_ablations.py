import argparse
import json
import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.core.config import settings
from backend.app.services.knowledge_graph import OccupationSkillKnowledgeGraph
from backend.app.services.ablation_study import AblationStudyService


def main():
    parser = argparse.ArgumentParser(description="Run policy ablation and sensitivity experiments.")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing dataset CSVs")
    parser.add_argument("--sample-size", type=int, default=80, help="Cohort sample size (default: 80)")
    parser.add_argument("--output-json", type=str, default=None, help="Output path for ablation metrics JSON")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else settings.DATA_SYNTHETIC_DIR
    out_json = Path(args.output_json) if args.output_json else (settings.EXPERIMENTS_DIR / "results" / "ablation_results.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)

    print("\n--- 1. Initializing Knowledge Graph & Ablation Service ---")
    kg = OccupationSkillKnowledgeGraph()
    kg.build_from_directory(data_dir)

    modules_df = pd.read_csv(data_dir / "training_modules.csv")
    emp_df = pd.read_csv(data_dir / "employees.csv")
    emp_skills_df = pd.read_csv(data_dir / "employee_skills.csv")

    service = AblationStudyService(kg, modules_df=modules_df)

    print(f"\n--- 2. Running 5 Policy Ablations (sample={args.sample_size}) ---")
    ablation_res = service.run_policy_ablations(emp_df, emp_skills_df, sample_size=args.sample_size)
    for variant, metrics in ablation_res.items():
        print(f"  [{variant}] Mean Steps: {metrics['mean_steps']} (std: {metrics['std_steps']}) | Hours: {metrics['mean_hours']}h | Success: {metrics['success_rate']:.1%}")

    print("\n--- 3. Running Sensitivity Analysis: Cluster Transfer Strength (beta) ---")
    cluster_sens = service.run_sensitivity_cluster_strength(emp_df, emp_skills_df, sample_size=args.sample_size)
    for row in cluster_sens:
        print(f"  * Beta = {row['cluster_bonus_weight_beta']:.2f} -> Mean Steps: {row['mean_steps']} (std: {row['std_steps']})")

    print("\n--- 4. Running Sensitivity Analysis: Readiness Target Threshold ---")
    target_sens = service.run_sensitivity_readiness_target(emp_df, emp_skills_df, sample_size=args.sample_size)
    for row in target_sens:
        print(f"  * Target = {row['target_readiness']:.0%} -> Mean Steps: {row['mean_steps']} (std: {row['std_steps']})")

    print("\n--- 5. Rendering Publication-Grade Figures ---")
    service.generate_all_plots(ablation_res, cluster_sens, target_sens)

    # Save comprehensive results
    combined_data = {
        "policy_ablations": ablation_res,
        "sensitivity_cluster_strength": cluster_sens,
        "sensitivity_readiness_target": target_sens
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=2)

    # Also save CSV summary
    ablation_df = pd.DataFrame.from_dict(ablation_res, orient="index")
    ablation_df.to_csv(settings.EXPERIMENTS_DIR / "results" / "ablation_summary.csv")

    print(f"\nAll ablation results and figures saved to:\n  - JSON: {out_json}\n  - Plots: {service.plots_dir}\n")


if __name__ == "__main__":
    main()
