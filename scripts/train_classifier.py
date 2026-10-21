import argparse
import json
import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.core.config import settings
from backend.app.services.knowledge_graph import OccupationSkillKnowledgeGraph
from backend.app.services.gap_analyzer import SkillGapAnalyzer
from backend.app.services.ml_classifier import CriticalGapClassifierService


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate critical-gap ML classifiers.")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing dataset CSVs")
    parser.add_argument("--output-results", type=str, default=None, help="Output path for results JSON")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else settings.DATA_SYNTHETIC_DIR
    out_results = Path(args.output_results) if args.output_results else (settings.EXPERIMENTS_DIR / "results" / "classifier_results.json")
    out_results.parent.mkdir(parents=True, exist_ok=True)

    print("\n--- 1. Loading Graph and Data ---")
    kg = OccupationSkillKnowledgeGraph()
    kg.build_from_directory(data_dir)

    emp_df = pd.read_csv(data_dir / "employees.csv")
    emp_skills_df = pd.read_csv(data_dir / "employee_skills.csv")

    print(f"Loaded {len(emp_df)} employees and {len(emp_skills_df)} skill records.")

    print("\n--- 2. Extracting Gap Features (G_e = S_r \\ P_e) ---")
    analyzer = SkillGapAnalyzer(kg)
    gaps_df = analyzer.generate_dataset_gap_features(emp_df, emp_skills_df)
    print(f"Extracted {len(gaps_df)} total missing skill instances.")

    classifier_service = CriticalGapClassifierService(kg)

    print("\n--- 3. Phase 6: Paper Reproduction Experiment (70/30 split) ---")
    paper_results = classifier_service.run_paper_reproduction(gaps_df, random_seed=42)
    print("Paper Reproduction Results:")
    for m_name, m_metrics in paper_results["models"].items():
        print(f"  [{m_name}] Accuracy: {m_metrics['accuracy']} | F1: {m_metrics['f1_score']} | ROC-AUC: {m_metrics['roc_auc']}")
    print(f"RF Feature Importances: {paper_results['random_forest_feature_importance']}")
    print(f"Reference Paper RF F1: {paper_results['paper_reported_reference_rf_f1']}")

    print("\n--- 4. Phase 7: Improved Defensible Evaluation (GroupKFold + Multi-seed CIs) ---")
    improved_results = classifier_service.run_improved_evaluation(gaps_df, n_splits=5, seeds=[42, 101, 202, 303, 404])
    print("Improved Evaluation (95% CI):")
    for m_name, m_stats in improved_results["metrics_with_95_ci"].items():
        f1_stat = m_stats['f1']
        auc_stat = m_stats['roc_auc']
        print(f"  [{m_name}] F1: {f1_stat['mean']} (95% CI: [{f1_stat['ci_95_lower']}, {f1_stat['ci_95_upper']}]) | ROC-AUC: {auc_stat['mean']}")

    full_report = {
        "phase_6_paper_reproduction": paper_results,
        "phase_7_improved_evaluation": improved_results
    }

    with open(out_results, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"\nSaved complete results to: {out_results}\n")


if __name__ == "__main__":
    main()
