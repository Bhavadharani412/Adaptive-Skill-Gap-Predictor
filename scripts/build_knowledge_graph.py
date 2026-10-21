import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.services.knowledge_graph import OccupationSkillKnowledgeGraph
from backend.app.core.config import settings


def main():
    parser = argparse.ArgumentParser(description="Construct Knowledge Graph and compute topological metrics.")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing occupations.csv, skills.csv, occupation_skill_edges.csv")
    parser.add_argument("--output-report", type=str, default=None, help="Path to output metrics JSON")
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else settings.DATA_SYNTHETIC_DIR
    out_report = Path(args.output_report) if args.output_report else (settings.REPORTS_DIR / "graph_metrics.json")
    out_report.parent.mkdir(parents=True, exist_ok=True)

    kg = OccupationSkillKnowledgeGraph()
    metrics = kg.build_from_directory(data_dir)

    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n================ KNOWLEDGE GRAPH METRICS ================")
    print(f"Data Source: {data_dir}")
    print(f"Total Nodes: {metrics['total_nodes']} ({metrics['occupation_nodes_count']} Occupations, {metrics['skill_nodes_count']} Skills)")
    print(f"Total Edges: {metrics['total_edges']}")
    print(f"Generic Skills: {metrics['generic_skills_count']}")
    print(f"Clusters: {metrics['num_clusters']}")
    print(f"Mean Occupation Degree: {metrics['mean_occupation_degree']}")
    print(f"Mean Skill Degree: {metrics['mean_skill_degree']}")
    print(f"Graph Density: {metrics['graph_density']}")
    print(f"Connected Components: {metrics['connected_components_count']}")
    print(f"Report Saved to: {out_report}")
    print("=========================================================\n")


if __name__ == "__main__":
    main()
