import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.services.synthetic_generator import SyntheticBaselineGenerator
from backend.app.core.config import settings


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic baseline dataset matching paper specifications.")
    parser.add_argument("--occupations", type=int, default=settings.NUM_OCCUPATIONS, help="Number of occupations (default: 40)")
    parser.add_argument("--skills", type=int, default=settings.NUM_SKILLS, help="Number of skills (default: 220)")
    parser.add_argument("--clusters", type=int, default=settings.NUM_CLUSTERS, help="Number of clusters (default: 8)")
    parser.add_argument("--generic-ratio", type=float, default=settings.GENERIC_SKILL_RATIO, help="Ratio of generic skills (default: 0.12)")
    parser.add_argument("--employees", type=int, default=settings.NUM_EMPLOYEES, help="Number of employees (default: 600)")
    parser.add_argument("--seed", type=int, default=settings.DEFAULT_RANDOM_SEED, help="Random seed (default: 42)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")

    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else settings.DATA_SYNTHETIC_DIR
    generator = SyntheticBaselineGenerator(
        output_dir=out_dir,
        num_occupations=args.occupations,
        num_skills=args.skills,
        num_clusters=args.clusters,
        generic_ratio=args.generic_ratio,
        num_employees=args.employees,
        random_seed=args.seed
    )

    metadata = generator.generate()

    print("\n================ SYNTHETIC DATASET GENERATION ================")
    print(f"Dataset Type: {metadata['dataset_type']}")
    print(f"Random Seed: {metadata['random_seed']}")
    print(f"Output Directory: {out_dir}")
    print("\nCounts & Distribution:")
    for k, v in metadata["counts"].items():
        print(f"  - {k}: {v}")
    print("\nAssumptions:")
    for a in metadata["assumptions_documentation"]:
        print(f"  * {a}")
    print("==============================================================\n")


if __name__ == "__main__":
    main()
