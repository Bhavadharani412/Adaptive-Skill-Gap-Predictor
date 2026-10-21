import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.services.data_importer import ESCOImporter
from backend.app.core.config import settings


def main():
    parser = argparse.ArgumentParser(description="Import ESCO taxonomy or generate synthetic fallback dataset.")
    parser.add_argument("--esco-dir", type=str, default=None, help="Directory containing official ESCO CSV files")
    parser.add_argument("--force-fallback", action="store_true", help="Force synthetic fallback generation")
    args = parser.parse_args()

    esco_dir = Path(args.esco_dir) if args.esco_dir else None
    importer = ESCOImporter()
    report = importer.run_import(esco_dir=esco_dir, force_fallback=args.force_fallback)

    print("\n================ DATA IMPORT REPORT ================")
    print(f"Source: {report['source']}")
    print(f"Status: {report['status']}")
    print("Metrics:")
    for k, v in report.get("metrics", {}).items():
        print(f"  - {k}: {v}")
    if report.get("warnings"):
        print("Warnings:")
        for w in report["warnings"]:
            print(f"  ! {w}")
    if report.get("errors"):
        print("Errors:")
        for e in report["errors"]:
            print(f"  X {e}")
    print("===================================================\n")


if __name__ == "__main__":
    main()
