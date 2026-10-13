#!/usr/bin/env python3
"""
CLI Entry Point: DisasterM3 Dataset Validation & Integrity Checker

Usage:
    python scripts/validate_dataset.py --dataset data/disasterm3/disasterm3_qa.json
"""

import argparse
from pathlib import Path
import sys

# Ensure Backend/ is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.dataset_validator import DatasetValidator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate DisasterM3 dataset integrity and generate reproducible splits."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/disasterm3/disasterm3_qa.json",
        help="Path to DisasterM3 dataset JSON",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=".",
        help="Base directory for resolving relative image paths",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="data/reports/validation_report.json",
        help="Output path for validation report JSON",
    )
    parser.add_argument(
        "--no-check-images",
        action="store_true",
        help="Skip checking physical image file existence on disk",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Generate stratified train/dev/test splits",
    )
    parser.add_argument(
        "--split-dir",
        type=str,
        default="data/disasterm3/splits",
        help="Directory to save generated dataset splits",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splitting",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = (BACKEND_DIR / dataset_path).resolve()

    base_dir = Path(args.base_dir)
    if not base_dir.is_absolute():
        base_dir = (BACKEND_DIR / base_dir).resolve()

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = (BACKEND_DIR / report_path).resolve()

    print("=" * 70)
    print("DISASTERM3 DATASET VALIDATION")
    print("=" * 70)
    print(f"Dataset path : {dataset_path}")
    print(f"Base dir     : {base_dir}")
    print(f"Check images : {not args.no_check_images}")
    print("-" * 70)

    validator = DatasetValidator(base_dir=base_dir)

    try:
        valid_records, report = validator.load_and_validate(
            json_path=dataset_path,
            check_images_on_disk=not args.no_check_images,
        )
    except Exception as exc:
        print(f"\n[CRITICAL ERROR] Failed to load dataset: {exc}", file=sys.stderr)
        sys.exit(1)

    # Save validation report
    saved_report = validator.save_report(report, report_path)
    print(f"\nReport saved to: {saved_report}")

    print("\nSUMMARY METRICS:")
    print(f"  Total Records        : {report.total_records}")
    print(f"  Valid Records        : {report.valid_records}")
    print(f"  Invalid Records      : {report.invalid_records}")
    print(f"  Duplicate IDs        : {report.duplicate_ids_count}")
    print(f"  Missing Image Issues : {report.missing_images_count}")
    print(f"  Validation Status    : [{report.summary_status}]")

    print("\nTASK DISTRIBUTION:")
    for task_name, count in sorted(report.task_distribution.items(), key=lambda x: -x[1]):
        print(f"  - {task_name:<38}: {count}")

    print("\nIMAGE TYPE DISTRIBUTION:")
    for itype, count in sorted(report.image_type_distribution.items(), key=lambda x: -x[1]):
        print(f"  - {itype:<15}: {count}")

    if report.issues:
        print(f"\nISSUES FOUND ({len(report.issues)} total, showing first 10):")
        for issue in report.issues[:10]:
            print(f"  [{issue.severity.value}] Index {issue.record_index} (ID: {issue.record_id}): {issue.issue_type} - {issue.message}")

    if args.split and report.summary_status == "PASS":
        print("\n" + "=" * 70)
        print("GENERATING STRATIFIED TRAIN / DEV / TEST SPLITS")
        print("=" * 70)
        split_dir = Path(args.split_dir)
        if not split_dir.is_absolute():
            split_dir = (BACKEND_DIR / split_dir).resolve()

        train, dev, test, summary = validator.split_dataset(
            records=valid_records,
            train_ratio=0.70,
            dev_ratio=0.15,
            test_ratio=0.15,
            seed=args.seed,
        )
        split_paths = validator.save_splits(train, dev, test, split_dir)
        print(f"  Train samples: {summary.train_count} -> {split_paths['train']}")
        print(f"  Dev samples  : {summary.dev_count} -> {split_paths['dev']}")
        print(f"  Test samples : {summary.test_count} -> {split_paths['test']}")
        print(f"  Splits saved to: {split_dir}")

    print("=" * 70)
    if report.summary_status == "PASS":
        print("VALIDATION SUCCESSFUL: All dataset integrity checks passed.")
        sys.exit(0)
    else:
        print("VALIDATION FAILED: Issues detected. Review the report.")
        sys.exit(1)


if __name__ == "__main__":
    main()
