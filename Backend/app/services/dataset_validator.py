"""DisasterM3 Dataset Validation and Splitting Service."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import ValidationError

from app.schemas.dataset import (
    DatasetSplitSummary,
    DisasterM3Record,
    IssueSeverity,
    ValidationIssue,
    ValidationReport,
)


class DatasetValidator:
    """Robust, non-destructive validator and splitter for DisasterM3 dataset."""

    def __init__(self, base_dir: Optional[Path | str] = None) -> None:
        """
        Initialize validator.
        :param base_dir: Base directory for resolving relative image paths.
                         Defaults to project root or Backend directory.
        """
        if base_dir:
            self.base_dir = Path(base_dir).resolve()
        else:
            # Fall back to current working directory or Backend parent
            cwd = Path.cwd().resolve()
            if (cwd / "data").exists():
                self.base_dir = cwd
            elif (cwd / "Backend" / "data").exists():
                self.base_dir = cwd / "Backend"
            else:
                self.base_dir = cwd

    def resolve_image_path(self, relative_path: Optional[str]) -> Optional[Path]:
        """Resolve a relative image path against base_dir."""
        if not relative_path:
            return None
        candidate = Path(relative_path)
        if candidate.is_absolute():
            return candidate
        return (self.base_dir / candidate).resolve()

    def validate_record(
        self,
        record_data: Any,
        index: int,
        seen_ids: Set[str],
        check_images_on_disk: bool = True,
    ) -> Tuple[Optional[DisasterM3Record], List[ValidationIssue]]:
        """
        Validate a single dictionary record against schema and business rules.
        Does not mutate the input dictionary.
        """
        issues: List[ValidationIssue] = []

        if not isinstance(record_data, dict):
            issues.append(
                ValidationIssue(
                    record_id=None,
                    record_index=index,
                    issue_type="INVALID_RECORD_TYPE",
                    message=f"Record at index {index} is {type(record_data).__name__}, expected dict",
                    severity=IssueSeverity.ERROR,
                )
            )
            return None, issues

        record_id = record_data.get("id")

        # 1. Check ID presence & duplicates
        if not record_id or not str(record_id).strip():
            issues.append(
                ValidationIssue(
                    record_id=None,
                    record_index=index,
                    field="id",
                    issue_type="MISSING_ID",
                    message="Missing or blank record ID",
                    severity=IssueSeverity.ERROR,
                )
            )
        else:
            record_id_str = str(record_id).strip()
            if record_id_str in seen_ids:
                issues.append(
                    ValidationIssue(
                        record_id=record_id_str,
                        record_index=index,
                        field="id",
                        issue_type="DUPLICATE_ID",
                        message=f"Duplicate ID '{record_id_str}' already seen in earlier record",
                        severity=IssueSeverity.ERROR,
                    )
                )
            else:
                seen_ids.add(record_id_str)

        # 2. Check question
        question = record_data.get("question")
        if not question or not str(question).strip():
            issues.append(
                ValidationIssue(
                    record_id=record_id,
                    record_index=index,
                    field="question",
                    issue_type="MISSING_QUESTION",
                    message="Missing, empty, or whitespace question",
                    severity=IssueSeverity.ERROR,
                )
            )

        # 3. Check answer
        answer = record_data.get("answer")
        if answer is None or not str(answer).strip():
            issues.append(
                ValidationIssue(
                    record_id=record_id,
                    record_index=index,
                    field="answer",
                    issue_type="MISSING_ANSWER",
                    message="Missing, empty, or whitespace ground truth answer",
                    severity=IssueSeverity.ERROR,
                )
            )

        # 4. Check image presence and file existence on disk
        has_image_ref = any(
            record_data.get(k) for k in ("image", "pre_image", "post_image")
        )
        if not has_image_ref:
            issues.append(
                ValidationIssue(
                    record_id=record_id,
                    record_index=index,
                    field="images",
                    issue_type="MISSING_IMAGE_REFERENCE",
                    message="No image path provided in record (image, pre_image, and post_image are null)",
                    severity=IssueSeverity.ERROR,
                )
            )
        elif check_images_on_disk:
            for img_key in ("pre_image", "post_image", "image"):
                path_val = record_data.get(img_key)
                if path_val:
                    resolved = self.resolve_image_path(str(path_val))
                    if resolved and not resolved.exists():
                        issues.append(
                            ValidationIssue(
                                record_id=record_id,
                                record_index=index,
                                field=img_key,
                                issue_type="MISSING_IMAGE_FILE",
                                message=f"Image referenced at '{path_val}' does not exist on disk (resolved: '{resolved}')",
                                severity=IssueSeverity.ERROR,
                            )
                        )

        # 5. Pydantic validation
        parsed_record: Optional[DisasterM3Record] = None
        try:
            parsed_record = DisasterM3Record(**record_data)
        except ValidationError as e:
            for err in e.errors():
                loc_str = ".".join(str(x) for x in err.get("loc", []))
                issues.append(
                    ValidationIssue(
                        record_id=record_id,
                        record_index=index,
                        field=loc_str or None,
                        issue_type="SCHEMA_VALIDATION_ERROR",
                        message=err.get("msg", "Invalid field"),
                        severity=IssueSeverity.ERROR,
                    )
                )

        return parsed_record, issues

    def validate_dataset(
        self,
        raw_records: List[Dict[str, Any]],
        dataset_source_path: str = "memory",
        check_images_on_disk: bool = True,
    ) -> Tuple[List[DisasterM3Record], ValidationReport]:
        """
        Validate full list of records and produce comprehensive report.
        Source data is left untouched.
        """
        seen_ids: Set[str] = set()
        valid_records: List[DisasterM3Record] = []
        all_issues: List[ValidationIssue] = []

        task_counter: Counter[str] = Counter()
        image_type_counter: Counter[str] = Counter()

        for idx, item in enumerate(raw_records):
            record, record_issues = self.validate_record(
                item,
                index=idx,
                seen_ids=seen_ids,
                check_images_on_disk=check_images_on_disk,
            )
            all_issues.extend(record_issues)

            if isinstance(item, dict):
                t = item.get("task")
                if t:
                    task_counter[str(t)] += 1
                it = item.get("image_type")
                if it:
                    image_type_counter[str(it)] += 1

            if record and not record_issues:
                valid_records.append(record)

        duplicate_ids_count = sum(1 for i in all_issues if i.issue_type == "DUPLICATE_ID")
        missing_images_count = sum(
            1 for i in all_issues if i.issue_type in ("MISSING_IMAGE_FILE", "MISSING_IMAGE_REFERENCE")
        )

        invalid_records_count = len(raw_records) - len(valid_records)
        status = "PASS" if (invalid_records_count == 0 and len(all_issues) == 0) else "FAIL"

        report = ValidationReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            dataset_path=str(dataset_source_path),
            total_records=len(raw_records),
            valid_records=len(valid_records),
            invalid_records=invalid_records_count,
            duplicate_ids_count=duplicate_ids_count,
            missing_images_count=missing_images_count,
            task_distribution=dict(task_counter),
            image_type_distribution=dict(image_type_counter),
            issues=all_issues,
            summary_status=status,
        )

        return valid_records, report

    def load_and_validate(
        self,
        json_path: Path | str,
        check_images_on_disk: bool = True,
    ) -> Tuple[List[DisasterM3Record], ValidationReport]:
        """Load JSON file from path and run validation."""
        path = Path(json_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if not isinstance(raw_data, list):
            raise ValueError(f"Expected top-level JSON list of records in {path}, got {type(raw_data).__name__}")

        return self.validate_dataset(
            raw_records=raw_data,
            dataset_source_path=str(path),
            check_images_on_disk=check_images_on_disk,
        )

    def split_dataset(
        self,
        records: List[DisasterM3Record],
        train_ratio: float = 0.70,
        dev_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple[List[DisasterM3Record], List[DisasterM3Record], List[DisasterM3Record], DatasetSplitSummary]:
        """
        Deterministically partition valid records into stratified train, dev, and test sets.
        Prevents data leakage across splits.
        """
        if not abs((train_ratio + dev_ratio + test_ratio) - 1.0) < 1e-5:
            raise ValueError("train_ratio + dev_ratio + test_ratio must sum to 1.0")

        # Group by task for stratification
        by_task: Dict[str, List[DisasterM3Record]] = defaultdict(list)
        for r in records:
            by_task[r.task].append(r)

        rng = random.Random(seed)

        train_set: List[DisasterM3Record] = []
        dev_set: List[DisasterM3Record] = []
        test_set: List[DisasterM3Record] = []

        for task_name, task_records in sorted(by_task.items()):
            # Sort deterministically by id before shuffle
            task_sorted = sorted(task_records, key=lambda x: x.id)
            rng.shuffle(task_sorted)

            n = len(task_sorted)
            n_train = int(round(n * train_ratio))
            n_dev = int(round(n * dev_ratio))
            # Put remainder in test
            train_subset = task_sorted[:n_train]
            dev_subset = task_sorted[n_train : n_train + n_dev]
            test_subset = task_sorted[n_train + n_dev :]

            train_set.extend(train_subset)
            dev_set.extend(dev_subset)
            test_set.extend(test_subset)

        # Verification of no ID leakage
        train_ids = {r.id for r in train_set}
        dev_ids = {r.id for r in dev_set}
        test_ids = {r.id for r in test_set}

        assert train_ids.isdisjoint(dev_ids), "Data leakage detected between train and dev sets"
        assert train_ids.isdisjoint(test_ids), "Data leakage detected between train and test sets"
        assert dev_ids.isdisjoint(test_ids), "Data leakage detected between dev and test sets"

        train_tasks = dict(Counter(r.task for r in train_set))
        dev_tasks = dict(Counter(r.task for r in dev_set))
        test_tasks = dict(Counter(r.task for r in test_set))

        summary = DatasetSplitSummary(
            total_samples=len(records),
            train_count=len(train_set),
            dev_count=len(dev_set),
            test_count=len(test_set),
            train_tasks=train_tasks,
            dev_tasks=dev_tasks,
            test_tasks=test_tasks,
            seed=seed,
        )

        return train_set, dev_set, test_set, summary

    def save_report(self, report: ValidationReport, output_path: Path | str) -> Path:
        """Save validation report to JSON file without modifying source dataset."""
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))
        return out

    def save_splits(
        self,
        train: List[DisasterM3Record],
        dev: List[DisasterM3Record],
        test: List[DisasterM3Record],
        output_dir: Path | str,
    ) -> Dict[str, Path]:
        """Save splits to separate JSON files."""
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        paths = {
            "train": out_dir / "train.json",
            "dev": out_dir / "dev.json",
            "test": out_dir / "test.json",
        }

        for split_name, dataset_subset in [("train", train), ("dev", dev), ("test", test)]:
            with open(paths[split_name], "w", encoding="utf-8") as f:
                json.dump(
                    [r.model_dump() for r in dataset_subset],
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

        return paths
