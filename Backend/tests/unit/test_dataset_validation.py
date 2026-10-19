"""
Unit tests for Topic 01: DisasterM3 Dataset Validation and Splitting.
"""

import hashlib
import json
from pathlib import Path
import pytest

from app.schemas.dataset import DisasterM3Record, ValidationReport
from app.services.dataset_validator import DatasetValidator


@pytest.fixture
def sample_valid_records(tmp_path):
    """Create sample valid records with actual dummy images."""
    img1 = tmp_path / "img1.png"
    img2 = tmp_path / "img2.png"
    img1.write_bytes(b"\x89PNG\r\n\x1a\n")
    img2.write_bytes(b"\x89PNG\r\n\x1a\n")

    return [
        {
            "id": "dm3_00001",
            "task": "Building Damage Counting",
            "question": "How many buildings are destroyed?",
            "answer": "5",
            "image_type": "Optical",
            "pre_image": str(img1),
            "post_image": str(img2),
            "image": None,
            "options": [1, 3, 5, 8],
            "answer_option": "C",
            "description": "Counting building damage",
        },
        {
            "id": "dm3_00002",
            "task": "Disaster Scene Recognition",
            "question": "What type of disaster is shown?",
            "answer": "Wildfire",
            "image_type": "Optical",
            "pre_image": None,
            "post_image": None,
            "image": str(img1),
            "options": ["Flood", "Wildfire", "Earthquake"],
            "answer_option": "B",
            "description": None,
        },
    ]


class TestDatasetValidation:
    """Test suite for DatasetValidator service."""

    def test_valid_records_parsing(self, sample_valid_records, tmp_path):
        validator = DatasetValidator(base_dir=tmp_path)
        valid, report = validator.validate_dataset(
            sample_valid_records,
            check_images_on_disk=True,
        )

        assert len(valid) == 2
        assert report.valid_records == 2
        assert report.invalid_records == 0
        assert report.summary_status == "PASS"
        assert len(report.issues) == 0

        # Check options types parsed correctly
        assert valid[0].options == [1, 3, 5, 8]
        assert valid[1].options == ["Flood", "Wildfire", "Earthquake"]

    def test_missing_question_error(self, tmp_path):
        img = tmp_path / "img.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")

        invalid_record = {
            "id": "dm3_00010",
            "task": "Disaster Type Recognition",
            "question": "   ",  # whitespace only
            "answer": "Flood",
            "image": str(img),
        }

        validator = DatasetValidator(base_dir=tmp_path)
        valid, report = validator.validate_dataset([invalid_record])

        assert len(valid) == 0
        assert report.invalid_records == 1
        assert report.summary_status == "FAIL"

        issue_types = [i.issue_type for i in report.issues]
        assert "MISSING_QUESTION" in issue_types

    def test_missing_answer_error(self, tmp_path):
        img = tmp_path / "img.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")

        invalid_record = {
            "id": "dm3_00011",
            "task": "Disaster Type Recognition",
            "question": "What happened?",
            "answer": "",  # empty answer
            "image": str(img),
        }

        validator = DatasetValidator(base_dir=tmp_path)
        valid, report = validator.validate_dataset([invalid_record])

        assert len(valid) == 0
        assert report.invalid_records == 1
        issue_types = [i.issue_type for i in report.issues]
        assert "MISSING_ANSWER" in issue_types

    def test_duplicate_id_detection(self, tmp_path):
        img = tmp_path / "img.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")

        records = [
            {
                "id": "dm3_dup",
                "task": "Task A",
                "question": "Question 1?",
                "answer": "Ans 1",
                "image": str(img),
            },
            {
                "id": "dm3_dup",  # duplicate ID
                "task": "Task B",
                "question": "Question 2?",
                "answer": "Ans 2",
                "image": str(img),
            },
        ]

        validator = DatasetValidator(base_dir=tmp_path)
        valid, report = validator.validate_dataset(records)

        assert report.duplicate_ids_count == 1
        assert any(i.issue_type == "DUPLICATE_ID" for i in report.issues)

    def test_missing_image_file_reported(self, tmp_path):
        record_with_missing_img = {
            "id": "dm3_no_img_file",
            "task": "Disaster Type Recognition",
            "question": "Is this a storm?",
            "answer": "Yes",
            "image": str(tmp_path / "non_existent_file.png"),
        }

        validator = DatasetValidator(base_dir=tmp_path)
        valid, report = validator.validate_dataset([record_with_missing_img], check_images_on_disk=True)

        assert report.missing_images_count >= 1
        assert any(i.issue_type == "MISSING_IMAGE_FILE" for i in report.issues)

    def test_missing_image_reference(self, tmp_path):
        record_no_ref = {
            "id": "dm3_no_ref",
            "task": "Disaster Type Recognition",
            "question": "What is this?",
            "answer": "Earthquake",
            "image": None,
            "pre_image": None,
            "post_image": None,
        }

        validator = DatasetValidator(base_dir=tmp_path)
        valid, report = validator.validate_dataset([record_no_ref])

        assert any(i.issue_type == "MISSING_IMAGE_REFERENCE" for i in report.issues)

    def test_invalid_item_type(self, tmp_path):
        validator = DatasetValidator(base_dir=tmp_path)
        valid, report = validator.validate_dataset(["not a dictionary", 12345])

        assert report.total_records == 2
        assert report.invalid_records == 2
        assert all(i.issue_type == "INVALID_RECORD_TYPE" for i in report.issues)

    def test_stratified_splits_reproducibility_and_zero_leakage(self, tmp_path):
        img = tmp_path / "img.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")

        # Generate 100 dummy records across 2 tasks
        records = []
        for i in range(100):
            task = "Task A" if i < 60 else "Task B"
            records.append(
                DisasterM3Record(
                    id=f"rec_{i:04d}",
                    task=task,
                    question=f"Question {i}?",
                    answer=f"Answer {i}",
                    image=str(img),
                )
            )

        validator = DatasetValidator(base_dir=tmp_path)
        train1, dev1, test1, summary1 = validator.split_dataset(records, train_ratio=0.7, dev_ratio=0.15, test_ratio=0.15, seed=123)
        train2, dev2, test2, summary2 = validator.split_dataset(records, train_ratio=0.7, dev_ratio=0.15, test_ratio=0.15, seed=123)

        # Reproducibility check
        assert [r.id for r in train1] == [r.id for r in train2]
        assert [r.id for r in dev1] == [r.id for r in dev2]
        assert [r.id for r in test1] == [r.id for r in test2]

        # Total counts
        assert len(train1) + len(dev1) + len(test1) == 100

        # Disjoint leakage check
        s_train = {r.id for r in train1}
        s_dev = {r.id for r in dev1}
        s_test = {r.id for r in test1}
        assert s_train.isdisjoint(s_dev)
        assert s_train.isdisjoint(s_test)
        assert s_dev.isdisjoint(s_test)

    def test_save_splits_and_reports(self, sample_valid_records, tmp_path):
        validator = DatasetValidator(base_dir=tmp_path)
        valid, report = validator.validate_dataset(sample_valid_records)

        rep_path = validator.save_report(report, tmp_path / "report.json")
        assert rep_path.exists()

        with open(rep_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["summary_status"] == "PASS"
        assert data["total_records"] == 2

    def test_source_dataset_file_unmodified(self):
        """Verify that validation does not alter or mutate the source dataset file."""
        dataset_path = Path("data/disasterm3/disasterm3_qa.json")
        if not dataset_path.exists():
            pytest.skip("disasterm3_qa.json not in current working dir")

        def compute_sha256(p: Path) -> str:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            return h.hexdigest()

        hash_before = compute_sha256(dataset_path)

        validator = DatasetValidator(base_dir=".")
        valid, report = validator.load_and_validate(dataset_path, check_images_on_disk=False)

        hash_after = compute_sha256(dataset_path)

        assert hash_before == hash_after, "Source dataset file was modified during validation!"
        assert report.summary_status == "PASS"
        assert len(valid) == 1599
