"""
Unit tests for Topic 03: Evidence Corpus Service and Schemas.
"""

import hashlib
import json
from pathlib import Path
import pytest

from app.schemas.evidence import (
    CorpusStatistics,
    EvidenceImageReference,
    EvidenceRecord,
    EvidenceSourceType,
    ImageRole,
)
from app.services.corpus_service import (
    EvidenceCorpusGenerator,
    extract_disaster_metadata,
    generate_stable_evidence_id,
)


class TestEvidenceCorpus:
    """Test suite for evidence corpus generation and schemas."""

    def test_deterministic_evidence_id(self):
        id1 = generate_stable_evidence_id("sample text content")
        id2 = generate_stable_evidence_id("sample text content")
        id3 = generate_stable_evidence_id("different text content")

        assert id1 == id2
        assert id1 != id3
        assert id1.startswith("evi_")

    def test_extract_disaster_metadata(self):
        d_type, ev_name = extract_disaster_metadata("test_images/joplin_tornado_00000065_pre_disaster.png")
        assert d_type == "Tornado"
        assert ev_name == "Joplin Tornado"

        d_type, ev_name = extract_disaster_metadata("test_images/libya_flood_post_98.png")
        assert d_type == "Flood"
        assert ev_name == "Libya Flood"

        d_type, ev_name = extract_disaster_metadata("test_images/turkey_earthquake5_post_353.png")
        assert d_type == "Earthquake"
        assert ev_name == "Turkey Earthquake"

        # Unknown filename returns None without fabrication
        d_type, ev_name = extract_disaster_metadata("random_image_123.png")
        assert d_type is None
        assert ev_name is None

    def test_create_evidence_from_qa_record(self):
        record = {
            "id": "dm3_00001",
            "task": "Disaster Bearing Bodies Recognition",
            "question": "What categories of objects have sustained damage?",
            "answer": "Buildings, Roads, Forest",
            "image_type": "Optical",
            "pre_image": "data/disasterm3/raw/DisasterM3_Bench/test_images/joplin_tornado_00000065_pre_disaster.png",
            "post_image": "data/disasterm3/raw/DisasterM3_Bench/test_images/joplin_tornado_00000065_post_disaster.png",
            "description": "Counting damaged objects",
        }

        generator = EvidenceCorpusGenerator()
        evidence = generator.create_evidence_from_qa_record(record)

        assert isinstance(evidence, EvidenceRecord)
        assert evidence.record_id == "dm3_00001"
        assert evidence.source_type == EvidenceSourceType.DATASET_QA
        assert evidence.disaster_type == "Tornado"
        assert evidence.location == "Joplin Tornado"
        assert len(evidence.images) == 2
        assert evidence.images[0].role == ImageRole.PRE_DISASTER
        assert evidence.images[1].role == ImageRole.POST_DISASTER
        assert evidence.observed_at is None  # Non-fabricated
        assert evidence.published_at is None  # Non-fabricated
        assert "Buildings, Roads, Forest" in evidence.text

    def test_create_evidence_from_retrieval_item(self):
        item = {
            "id": "retrieval_00001",
            "task": "Building Damage Counting",
            "question": "How many buildings survived?",
            "answer": "12",
            "source": "DisasterM3",
        }

        generator = EvidenceCorpusGenerator()
        evidence = generator.create_evidence_from_retrieval_item(item)

        assert evidence.record_id == "retrieval_00001"
        assert evidence.source_type == EvidenceSourceType.VERIFIED_DESCRIPTION
        assert evidence.source_reference == "DisasterM3_Bench"
        assert len(evidence.images) == 0

    def test_corpus_deduplication(self):
        records = [
            {
                "id": "dm3_1",
                "task": "Task A",
                "question": "Identical question?",
                "answer": "Same answer",
            },
            {
                "id": "dm3_2",  # Different ID, identical content
                "task": "Task A",
                "question": "Identical question?",
                "answer": "Same answer",
            },
        ]

        generator = EvidenceCorpusGenerator()
        corpus, stats = generator.build_corpus(records)

        assert len(corpus) == 1
        assert stats.total_records == 1

    def test_source_dataset_remains_unmodified(self):
        qa_path = Path("data/disasterm3/disasterm3_qa.json")
        if not qa_path.exists():
            pytest.skip("Dataset file not available in cwd")

        def compute_sha256(p: Path) -> str:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            return h.hexdigest()

        hash_before = compute_sha256(qa_path)

        generator = EvidenceCorpusGenerator()
        with open(qa_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        generator.build_corpus(data[:10])

        hash_after = compute_sha256(qa_path)
        assert hash_before == hash_after, "Source dataset file was modified!"
