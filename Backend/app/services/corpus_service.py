"""
Evidence Corpus Service: Standardized multimodal evidence record generation.

Features:
- Deterministic, collision-resistant evidence ID generation
- Clean provenance separation (dataset QA vs verified description vs external report)
- Explicit image role tracking (pre-disaster vs post-disaster vs observation)
- Disaster type and event recognition from verified sources without fabrication
- Deduplication and corpus statistics aggregation
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.schemas.evidence import (
    CorpusStatistics,
    EvidenceImageReference,
    EvidenceRecord,
    EvidenceSourceType,
    ImageRole,
)


def generate_stable_evidence_id(key: str) -> str:
    """Generate a stable, deterministic evidence identifier based on content hash."""
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    return f"evi_{h}"


def extract_disaster_metadata(image_path: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract known disaster event and type from standardized DisasterM3 filenames.
    Returns (disaster_type, event_name). Never fabricates unverified data.
    """
    if not image_path:
        return None, None

    filename = Path(image_path).name.lower()

    event_patterns = [
        (r"tornado", "Tornado"),
        (r"earthquake", "Earthquake"),
        (r"flood|flooding", "Flood"),
        (r"fire|wildfire|bushfire", "Wildfire"),
        (r"tsunami", "Tsunami"),
        (r"hurricane", "Hurricane"),
        (r"volcano", "Volcano"),
        (r"explosion", "Explosion"),
        (r"conflict", "Conflict"),
    ]

    disaster_type = None
    for pattern, d_name in event_patterns:
        if re.search(pattern, filename):
            disaster_type = d_name
            break

    # Extract clean event name only if a disaster type was recognized
    event_name = None
    if disaster_type:
        stem = Path(filename).stem.lower()
        # Remove temporal markers and trailing suffixes
        cleaned = re.sub(r"_(pre|post|sar|disaster).*$", "", stem)
        cleaned = re.sub(r"[_\d]+$", "", cleaned)
        if cleaned:
            event_name = cleaned.replace("_", " ").title()

    return disaster_type, event_name


class EvidenceCorpusGenerator:
    """Deterministic generator for the standardized DisasterMind evidence corpus."""

    def __init__(self, base_dir: Optional[Path | str] = None) -> None:
        if base_dir:
            self.base_dir = Path(base_dir).resolve()
        else:
            cwd = Path.cwd().resolve()
            if (cwd / "data").exists():
                self.base_dir = cwd
            elif (cwd / "Backend" / "data").exists():
                self.base_dir = cwd / "Backend"
            else:
                self.base_dir = cwd

    def create_evidence_from_qa_record(
        self,
        record: Dict[str, Any],
        source_reference: str = "DisasterM3_QA",
    ) -> EvidenceRecord:
        """Transform a DisasterM3 QA record into a standardized evidence item."""
        record_id = str(record.get("id", ""))
        task = record.get("task", "")
        question = record.get("question", "").strip()
        answer = str(record.get("answer", "")).strip()
        desc = record.get("description")
        modality = record.get("image_type")

        # Deterministic text formulation
        text_parts = [
            f"Task: {task}",
            f"Question: {question}",
            f"Observed Historical Answer: {answer}",
        ]
        if desc:
            text_parts.append(f"Context: {desc.strip()}")

        text_content = "\n".join(text_parts).strip()

        # Build image references
        images: List[EvidenceImageReference] = []
        disaster_type = None
        event_name = None

        pre_img = record.get("pre_image")
        if pre_img:
            images.append(
                EvidenceImageReference(
                    image_path=str(pre_img),
                    role=ImageRole.PRE_DISASTER,
                    modality=modality,
                )
            )
            d_type, ev_name = extract_disaster_metadata(pre_img)
            disaster_type = disaster_type or d_type
            event_name = event_name or ev_name

        post_img = record.get("post_image")
        if post_img:
            images.append(
                EvidenceImageReference(
                    image_path=str(post_img),
                    role=ImageRole.POST_DISASTER,
                    modality=modality,
                )
            )
            d_type, ev_name = extract_disaster_metadata(post_img)
            disaster_type = disaster_type or d_type
            event_name = event_name or ev_name

        single_img = record.get("image")
        if single_img:
            images.append(
                EvidenceImageReference(
                    image_path=str(single_img),
                    role=ImageRole.OBSERVATION,
                    modality=modality,
                )
            )
            d_type, ev_name = extract_disaster_metadata(single_img)
            disaster_type = disaster_type or d_type
            event_name = event_name or ev_name

        evidence_id = generate_stable_evidence_id(f"{record_id}_{text_content}")

        return EvidenceRecord(
            evidence_id=evidence_id,
            record_id=record_id or None,
            text=text_content,
            task_type=task or None,
            question=question or None,
            answer=answer or None,
            images=images,
            disaster_type=disaster_type,
            location=event_name,  # Verified event name from filename metadata
            source_type=EvidenceSourceType.DATASET_QA,
            source_reference=source_reference,
            observed_at=None,  # Not fabricated
            published_at=None,  # Not fabricated
            metadata={
                "image_modality": modality,
                "options": record.get("options"),
                "answer_option": record.get("answer_option"),
            },
        )

    def create_evidence_from_retrieval_item(
        self,
        item: Dict[str, Any],
        source_reference: str = "DisasterM3_Bench",
    ) -> EvidenceRecord:
        """Transform a retrieval corpus item into a standardized evidence item."""
        retrieval_id = str(item.get("id", ""))
        task = item.get("task", "")
        question = item.get("question", "").strip()
        answer = str(item.get("answer", "")).strip()

        text_content = f"Task: {task}\nQuestion: {question}\nHistorical answer: {answer}".strip()
        evidence_id = generate_stable_evidence_id(f"{retrieval_id}_{text_content}")

        return EvidenceRecord(
            evidence_id=evidence_id,
            record_id=retrieval_id or None,
            text=text_content,
            task_type=task or None,
            question=question or None,
            answer=answer or None,
            images=[],
            disaster_type=None,
            location=None,
            source_type=EvidenceSourceType.VERIFIED_DESCRIPTION,
            source_reference=source_reference,
            observed_at=None,
            published_at=None,
            metadata={"original_source": item.get("source", "DisasterM3")},
        )

    def build_corpus(
        self,
        qa_records: List[Dict[str, Any]],
        retrieval_records: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[EvidenceRecord], CorpusStatistics]:
        """Generate deduplicated, standardized multimodal evidence corpus."""
        corpus: List[EvidenceRecord] = []
        seen_texts: Set[str] = set()

        # 1. Process QA records
        for r in qa_records:
            evidence = self.create_evidence_from_qa_record(r)
            if evidence.text not in seen_texts:
                seen_texts.add(evidence.text)
                corpus.append(evidence)

        # 2. Process retrieval records if provided
        if retrieval_records:
            for r in retrieval_records:
                evidence = self.create_evidence_from_retrieval_item(r)
                if evidence.text not in seen_texts:
                    seen_texts.add(evidence.text)
                    corpus.append(evidence)

        # Compute statistics
        source_counts = Counter(e.source_type.value for e in corpus)
        task_counts = Counter(e.task_type or "Unknown" for e in corpus)
        disaster_counts = Counter(e.disaster_type or "Unspecified" for e in corpus)
        modality_counts = Counter()
        total_images = 0

        for e in corpus:
            total_images += len(e.images)
            for img in e.images:
                if img.modality:
                    modality_counts[img.modality] += 1

        stats = CorpusStatistics(
            total_records=len(corpus),
            source_type_distribution=dict(source_counts),
            task_distribution=dict(task_counts),
            disaster_type_distribution=dict(disaster_counts),
            modality_distribution=dict(modality_counts),
            total_images_referenced=total_images,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

        return corpus, stats

    def save_corpus(
        self,
        corpus: List[EvidenceRecord],
        output_path: Path | str,
    ) -> Path:
        """Persist evidence corpus to JSON file."""
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(
                [e.model_dump() for e in corpus],
                f,
                indent=2,
                ensure_ascii=False,
            )
        return out

    def save_statistics(
        self,
        stats: CorpusStatistics,
        output_path: Path | str,
    ) -> Path:
        """Persist corpus statistics to JSON file."""
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(stats.model_dump_json(indent=2))
        return out
