"""
Grounding and Citation Service for DisasterMind (Topic 08).

Analyzes generated model predictions against retrieved evidence to:
- Detect citations and evidence references.
- Decompose responses into atomic claims.
- Measure token/n-gram overlap against evidence snippets.
- Compute grounding confidence score and hallucination risk level.
- Format responses with structured inline citations.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from app.schemas.grounding import (
    Citation,
    GroundedClaim,
    GroundedPrediction,
    HallucinationRisk,
)
from app.schemas.rag import EvidenceSnippet


class GroundingService:
    """Service to evaluate and enforce factual grounding in disaster predictions."""

    def __init__(self, support_threshold: float = 0.25) -> None:
        self.support_threshold = support_threshold

    def decompose_claims(self, text: str) -> List[str]:
        """Split text into distinct sentence-level claims."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        claims = [s.strip() for s in sentences if len(s.strip()) > 3]
        return claims if claims else [text.strip()]

    def compute_overlap_score(self, claim: str, evidence_text: str) -> float:
        """Compute lexical word overlap between claim and evidence snippet."""
        claim_words = set(re.findall(r"\w+", claim.lower()))
        evidence_words = set(re.findall(r"\w+", evidence_text.lower()))

        # Remove stop words
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to",
            "for", "of", "and", "or", "that", "this", "it", "with", "as", "by",
            "from", "there", "be", "have", "has", "had", "not", "but"
        }
        claim_tokens = claim_words - stopwords
        evidence_tokens = evidence_words - stopwords

        if not claim_tokens:
            return 0.0

        intersection = claim_tokens & evidence_tokens
        return len(intersection) / len(claim_tokens)

    def evaluate(
        self,
        prediction: str,
        evidence_snippets: List[EvidenceSnippet],
    ) -> GroundedPrediction:
        """Evaluate a prediction against retrieved evidence."""
        if not prediction or not prediction.strip():
            return GroundedPrediction(
                original_prediction="",
                grounded_prediction="",
                citations=[],
                claims=[],
                grounding_score=0.0,
                hallucination_risk=HallucinationRisk.HIGH,
                evidence_count=len(evidence_snippets),
            )

        if not evidence_snippets:
            return GroundedPrediction(
                original_prediction=prediction,
                grounded_prediction=prediction,
                citations=[],
                claims=[
                    GroundedClaim(
                        claim_text=c,
                        is_supported=False,
                        confidence=0.0,
                        supporting_evidence_ids=[],
                    )
                    for c in self.decompose_claims(prediction)
                ],
                grounding_score=0.0,
                hallucination_risk=HallucinationRisk.HIGH,
                evidence_count=0,
            )

        claims_raw = self.decompose_claims(prediction)
        evaluated_claims: List[GroundedClaim] = []
        citations_map: dict[str, Citation] = {}
        total_confidence = 0.0

        for claim in claims_raw:
            best_score = 0.0
            matching_ids: List[str] = []

            for snippet in evidence_snippets:
                combined_evidence = f"{snippet.question} {snippet.answer}"
                score = self.compute_overlap_score(claim, combined_evidence)

                if score > best_score:
                    best_score = score

                if score >= self.support_threshold:
                    matching_ids.append(snippet.evidence_id)
                    if snippet.evidence_id not in citations_map:
                        citations_map[snippet.evidence_id] = Citation(
                            citation_id=f"cit_{len(citations_map) + 1}",
                            evidence_id=snippet.evidence_id,
                            rank=snippet.rank,
                            task_type=snippet.task_type,
                            source_type=snippet.source_type,
                            snippet_preview=(
                                snippet.answer[:80] + "..."
                                if len(snippet.answer) > 80
                                else snippet.answer
                            ),
                        )

            is_supported = best_score >= self.support_threshold
            total_confidence += min(best_score, 1.0)
            evaluated_claims.append(
                GroundedClaim(
                    claim_text=claim,
                    is_supported=is_supported,
                    confidence=round(min(best_score, 1.0), 3),
                    supporting_evidence_ids=matching_ids,
                )
            )

        avg_grounding_score = (
            total_confidence / len(claims_raw) if claims_raw else 0.0
        )
        avg_grounding_score = round(min(avg_grounding_score, 1.0), 3)

        if avg_grounding_score >= 0.5:
            risk = HallucinationRisk.LOW
        elif avg_grounding_score >= 0.2:
            risk = HallucinationRisk.MEDIUM
        else:
            risk = HallucinationRisk.HIGH

        # Build grounded prediction string with citations
        citation_list = list(citations_map.values())
        citation_tags = " ".join([f"[{c.citation_id}: {c.evidence_id}]" for c in citation_list])
        grounded_text = f"{prediction}\n\nSources: {citation_tags}" if citation_tags else prediction

        return GroundedPrediction(
            original_prediction=prediction,
            grounded_prediction=grounded_text,
            citations=citation_list,
            claims=evaluated_claims,
            grounding_score=avg_grounding_score,
            hallucination_risk=risk,
            evidence_count=len(evidence_snippets),
        )


_GROUNDING_SERVICE: Optional[GroundingService] = None


def get_grounding_service() -> GroundingService:
    global _GROUNDING_SERVICE
    if _GROUNDING_SERVICE is None:
        _GROUNDING_SERVICE = GroundingService()
    return _GROUNDING_SERVICE
