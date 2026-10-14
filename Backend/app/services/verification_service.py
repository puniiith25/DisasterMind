"""
Evidence Verification and Contradiction Detection Service (Topic 10).

Verifies whether claims made in disaster assessments are:
- SUPPORTED by ground-truth evidence
- CONTRADICTED by evidence (e.g. conflicting numbers, opposing disaster types, negated damage)
- UNVERIFIED (insufficient data in retrieval corpus)
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from app.schemas.rag import EvidenceSnippet
from app.schemas.verification import (
    ClaimVerificationResult,
    VerificationReport,
    VerificationVerdict,
)


class EvidenceVerificationService:
    """Automated fact-checking engine for disaster assessments."""

    NEGATION_WORDS = {"no", "not", "none", "zero", "neither", "never", "without"}
    DISASTER_TYPES = {
        "earthquake", "flood", "hurricane", "tornado", "tsunami", "wildfire", "volcano", "explosion"
    }

    def extract_numbers(self, text: str) -> List[int]:
        """Extract integer counts mentioned in text."""
        return [int(n) for n in re.findall(r"\b\d+\b", text)]

    def extract_disasters(self, text: str) -> set[str]:
        """Extract disaster types mentioned."""
        words = set(re.findall(r"\w+", text.lower()))
        return words & self.DISASTER_TYPES

    def check_claim_against_snippet(
        self, claim: str, snippet: EvidenceSnippet
    ) -> Tuple[VerificationVerdict, float, str]:
        """Compare a claim against a single evidence snippet."""
        claim_lower = claim.lower()
        evidence_lower = f"{snippet.question} {snippet.answer}".lower()

        claim_disasters = self.extract_disasters(claim_lower)
        evidence_disasters = self.extract_disasters(evidence_lower)

        # Conflict check: Different disaster type explicitly claimed
        if claim_disasters and evidence_disasters and not (claim_disasters & evidence_disasters):
            return (
                VerificationVerdict.CONTRADICTED,
                0.85,
                f"Claim refers to {claim_disasters}, but evidence specifies {evidence_disasters}.",
            )

        # Conflict check: Negation contradiction
        claim_has_neg = bool(set(re.findall(r"\w+", claim_lower)) & self.NEGATION_WORDS)
        evidence_has_neg = bool(set(re.findall(r"\w+", evidence_lower)) & self.NEGATION_WORDS)

        # Numerical conflict check
        claim_nums = self.extract_numbers(claim)
        evidence_nums = self.extract_numbers(snippet.answer)
        if claim_nums and evidence_nums:
            if claim_nums[0] != evidence_nums[0]:
                # If both talk about damage or counts but with conflicting numbers
                if any(w in claim_lower for w in ["building", "road", "collapsed", "damaged", "destroyed"]):
                    return (
                        VerificationVerdict.CONTRADICTED,
                        0.90,
                        f"Numerical contradiction: Claim asserts {claim_nums[0]}, but evidence documents {evidence_nums[0]}.",
                    )

        # Positive alignment check
        claim_words = set(re.findall(r"\w+", claim_lower))
        evidence_words = set(re.findall(r"\w+", evidence_lower))
        stopwords = {"the", "a", "an", "is", "was", "are", "in", "at", "to", "of", "and"}
        overlap = (claim_words - stopwords) & (evidence_words - stopwords)

        if len(claim_words - stopwords) > 0:
            overlap_ratio = len(overlap) / len(claim_words - stopwords)
            if overlap_ratio >= 0.35 and claim_has_neg == evidence_has_neg:
                return (
                    VerificationVerdict.SUPPORTED,
                    round(overlap_ratio, 2),
                    f"Verified supported by evidence ({snippet.evidence_id}).",
                )

        return (
            VerificationVerdict.UNVERIFIED,
            0.1,
            "Insufficient overlap to establish verification.",
        )

    def verify(
        self, prediction: str, evidence_snippets: List[EvidenceSnippet]
    ) -> VerificationReport:
        """Run full verification on all claims in prediction."""
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", prediction.strip())
            if len(s.strip()) > 3
        ]
        if not sentences:
            sentences = [prediction.strip()] if prediction.strip() else []

        if not sentences or not evidence_snippets:
            return VerificationReport(
                total_claims=len(sentences),
                supported_count=0,
                contradicted_count=0,
                unverified_count=len(sentences),
                veracity_score=0.0,
                has_critical_contradiction=False,
                claim_results=[
                    ClaimVerificationResult(
                        claim=s,
                        verdict=VerificationVerdict.UNVERIFIED,
                        confidence=0.0,
                        matched_evidence_id=None,
                        explanation="No evidence available to verify statement.",
                    )
                    for s in sentences
                ],
            )

        results: List[ClaimVerificationResult] = []
        supported_count = 0
        contradicted_count = 0
        unverified_count = 0

        for claim in sentences:
            best_verdict = VerificationVerdict.UNVERIFIED
            best_conf = 0.0
            best_explanation = "No matching evidence found."
            best_evi_id = None

            for snip in evidence_snippets:
                verdict, conf, expl = self.check_claim_against_snippet(claim, snip)
                if verdict == VerificationVerdict.CONTRADICTED:
                    # Contradictions are high priority
                    best_verdict = verdict
                    best_conf = conf
                    best_explanation = expl
                    best_evi_id = snip.evidence_id
                    break
                elif verdict == VerificationVerdict.SUPPORTED and conf > best_conf:
                    best_verdict = verdict
                    best_conf = conf
                    best_explanation = expl
                    best_evi_id = snip.evidence_id

            if best_verdict == VerificationVerdict.SUPPORTED:
                supported_count += 1
            elif best_verdict == VerificationVerdict.CONTRADICTED:
                contradicted_count += 1
            else:
                unverified_count += 1

            results.append(
                ClaimVerificationResult(
                    claim=claim,
                    verdict=best_verdict,
                    confidence=best_conf,
                    matched_evidence_id=best_evi_id,
                    explanation=best_explanation,
                )
            )

        veracity_score = (
            round((supported_count - (contradicted_count * 1.5)) / len(sentences), 2)
            if sentences
            else 0.0
        )
        veracity_score = max(0.0, min(1.0, veracity_score))

        return VerificationReport(
            total_claims=len(sentences),
            supported_count=supported_count,
            contradicted_count=contradicted_count,
            unverified_count=unverified_count,
            veracity_score=veracity_score,
            has_critical_contradiction=contradicted_count > 0,
            claim_results=results,
        )


_VERIFICATION_SERVICE: Optional[EvidenceVerificationService] = None


def get_verification_service() -> EvidenceVerificationService:
    global _VERIFICATION_SERVICE
    if _VERIFICATION_SERVICE is None:
        _VERIFICATION_SERVICE = EvidenceVerificationService()
    return _VERIFICATION_SERVICE
