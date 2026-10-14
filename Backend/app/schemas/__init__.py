"""DisasterMind Schema Definitions."""

from app.schemas.dataset import (
    DisasterM3Record,
    ValidationIssue,
    ValidationReport,
    DatasetSplitSummary,
    TaskType,
)

from app.schemas.vlm import (
    VLMInferenceRequest,
    VLMInferenceResponse,
    VLMBenchmarkSampleResult,
    VLMBenchmarkReport,
)

from app.schemas.evidence import (
    EvidenceSourceType,
    ImageRole,
    EvidenceImageReference,
    EvidenceRecord,
    CorpusStatistics,
)

from app.schemas.embedding import (
    EmbeddingRepresentation,
    EmbeddingMetadata,
    EmbeddingBatchResult,
    EmbeddingBenchmarkReport,
)

from app.schemas.retrieval import (
    CollectionStatus,
    IndexedPoint,
    IndexBuildReport,
    RetrievalResult,
    RetrievalResponse,
)

from app.schemas.rag import (
    RAGRequest,
    EvidenceSnippet,
    RAGContext,
    RAGResponse,
)

from app.services.multimodal_service import (
    MultimodalRetrievalRequest,
    FusedEvidenceResult,
    MultimodalRetrievalResponse,
)

from app.schemas.grounding import (
    Citation,
    GroundedClaim,
    GroundedPrediction,
    HallucinationRisk,
)

from app.schemas.live_feed import (
    LiveDisasterFeedItem,
    LiveFeedIngestRequest,
    LiveFeedIngestResponse,
)

from app.schemas.verification import (
    VerificationVerdict,
    ClaimVerificationResult,
    VerificationReport,
)

__all__ = [
    "DisasterM3Record",
    "ValidationIssue",
    "ValidationReport",
    "DatasetSplitSummary",
    "TaskType",
    "VLMInferenceRequest",
    "VLMInferenceResponse",
    "VLMBenchmarkSampleResult",
    "VLMBenchmarkReport",
    "EvidenceSourceType",
    "ImageRole",
    "EvidenceImageReference",
    "EvidenceRecord",
    "CorpusStatistics",
    "EmbeddingRepresentation",
    "EmbeddingMetadata",
    "EmbeddingBatchResult",
    "EmbeddingBenchmarkReport",
    "CollectionStatus",
    "IndexedPoint",
    "IndexBuildReport",
    "RetrievalResult",
    "RetrievalResponse",
    "RAGRequest",
    "EvidenceSnippet",
    "RAGContext",
    "RAGResponse",
    "MultimodalRetrievalRequest",
    "FusedEvidenceResult",
    "MultimodalRetrievalResponse",
    "Citation",
    "GroundedClaim",
    "GroundedPrediction",
    "HallucinationRisk",
    "LiveDisasterFeedItem",
    "LiveFeedIngestRequest",
    "LiveFeedIngestResponse",
    "VerificationVerdict",
    "ClaimVerificationResult",
    "VerificationReport",
]


