"""FaultMaven Core Models"""

from fm_core_lib.models.case import (
    Case,
    CaseStatus,
    CaseStatusTransition,
    InvestigationProgress,
    InvestigationStrategy,
    MessageType,
    is_valid_transition,
)
from fm_core_lib.models.evidence import (
    Evidence,
    EvidenceType,
    EvidenceCategory,
)
from fm_core_lib.models.common import (
    Hypothesis,
    HypothesisStatus,
    ConfidenceLevel,
    Solution,
    SolutionStatus,
)
from fm_core_lib.models.api_models import (
    CreateCaseRequest,
    AddMessageRequest,
    AddEvidenceRequest,
    UpdateCaseStatusRequest,
)

__all__ = [
    # Case models
    "Case",
    "CaseStatus",
    "CaseStatusTransition",
    "InvestigationProgress",
    "InvestigationStrategy",
    "MessageType",
    "is_valid_transition",
    # Evidence models
    "Evidence",
    "EvidenceType",
    "EvidenceCategory",
    # Common models
    "Hypothesis",
    "HypothesisStatus",
    "ConfidenceLevel",
    "Solution",
    "SolutionStatus",
    # API models
    "CreateCaseRequest",
    "AddMessageRequest",
    "AddEvidenceRequest",
    "UpdateCaseStatusRequest",
]
