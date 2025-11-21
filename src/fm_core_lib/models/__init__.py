"""
Shared data models for FaultMaven microservices.

This package provides Pydantic models that are shared across all FaultMaven
microservices to ensure consistency and avoid duplication.
"""

from fm_core_lib.models.case import (
    # Core case model
    Case,
    CaseStatus,
    CaseStatusTransition,

    # Investigation components
    InvestigationProgress,
    InvestigationStage,
    InvestigationPath,
    ConsultingData,
    ProblemConfirmation,
    ProblemVerification,
    PathSelection,
    WorkingConclusion,
    RootCauseConclusion,
    TemporalState,
    UrgencyLevel,
    ConfidenceLevel,

    # Evidence and data
    Evidence,
    UploadedFile,
    EvidenceCategory,
    EvidenceSourceType,
    EvidenceForm,
    EvidenceStance,

    # Hypotheses
    Hypothesis,
    HypothesisEvidenceLink,
    HypothesisCategory,
    HypothesisStatus,
    HypothesisGenerationMode,

    # Solutions
    Solution,
    SolutionType,

    # Turn tracking
    TurnProgress,
    TurnOutcome,

    # Special states
    DegradedMode,
    DegradedModeType,
    EscalationState,

    # Documentation
    DocumentationData,
    GeneratedDocument,

    # Supporting models
    Change,
    Correlation,
)

__all__ = [
    # Core case
    "Case", "CaseStatus", "CaseStatusTransition",
    # Investigation
    "InvestigationProgress", "InvestigationStage", "InvestigationPath",
    "ConsultingData", "ProblemConfirmation", "ProblemVerification",
    "PathSelection", "WorkingConclusion", "RootCauseConclusion",
    "TemporalState", "UrgencyLevel", "ConfidenceLevel",
    # Evidence
    "Evidence", "UploadedFile", "EvidenceCategory", "EvidenceSourceType",
    "EvidenceForm", "EvidenceStance",
    # Hypotheses
    "Hypothesis", "HypothesisEvidenceLink", "HypothesisCategory",
    "HypothesisStatus", "HypothesisGenerationMode",
    # Solutions
    "Solution", "SolutionType",
    # Turn tracking
    "TurnProgress", "TurnOutcome",
    # Special states
    "DegradedMode", "DegradedModeType", "EscalationState",
    # Documentation
    "DocumentationData", "GeneratedDocument",
    # Supporting
    "Change", "Correlation",
]
