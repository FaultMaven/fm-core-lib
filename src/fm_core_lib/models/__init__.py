"""
Shared data models for FaultMaven microservices.

This package provides Pydantic models that are shared across all FaultMaven
microservices to ensure consistency and avoid duplication.
"""

from fm_core_lib.models.case import (
    # Core case model
    Case,
    CaseStatus,
    
    # Investigation components
    InvestigationProgress,
    ConsultingData,
    ProblemConfirmation,
    ProblemVerification,
    PathSelection,
    WorkingConclusion,
    RootCauseConclusion,
    
    # Evidence and data
    Evidence,
    UploadedFile,
    
    # Hypotheses
    Hypothesis,
    HypothesisEvidenceLink,
    
    # Solutions
    Solution,
    
    # Turn tracking
    TurnProgress,
    CaseStatusTransition,
    
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
    "Case", "CaseStatus",
    "InvestigationProgress", "ConsultingData", "ProblemConfirmation",
    "ProblemVerification", "PathSelection", "WorkingConclusion",
    "RootCauseConclusion", "Evidence", "UploadedFile",
    "Hypothesis", "HypothesisEvidenceLink", "Solution",
    "TurnProgress", "CaseStatusTransition",
    "DegradedMode", "DegradedModeType", "EscalationState",
    "DocumentationData", "GeneratedDocument",
    "Change", "Correlation",
]
