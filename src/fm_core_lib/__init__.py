"""FaultMaven Core Library

Shared models, LLM infrastructure, and preprocessing logic for FaultMaven microservices.
"""

__version__ = "0.2.0"

# Export shared models
from fm_core_lib.models import (
    Case, CaseStatus, Evidence, Hypothesis, Solution,
    UploadedFile, InvestigationProgress, ConsultingData
)

# Export HTTP clients
from fm_core_lib.clients import CaseServiceClient

__all__ = [
    "Case", "CaseStatus", "Evidence", "Hypothesis", "Solution",
    "UploadedFile", "InvestigationProgress", "ConsultingData",
    "CaseServiceClient",
]
