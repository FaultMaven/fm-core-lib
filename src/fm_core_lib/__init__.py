"""FaultMaven Core Library

Shared models, LLM infrastructure, and preprocessing logic for FaultMaven microservices.
"""

__version__ = "0.3.0"

# Export shared models
from fm_core_lib.models import (
    Case, CaseStatus, Evidence, Hypothesis, Solution,
    UploadedFile, InvestigationProgress, ConsultingData
)

# Export HTTP clients
from fm_core_lib.clients import CaseServiceClient

# Export service discovery
from fm_core_lib.discovery import (
    ServiceRegistry,
    DeploymentMode,
    get_service_registry,
    reset_service_registry,
)

__all__ = [
    # Models
    "Case", "CaseStatus", "Evidence", "Hypothesis", "Solution",
    "UploadedFile", "InvestigationProgress", "ConsultingData",
    # Clients
    "CaseServiceClient",
    # Service Discovery
    "ServiceRegistry",
    "DeploymentMode",
    "get_service_registry",
    "reset_service_registry",
]
