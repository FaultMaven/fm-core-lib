"""FaultMaven Core Library

Shared models, LLM infrastructure, and preprocessing logic for FaultMaven microservices.
"""

__version__ = "0.3.0"

# Export shared models first (no dependencies)
from fm_core_lib.models import (
    Case, CaseStatus, Evidence, Hypothesis, Solution,
    UploadedFile, InvestigationProgress, ConsultingData
)

# Export service discovery (no model dependencies)
from fm_core_lib.discovery import (
    ServiceRegistry,
    DeploymentMode,
    get_service_registry,
    reset_service_registry,
)

# Lazy import for clients to avoid circular dependency
# Clients depend on models, so import them last
def __getattr__(name):
    """Lazy import for CaseServiceClient to avoid circular import."""
    if name == "CaseServiceClient":
        from fm_core_lib.clients import CaseServiceClient
        return CaseServiceClient
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    # Models
    "Case", "CaseStatus", "Evidence", "Hypothesis", "Solution",
    "UploadedFile", "InvestigationProgress", "ConsultingData",
    # Clients (lazy loaded)
    "CaseServiceClient",
    # Service Discovery
    "ServiceRegistry",
    "DeploymentMode",
    "get_service_registry",
    "reset_service_registry",
]
