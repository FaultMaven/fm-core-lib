"""Service Discovery Module

Deployment-neutral service URL resolution for microservices communication.
"""

from .service_registry import (
    ServiceRegistry,
    DeploymentMode,
    get_service_registry,
    reset_service_registry,
)

__all__ = [
    "ServiceRegistry",
    "DeploymentMode",
    "get_service_registry",
    "reset_service_registry",
]
