"""Service-to-service authentication library for FaultMaven microservices.

This module provides JWT-based authentication for FaultMaven microservices.
"""

from fm_core_lib.auth.service_auth import (
    ServiceIdentity,
    RequestContext,
    ServiceAuthMiddleware,
    get_request_context,
)
from fm_core_lib.auth.token_provider import ServiceTokenProvider

__all__ = [
    "ServiceIdentity",
    "RequestContext",
    "ServiceAuthMiddleware",
    "ServiceTokenProvider",
    "get_request_context",
]
