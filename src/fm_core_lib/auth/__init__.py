"""Authentication utilities for FaultMaven microservices.

This module provides request context extraction from API Gateway headers.
Services trust X-User-* headers from the gateway without JWT validation.
"""

from fm_core_lib.auth.request_context import RequestContext, get_request_context

__all__ = [
    "RequestContext",
    "get_request_context",
]
