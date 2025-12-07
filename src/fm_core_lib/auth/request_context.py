"""Request context extraction from API Gateway headers.

This module provides utilities to extract user context from X-User-* headers
added by the API Gateway after JWT validation.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from fastapi import HTTPException, Request, status

logger = logging.getLogger(__name__)


@dataclass
class RequestContext:
    """User request context extracted from API Gateway headers.

    The API Gateway validates user JWTs and adds these headers after
    stripping any client-provided X-User-* headers to prevent injection.

    Attributes:
        user_id: User ID from X-User-ID header
        user_email: User email from X-User-Email header
        user_roles: User roles from X-User-Roles header (JSON array)
        correlation_id: Optional correlation ID for request tracing
    """

    user_id: str
    user_email: Optional[str] = None
    user_roles: List[str] = field(default_factory=list)
    correlation_id: Optional[str] = None


def get_request_context(request: Request) -> RequestContext:
    """Extract request context from API Gateway headers.

    This function extracts user context from X-User-* headers that are
    added by the API Gateway after JWT validation. Services trust these
    headers without additional validation.

    Args:
        request: FastAPI request object

    Returns:
        RequestContext with user information

    Raises:
        HTTPException: If required X-User-ID header is missing
    """
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        logger.error("Missing X-User-ID header in request")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-User-ID header required (should be added by API Gateway)",
        )

    user_email = request.headers.get("X-User-Email")
    correlation_id = request.headers.get("X-Correlation-ID")

    # Parse roles from JSON array
    user_roles = []
    roles_header = request.headers.get("X-User-Roles")
    if roles_header:
        try:
            user_roles = json.loads(roles_header)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse X-User-Roles header: {roles_header}")

    return RequestContext(
        user_id=user_id,
        user_email=user_email,
        user_roles=user_roles,
        correlation_id=correlation_id,
    )
