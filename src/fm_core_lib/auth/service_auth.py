"""Service-to-service authentication middleware and context management."""

import logging
from dataclasses import dataclass
from typing import List, Optional

import jwt
from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

logger = logging.getLogger(__name__)


@dataclass
class ServiceIdentity:
    """Identity of the calling service extracted from JWT."""

    service_id: str
    permissions: List[str]

    def has_permission(self, permission: str) -> bool:
        """Check if service has a specific permission.

        Args:
            permission: Permission string (e.g., "case:read", "knowledge:write")

        Returns:
            True if service has the permission or wildcard "*" permission
        """
        return "*" in self.permissions or permission in self.permissions


@dataclass
class RequestContext:
    """Complete request context with service and user identity.

    Attributes:
        service: Identity of the calling service (from JWT)
        user_id: Optional user ID context (from X-User-ID header)
        correlation_id: Optional correlation ID for request tracing
    """

    service: ServiceIdentity
    user_id: Optional[str] = None
    correlation_id: Optional[str] = None


class ServiceAuthMiddleware(BaseHTTPMiddleware):
    """Middleware to verify service JWTs and extract request context.

    This middleware:
    1. Extracts and validates service JWT from Authorization header
    2. Extracts user context from X-User-ID header
    3. Extracts correlation ID from X-Correlation-ID header
    4. Populates request.state.context with ServiceIdentity and user context

    Usage:
        app.add_middleware(
            ServiceAuthMiddleware,
            public_key=public_key_pem,
            jwt_algorithm="RS256",
            jwt_audience="faultmaven-api",
            jwt_issuer="fm-auth-service"
        )
    """

    def __init__(
        self,
        app,
        public_key: str,
        jwt_algorithm: str = "RS256",
        jwt_audience: str = "faultmaven-api",
        jwt_issuer: str = "fm-auth-service",
        skip_paths: Optional[List[str]] = None,
    ):
        """Initialize middleware.

        Args:
            app: FastAPI application
            public_key: PEM-formatted RSA public key for JWT verification
            jwt_algorithm: JWT signing algorithm (default: RS256)
            jwt_audience: Expected JWT audience claim
            jwt_issuer: Expected JWT issuer claim
            skip_paths: List of paths to skip authentication (e.g., ["/health", "/metrics"])
        """
        super().__init__(app)
        self.public_key = public_key
        self.jwt_algorithm = jwt_algorithm
        self.jwt_audience = jwt_audience
        self.jwt_issuer = jwt_issuer
        self.skip_paths = skip_paths or ["/health", "/metrics", "/docs", "/openapi.json"]

    async def dispatch(self, request: Request, call_next) -> Response:
        """Process request and verify service authentication."""
        # Skip authentication for health checks and docs
        if request.url.path in self.skip_paths:
            return await call_next(request)

        # Extract Authorization header
        auth_header = request.headers.get("Authorization")

        if not auth_header:
            logger.warning(f"Missing Authorization header for {request.url.path}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing service authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if not auth_header.startswith("Bearer "):
            logger.warning(f"Invalid Authorization header format for {request.url.path}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Authorization header format (expected 'Bearer <token>')",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = auth_header[7:]  # Remove "Bearer " prefix

        try:
            # Verify JWT with public key
            payload = jwt.decode(
                token,
                self.public_key,
                algorithms=[self.jwt_algorithm],
                audience=self.jwt_audience,
                issuer=self.jwt_issuer,
                options={"verify_signature": True, "verify_exp": True},
            )

            # Extract service identity
            service_id = payload.get("service_id")
            if not service_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid service token: missing service_id claim",
                )

            permissions = payload.get("permissions", [])

            # Extract user context from headers
            user_id = request.headers.get("X-User-ID")
            correlation_id = request.headers.get("X-Correlation-ID")

            # Store in request state for route handlers
            request.state.context = RequestContext(
                service=ServiceIdentity(service_id=service_id, permissions=permissions),
                user_id=user_id,
                correlation_id=correlation_id,
            )

            logger.info(
                f"Authenticated service request: service={service_id}, "
                f"user={user_id}, correlation={correlation_id}, path={request.url.path}"
            )

        except jwt.ExpiredSignatureError:
            logger.warning(f"Expired service token for {request.url.path}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Service token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid service token for {request.url.path}: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid service token: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"},
            )

        response = await call_next(request)
        return response


def get_request_context(request: Request) -> RequestContext:
    """FastAPI dependency to extract request context from middleware.

    Usage in route:
        @router.get("/cases/{case_id}")
        async def get_case(
            case_id: str,
            context: RequestContext = Depends(get_request_context)
        ):
            # Verify permissions
            if not context.service.has_permission("case:read"):
                raise HTTPException(403, "Service lacks case:read permission")

            # Use user context
            user_id = context.user_id or "system"
            ...

    Args:
        request: FastAPI request object

    Returns:
        RequestContext with service identity and user context

    Raises:
        HTTPException: If authentication context is missing
    """
    if not hasattr(request.state, "context"):
        logger.error("Missing authentication context in request state")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication context (middleware not configured?)",
        )

    return request.state.context
