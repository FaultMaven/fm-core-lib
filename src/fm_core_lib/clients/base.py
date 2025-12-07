"""Base service client for internal service-to-service calls."""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class BaseServiceClient:
    """Base class for internal service-to-service HTTP clients.

    Services call each other directly without JWT authentication.
    User context is propagated via X-User-* headers.

    Usage:
        class CaseServiceClient(BaseServiceClient):
            def __init__(self, base_url: str):
                super().__init__(base_url=base_url)

            async def get_case(self, case_id: str, user_id: str) -> Case:
                async with self._get_client() as client:
                    response = await client.get(
                        f"{self.base_url}/api/v1/cases/{case_id}",
                        headers=self._headers(user_id=user_id)
                    )
                    response.raise_for_status()
                    return Case(**response.json())
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
    ):
        """Initialize service client.

        Args:
            base_url: Service base URL (e.g., http://fm-case-service:8000)
            timeout: Request timeout in seconds (default: 30.0)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        logger.info(f"Initialized {self.__class__.__name__} with base_url={base_url}")

    def _headers(
        self,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
        user_roles: Optional[list] = None,
        correlation_id: Optional[str] = None
    ) -> dict:
        """Generate request headers with user context.

        Args:
            user_id: User ID for X-User-ID header
            user_email: User email for X-User-Email header
            user_roles: User roles for X-User-Roles header
            correlation_id: Optional correlation ID for request tracing

        Returns:
            Headers dict with X-User-* headers and correlation ID
        """
        headers = {
            "Content-Type": "application/json",
        }

        # Add user context headers
        if user_id:
            headers["X-User-ID"] = user_id

        if user_email:
            headers["X-User-Email"] = user_email

        if user_roles:
            import json
            headers["X-User-Roles"] = json.dumps(user_roles)

        # Add correlation ID if provided
        if correlation_id:
            headers["X-Correlation-ID"] = correlation_id

        return headers

    def _get_client(self) -> httpx.AsyncClient:
        """Get HTTP client instance with configured timeout.

        Returns:
            Configured AsyncClient ready for use with async context manager
        """
        return httpx.AsyncClient(timeout=self.timeout)

    async def close(self):
        """Close any persistent connections.

        Override this if your client maintains a persistent httpx.AsyncClient.
        """
        pass
