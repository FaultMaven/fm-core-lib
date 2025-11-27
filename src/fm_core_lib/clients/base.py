"""Base service client with service-to-service authentication."""

import logging
from typing import Callable, Optional

import httpx

logger = logging.getLogger(__name__)


class BaseServiceClient:
    """Base class for internal service-to-service HTTP clients.

    This class provides:
    1. Automatic service JWT authentication via Authorization header
    2. User context propagation via X-User-ID header
    3. Correlation ID propagation via X-Correlation-ID header
    4. Consistent timeout and error handling

    Usage:
        class CaseServiceClient(BaseServiceClient):
            def __init__(
                self,
                base_url: str,
                token_provider: ServiceTokenProvider
            ):
                super().__init__(
                    base_url=base_url,
                    token_provider=token_provider.get_token
                )

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
        token_provider: Callable[[], str],
        timeout: float = 30.0,
    ):
        """Initialize service client.

        Args:
            base_url: Service base URL (e.g., http://fm-case-service:8003)
            token_provider: Async function that returns current service JWT
            timeout: Request timeout in seconds (default: 30.0)
        """
        self.base_url = base_url.rstrip("/")
        self.token_provider = token_provider
        self.timeout = timeout

        logger.info(f"Initialized {self.__class__.__name__} with base_url={base_url}")

    def _headers(
        self, user_id: Optional[str] = None, correlation_id: Optional[str] = None
    ) -> dict:
        """Generate request headers with service authentication and user context.

        Args:
            user_id: Optional user ID for user context propagation
            correlation_id: Optional correlation ID for request tracing

        Returns:
            Headers dict with Authorization, X-User-ID, and X-Correlation-ID
        """
        headers = {
            "Content-Type": "application/json",
        }

        # Add service authentication token
        # Note: token_provider may be async, handle both cases
        try:
            import asyncio
            import inspect

            if inspect.iscoroutinefunction(self.token_provider):
                # Async token provider - get event loop
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # We're already in an async context, can't use run_until_complete
                        # This shouldn't happen in practice since we're called from async methods
                        raise RuntimeError(
                            "Cannot call async token_provider from sync context within event loop"
                        )
                    token = loop.run_until_complete(self.token_provider())
                except RuntimeError:
                    # No event loop or already running - create task instead
                    # This is a fallback, should not happen in normal usage
                    logger.warning(
                        "Attempting to get token in running event loop - this may cause issues"
                    )
                    token = asyncio.create_task(self.token_provider())
            else:
                # Sync token provider
                token = self.token_provider()

            headers["Authorization"] = f"Bearer {token}"

        except Exception as e:
            logger.error(f"Failed to get service token: {e}")
            raise RuntimeError(f"Failed to obtain service authentication token: {e}")

        # Add user context if provided
        if user_id:
            headers["X-User-ID"] = user_id

        # Add correlation ID if provided
        if correlation_id:
            headers["X-Correlation-ID"] = correlation_id

        return headers

    async def _headers_async(
        self, user_id: Optional[str] = None, correlation_id: Optional[str] = None
    ) -> dict:
        """Async version of _headers for use with async token providers.

        Prefer this method when using async token providers (ServiceTokenProvider.get_token).

        Args:
            user_id: Optional user ID for user context propagation
            correlation_id: Optional correlation ID for request tracing

        Returns:
            Headers dict with Authorization, X-User-ID, and X-Correlation-ID
        """
        headers = {
            "Content-Type": "application/json",
        }

        # Get service authentication token (async)
        try:
            import inspect

            if inspect.iscoroutinefunction(self.token_provider):
                token = await self.token_provider()
            else:
                token = self.token_provider()

            headers["Authorization"] = f"Bearer {token}"

        except Exception as e:
            logger.error(f"Failed to get service token: {e}")
            raise RuntimeError(f"Failed to obtain service authentication token: {e}")

        # Add user context if provided
        if user_id:
            headers["X-User-ID"] = user_id

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
