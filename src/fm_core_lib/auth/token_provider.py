"""Service token provider with automatic caching and refresh."""

import asyncio
import logging
import time
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)


class ServiceTokenProvider:
    """Manages service-to-service JWT tokens with automatic caching and refresh.

    This class handles:
    1. Fetching service tokens from fm-auth-service
    2. Caching tokens with TTL-based expiration
    3. Automatic refresh before expiration (5-minute buffer)
    4. Thread-safe token access

    Usage:
        provider = ServiceTokenProvider(
            auth_service_url="http://fm-auth-service:8000",
            service_id="fm-agent-service",
            audience=["fm-case-service", "fm-knowledge-service"]
        )

        # Get current token (auto-refreshes if needed)
        token = await provider.get_token()

        # Use in service client
        headers = {"Authorization": f"Bearer {token}"}
    """

    def __init__(
        self,
        auth_service_url: str,
        service_id: str,
        audience: List[str],
        refresh_buffer_seconds: int = 300,  # Refresh 5 minutes before expiration
        timeout_seconds: float = 10.0,
    ):
        """Initialize token provider.

        Args:
            auth_service_url: Base URL of fm-auth-service (e.g., http://fm-auth-service:8000)
            service_id: This service's identifier (e.g., "fm-agent-service")
            audience: List of services this service needs to call
            refresh_buffer_seconds: Refresh token this many seconds before expiration
            timeout_seconds: HTTP request timeout
        """
        self.auth_service_url = auth_service_url.rstrip("/")
        self.service_id = service_id
        self.audience = audience
        self.refresh_buffer_seconds = refresh_buffer_seconds
        self.timeout_seconds = timeout_seconds

        # Token cache
        self._token: Optional[str] = None
        self._token_expires_at: float = 0
        self._lock = asyncio.Lock()

        logger.info(
            f"Initialized ServiceTokenProvider: service_id={service_id}, "
            f"audience={audience}, auth_url={auth_service_url}"
        )

    async def get_token(self) -> str:
        """Get current service token (fetches new token if expired or missing).

        This method is thread-safe and handles concurrent requests efficiently.

        Returns:
            Valid JWT service token

        Raises:
            httpx.HTTPError: If token request fails
        """
        now = time.time()

        # Check if token needs refresh (expired or within refresh buffer)
        if self._token is None or now >= (self._token_expires_at - self.refresh_buffer_seconds):
            async with self._lock:
                # Double-check after acquiring lock (another thread may have refreshed)
                now = time.time()
                if self._token is None or now >= (
                    self._token_expires_at - self.refresh_buffer_seconds
                ):
                    await self._refresh_token()

        if self._token is None:
            raise RuntimeError("Failed to obtain service token")

        return self._token

    async def _refresh_token(self) -> None:
        """Fetch new service token from auth service.

        Raises:
            httpx.HTTPError: If token request fails
        """
        logger.info(f"Refreshing service token for {self.service_id}")

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{self.auth_service_url}/api/v1/auth/service-token",
                    json={"service_id": self.service_id, "audience": self.audience},
                )
                response.raise_for_status()

                data = response.json()
                self._token = data["token"]
                self._token_expires_at = data["expires_at"]

                logger.info(
                    f"Successfully refreshed service token for {self.service_id} "
                    f"(expires in {int(self._token_expires_at - time.time())}s)"
                )

        except httpx.HTTPError as e:
            logger.error(f"Failed to refresh service token: {e}")
            raise

    def get_token_sync(self) -> str:
        """Synchronous wrapper for get_token() (for non-async contexts).

        WARNING: This creates a new event loop if none exists. Prefer async version.

        Returns:
            Valid JWT service token
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.get_token())

    async def invalidate_token(self) -> None:
        """Force token refresh on next get_token() call.

        Useful for handling 401 responses (token may have been revoked).
        """
        async with self._lock:
            logger.info(f"Invalidating cached token for {self.service_id}")
            self._token = None
            self._token_expires_at = 0

    @property
    def is_token_valid(self) -> bool:
        """Check if cached token is still valid (not expired).

        Returns:
            True if token exists and is not expired (with buffer)
        """
        if self._token is None:
            return False

        now = time.time()
        return now < (self._token_expires_at - self.refresh_buffer_seconds)
