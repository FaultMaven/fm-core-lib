"""Resilience utilities for FaultMaven services.

This module provides standard retry policies and connection verification helpers
for handling transient failures in distributed systems (K8s, scale-to-zero, etc.).
"""

import logging
from typing import Any, Callable, TypeVar

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    RetryCallState,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _log_retry_attempt(retry_state: RetryCallState) -> None:
    """Custom retry logging with deployment context."""
    if retry_state.attempt_number > 1:
        logger.warning(
            f"[Resilience] Retry attempt {retry_state.attempt_number} for "
            f"{retry_state.fn.__name__} after {retry_state.seconds_since_start:.1f}s. "
            f"Exception: {retry_state.outcome.exception() if retry_state.outcome else 'Unknown'}"
        )


# Standard retry policy for service startup connections
# - Wait 2^x * 1 seconds between retries (2s, 4s, 8s, 16s, 32s)
# - Stop after 5 attempts (total ~62s wait time)
# - Log warnings before sleeping (visible in K8s logs)
# - Re-raise the exception if all retries fail
service_startup_retry = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=32),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


def create_custom_retry(
    max_attempts: int = 5,
    min_wait: int = 2,
    max_wait: int = 32,
    multiplier: int = 1,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Create a custom retry decorator with specific parameters.

    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        multiplier: Exponential backoff multiplier

    Returns:
        A retry decorator configured with the specified parameters

    Example:
        ```python
        # Custom retry for external API calls (more attempts, shorter waits)
        api_retry = create_custom_retry(max_attempts=10, min_wait=1, max_wait=10)

        @api_retry
        async def call_external_api():
            # ... API call logic
            pass
        ```
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=multiplier, min=min_wait, max=max_wait),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
