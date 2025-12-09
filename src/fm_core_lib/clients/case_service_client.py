"""HTTP client for fm-case-service."""

from typing import List, Optional

from fm_core_lib.clients.base import BaseServiceClient
from fm_core_lib.models import Case, Evidence


class CaseServiceClient(BaseServiceClient):
    """Async HTTP client for communicating with fm-case-service.

    This client provides a Pythonic interface to the fm-case-service REST API,
    handling serialization/deserialization and error handling. User context is
    propagated via X-User-* headers.

    Usage:
        client = CaseServiceClient(base_url="http://fm-case-service:8000")
        case = await client.get_case(case_id="case-123", user_id="user-456")
    """

    def __init__(
        self,
        base_url: str = "http://fm-case-service:8000",
        timeout: float = 30.0,
    ):
        """Initialize client.

        Args:
            base_url: Base URL of fm-case-service (default: http://fm-case-service:8000)
            timeout: Request timeout in seconds (default: 30.0)
        """
        super().__init__(base_url=base_url, timeout=timeout)

    async def get_case(
        self, case_id: str, user_id: str, correlation_id: Optional[str] = None
    ) -> Case:
        """Get case by ID.

        Args:
            case_id: Case identifier
            user_id: User ID for X-User-ID header
            correlation_id: Optional correlation ID for request tracing

        Returns:
            Case object

        Raises:
            httpx.HTTPStatusError: If case not found or other HTTP error
        """
        async with self._get_client() as client:
            response = await client.get(
                f"{self.base_url}/api/v1/cases/{case_id}",
                headers=self._headers(user_id=user_id, correlation_id=correlation_id),
            )
            response.raise_for_status()
            return Case(**response.json())

    async def create_case(
        self, case: Case, user_id: str, correlation_id: Optional[str] = None
    ) -> Case:
        """Create new case.

        Args:
            case: Case object to create
            user_id: User ID for X-User-ID header
            correlation_id: Optional correlation ID for request tracing

        Returns:
            Created case with generated ID
        """
        async with self._get_client() as client:
            response = await client.post(
                f"{self.base_url}/api/v1/cases",
                json=case.model_dump(mode='json', exclude_unset=True),
                headers=self._headers(user_id=user_id, correlation_id=correlation_id),
            )
            response.raise_for_status()
            return Case(**response.json())

    async def update_case(
        self, case_id: str, case: Case, user_id: str, correlation_id: Optional[str] = None
    ) -> Case:
        """Update existing case.

        Args:
            case_id: Case identifier
            case: Updated case object
            user_id: User ID for X-User-ID header
            correlation_id: Optional correlation ID for request tracing

        Returns:
            Updated case
        """
        async with self._get_client() as client:
            response = await client.put(
                f"{self.base_url}/api/v1/cases/{case_id}",
                json=case.model_dump(mode='json'),
                headers=self._headers(user_id=user_id, correlation_id=correlation_id),
            )
            response.raise_for_status()
            return Case(**response.json())

    async def delete_case(
        self, case_id: str, user_id: str, correlation_id: Optional[str] = None
    ) -> bool:
        """Delete case.

        Args:
            case_id: Case identifier
            user_id: User ID for X-User-ID header
            correlation_id: Optional correlation ID for request tracing

        Returns:
            True if deleted successfully
        """
        async with self._get_client() as client:
            response = await client.delete(
                f"{self.base_url}/api/v1/cases/{case_id}",
                headers=self._headers(user_id=user_id, correlation_id=correlation_id),
            )
            response.raise_for_status()
            return True

    async def add_evidence(
        self, case_id: str, evidence: Evidence, user_id: str, correlation_id: Optional[str] = None
    ) -> Evidence:
        """Add evidence to case.

        Args:
            case_id: Case identifier
            evidence: Evidence object to add
            user_id: User ID for X-User-ID header
            correlation_id: Optional correlation ID for request tracing

        Returns:
            Added evidence with generated ID
        """
        async with self._get_client() as client:
            response = await client.post(
                f"{self.base_url}/api/v1/cases/{case_id}/evidence",
                json=evidence.model_dump(exclude_unset=True),
                headers=self._headers(user_id=user_id, correlation_id=correlation_id),
            )
            response.raise_for_status()
            return Evidence(**response.json())

    async def get_cases_by_session(
        self, session_id: str, user_id: str, correlation_id: Optional[str] = None
    ) -> List[Case]:
        """Get all cases for a session.

        Args:
            session_id: Session identifier
            user_id: User ID for X-User-ID header
            correlation_id: Optional correlation ID for request tracing

        Returns:
            List of cases for the session
        """
        async with self._get_client() as client:
            response = await client.get(
                f"{self.base_url}/api/v1/cases/sessions/{session_id}/cases",
                headers=self._headers(user_id=user_id, correlation_id=correlation_id),
            )
            response.raise_for_status()
            return [Case(**case) for case in response.json()]
