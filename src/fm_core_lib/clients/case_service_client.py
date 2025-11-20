"""HTTP client for fm-case-service."""

import httpx
from typing import List, Optional
from fm_core_lib.models import Case, Evidence, UploadedFile


class CaseServiceClient:
    """Async HTTP client for communicating with fm-case-service.
    
    This client provides a Pythonic interface to the fm-case-service REST API,
    handling serialization/deserialization and error handling.
    """
    
    def __init__(self, base_url: str = "http://fm-case-service:8003"):
        """Initialize client.
        
        Args:
            base_url: Base URL of fm-case-service (default: http://fm-case-service:8003)
        """
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={"Content-Type": "application/json"}
        )
    
    async def get_case(self, case_id: str) -> Case:
        """Get case by ID.
        
        Args:
            case_id: Case identifier
            
        Returns:
            Case object
            
        Raises:
            httpx.HTTPStatusError: If case not found or other HTTP error
        """
        response = await self.client.get(f"{self.base_url}/api/v1/cases/{case_id}")
        response.raise_for_status()
        return Case(**response.json())
    
    async def create_case(self, case: Case) -> Case:
        """Create new case.
        
        Args:
            case: Case object to create
            
        Returns:
            Created case with generated ID
        """
        response = await self.client.post(
            f"{self.base_url}/api/v1/cases",
            json=case.model_dump(exclude_unset=True)
        )
        response.raise_for_status()
        return Case(**response.json())
    
    async def update_case(self, case_id: str, case: Case) -> Case:
        """Update existing case.
        
        Args:
            case_id: Case identifier
            case: Updated case object
            
        Returns:
            Updated case
        """
        response = await self.client.put(
            f"{self.base_url}/api/v1/cases/{case_id}",
            json=case.model_dump()
        )
        response.raise_for_status()
        return Case(**response.json())
    
    async def delete_case(self, case_id: str) -> bool:
        """Delete case.
        
        Args:
            case_id: Case identifier
            
        Returns:
            True if deleted successfully
        """
        response = await self.client.delete(f"{self.base_url}/api/v1/cases/{case_id}")
        response.raise_for_status()
        return True
    
    async def add_evidence(self, case_id: str, evidence: Evidence) -> Evidence:
        """Add evidence to case.
        
        Args:
            case_id: Case identifier
            evidence: Evidence object to add
            
        Returns:
            Added evidence with generated ID
        """
        response = await self.client.post(
            f"{self.base_url}/api/v1/cases/{case_id}/evidence",
            json=evidence.model_dump(exclude_unset=True)
        )
        response.raise_for_status()
        return Evidence(**response.json())
    
    async def get_cases_by_session(self, session_id: str) -> List[Case]:
        """Get all cases for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of cases for the session
        """
        response = await self.client.get(
            f"{self.base_url}/api/v1/cases/sessions/{session_id}/cases"
        )
        response.raise_for_status()
        return [Case(**case) for case in response.json()]
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
