"""Service Discovery for Deployment-Neutral Service Communication

Provides automatic service URL resolution for different deployment modes:
- Docker Compose: Container name resolution
- Kubernetes: DNS-based service discovery
- Local: Localhost port mapping
"""

import logging
import os
from enum import Enum
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class DeploymentMode(Enum):
    """Deployment environment types."""

    DOCKER = "docker"  # Docker Compose
    KUBERNETES = "kubernetes"  # Kubernetes cluster
    LOCAL = "local"  # Local development (localhost)


class ServiceRegistry:
    """Service discovery registry for deployment-neutral URL resolution.

    Automatically constructs service URLs based on deployment mode:
    - Docker: http://fm-{service}-service:{port}
    - K8s: http://fm-{service}-service.{namespace}.svc.cluster.local:{port}
    - Local: http://localhost:{port}

    Environment Variables:
        DEPLOYMENT_MODE: "docker" (default), "kubernetes", or "local"
        K8S_NAMESPACE: Kubernetes namespace (default: "faultmaven")
        SERVICE_{NAME}_PORT: Override default port for a service

    Example:
        ```python
        registry = ServiceRegistry()
        agent_url = registry.get_url("agent")
        # Docker: http://fm-agent-service:8001
        # K8s: http://fm-agent-service.faultmaven.svc.cluster.local:8001
        # Local: http://localhost:8001
        ```
    """

    # Default service port mappings
    DEFAULT_PORTS: Dict[str, int] = {
        "auth": 8000,
        "agent": 8001,
        "session": 8002,
        "knowledge": 8003,
        "evidence": 8004,
        "case": 8005,
        "gateway": 8090,
        "job-worker": 8080,  # Celery worker API (if exposed)
    }

    def __init__(
        self,
        mode: Optional[str] = None,
        namespace: Optional[str] = None,
        custom_ports: Optional[Dict[str, int]] = None
    ):
        """Initialize service registry.

        Args:
            mode: Deployment mode (overrides DEPLOYMENT_MODE env var)
            namespace: Kubernetes namespace (overrides K8S_NAMESPACE env var)
            custom_ports: Custom port mappings (overrides defaults)
        """
        # Determine deployment mode
        mode_str = mode or os.getenv("DEPLOYMENT_MODE", "docker")
        try:
            self.mode = DeploymentMode(mode_str.lower())
        except ValueError:
            logger.warning(
                f"Invalid DEPLOYMENT_MODE '{mode_str}', defaulting to 'docker'"
            )
            self.mode = DeploymentMode.DOCKER

        # Kubernetes namespace
        self.namespace = namespace or os.getenv("K8S_NAMESPACE", "faultmaven")

        # Build service port mapping
        self.services = self.DEFAULT_PORTS.copy()
        if custom_ports:
            self.services.update(custom_ports)

        # Allow environment variable overrides for ports
        for service_name in self.services.keys():
            env_key = f"SERVICE_{service_name.upper().replace('-', '_')}_PORT"
            env_port = os.getenv(env_key)
            if env_port:
                try:
                    self.services[service_name] = int(env_port)
                except ValueError:
                    logger.warning(f"Invalid port in {env_key}: {env_port}")

        logger.info(
            f"ServiceRegistry initialized: mode={self.mode.value}, "
            f"namespace={self.namespace}, "
            f"services={len(self.services)}"
        )

    def get_url(self, service_name: str, protocol: str = "http") -> str:
        """Get the full URL for a service.

        Args:
            service_name: Service name (e.g., "agent", "knowledge")
            protocol: Protocol to use (default: "http")

        Returns:
            Full service URL

        Raises:
            ValueError: If service name is unknown

        Example:
            >>> registry = ServiceRegistry()
            >>> registry.get_url("agent")
            'http://fm-agent-service:8001'
        """
        if service_name not in self.services:
            raise ValueError(
                f"Unknown service: {service_name}. "
                f"Known services: {list(self.services.keys())}"
            )

        port = self.services[service_name]
        service_full_name = f"fm-{service_name}-service"

        if self.mode == DeploymentMode.KUBERNETES:
            # Kubernetes DNS: service.namespace.svc.cluster.local
            host = f"{service_full_name}.{self.namespace}.svc.cluster.local"
            url = f"{protocol}://{host}:{port}"

        elif self.mode == DeploymentMode.LOCAL:
            # Local development: localhost with port mapping
            url = f"{protocol}://localhost:{port}"

        else:  # Docker Compose
            # Docker Compose: container name resolution
            url = f"{protocol}://{service_full_name}:{port}"

        logger.debug(f"Resolved {service_name} -> {url}")
        return url

    def get_host(self, service_name: str) -> str:
        """Get just the hostname (without protocol or port).

        Args:
            service_name: Service name

        Returns:
            Hostname only

        Example:
            >>> registry = ServiceRegistry()
            >>> registry.get_host("agent")
            'fm-agent-service'  # Docker mode
        """
        if service_name not in self.services:
            raise ValueError(f"Unknown service: {service_name}")

        service_full_name = f"fm-{service_name}-service"

        if self.mode == DeploymentMode.KUBERNETES:
            return f"{service_full_name}.{self.namespace}.svc.cluster.local"
        elif self.mode == DeploymentMode.LOCAL:
            return "localhost"
        else:  # Docker
            return service_full_name

    def get_port(self, service_name: str) -> int:
        """Get the port for a service.

        Args:
            service_name: Service name

        Returns:
            Port number

        Raises:
            ValueError: If service name is unknown
        """
        if service_name not in self.services:
            raise ValueError(f"Unknown service: {service_name}")

        return self.services[service_name]

    def register_service(self, service_name: str, port: int) -> None:
        """Register a new service or update existing one.

        Args:
            service_name: Service name
            port: Service port

        Example:
            >>> registry = ServiceRegistry()
            >>> registry.register_service("custom-service", 9000)
        """
        self.services[service_name] = port
        logger.info(f"Registered service: {service_name} -> port {port}")

    def list_services(self) -> Dict[str, str]:
        """List all registered services with their URLs.

        Returns:
            Dictionary of service_name -> full_url

        Example:
            >>> registry = ServiceRegistry()
            >>> registry.list_services()
            {'agent': 'http://fm-agent-service:8001', ...}
        """
        return {name: self.get_url(name) for name in self.services.keys()}


# Singleton instance for global access
_registry_instance: Optional[ServiceRegistry] = None


def get_service_registry() -> ServiceRegistry:
    """Get or create the global ServiceRegistry instance.

    Returns:
        Global ServiceRegistry singleton

    Example:
        ```python
        from fm_core_lib.discovery import get_service_registry

        registry = get_service_registry()
        knowledge_url = registry.get_url("knowledge")
        ```
    """
    global _registry_instance

    if _registry_instance is None:
        _registry_instance = ServiceRegistry()

    return _registry_instance


def reset_service_registry():
    """Reset the global ServiceRegistry instance.

    Used for testing or reconfiguration.
    """
    global _registry_instance
    _registry_instance = None
    logger.warning("ServiceRegistry instance reset")
