"""Redis Connection Factory with Sentinel Support

Provides deployment-neutral Redis connection management supporting:
- Standalone Redis (development/self-hosted)
- Redis Sentinel (enterprise HA/Kubernetes)

This enables automatic failover in production environments without code changes.
"""

import logging
import os
from typing import List, Tuple

from redis.asyncio import Redis
from redis.asyncio.sentinel import Sentinel
from fm_core_lib.utils import service_startup_retry

logger = logging.getLogger(__name__)


def _parse_sentinel_hosts(hosts_str: str) -> List[Tuple[str, int]]:
    """Parse comma-separated sentinel host:port string.

    Args:
        hosts_str: Comma-separated list of host:port (e.g., "host1:26379,host2:26379")

    Returns:
        List of (host, port) tuples

    Example:
        >>> _parse_sentinel_hosts("sentinel1:26379,sentinel2:26379")
        [('sentinel1', 26379), ('sentinel2', 26379)]
    """
    sentinels = []

    for host_port in hosts_str.split(","):
        host_port = host_port.strip()
        if not host_port:
            continue

        if ":" in host_port:
            host, port_str = host_port.rsplit(":", 1)
            sentinels.append((host, int(port_str)))
        else:
            # Default Sentinel port
            sentinels.append((host_port, 26379))

    return sentinels


@service_startup_retry
async def _verify_redis_connection(client: Redis) -> None:
    """Verify Redis connection with retry logic.

    Args:
        client: Redis client to verify

    Raises:
        Exception: If connection fails after retries
    """
    await client.ping()
    logger.info("Redis connection verified")


async def get_redis_client(
    mode: str = None,
    host: str = None,
    port: int = None,
    db: int = None,
    password: str = None,
    sentinel_hosts: str = None,
    master_set: str = None,
    decode_responses: bool = True,
    socket_keepalive: bool = True,
    health_check_interval: int = 30
) -> Redis:
    """Get Redis client with automatic Sentinel/Standalone selection.

    Supports both standalone Redis and Redis Sentinel for HA deployments.
    Configuration is primarily driven by environment variables for deployment neutrality.

    Args:
        mode: Redis mode ("standalone" or "sentinel", default from REDIS_MODE env)
        host: Redis host (default from REDIS_HOST env)
        port: Redis port (default from REDIS_PORT env)
        db: Database index (default from REDIS_DB env)
        password: Redis password (default from REDIS_PASSWORD env)
        sentinel_hosts: Comma-separated sentinel hosts (default from REDIS_SENTINEL_HOSTS env)
        master_set: Sentinel master set name (default from REDIS_MASTER_SET env)
        decode_responses: Decode responses to strings (default: True)
        socket_keepalive: Enable socket keepalive (default: True)
        health_check_interval: Health check interval in seconds (default: 30)

    Returns:
        Async Redis client (either standalone or Sentinel-managed)

    Environment Variables:
        REDIS_MODE: "standalone" (default) or "sentinel"

        For standalone:
            REDIS_HOST: Redis host (default: "localhost")
            REDIS_PORT: Redis port (default: 6379)
            REDIS_DB: Database index (default: 0)
            REDIS_PASSWORD: Redis password (optional)

        For Sentinel:
            REDIS_SENTINEL_HOSTS: Comma-separated "host:port" pairs
                                 (e.g., "sentinel1:26379,sentinel2:26379,sentinel3:26379")
            REDIS_MASTER_SET: Master set name (default: "mymaster")
            REDIS_PASSWORD: Redis password (optional)
            REDIS_DB: Database index (default: 0)

    Example:
        ```python
        # Self-hosted deployment (docker-compose)
        REDIS_MODE=standalone
        REDIS_HOST=redis
        REDIS_PORT=6379

        # Enterprise K8s with Sentinel (Bitnami Redis Cluster)
        REDIS_MODE=sentinel
        REDIS_SENTINEL_HOSTS=redis-node-0.redis-headless:26379,redis-node-1.redis-headless:26379,redis-node-2.redis-headless:26379
        REDIS_MASTER_SET=mymaster
        REDIS_PASSWORD=secret-password
        ```

    Raises:
        ValueError: If Sentinel mode is configured but sentinel_hosts is missing
        ConnectionError: If Redis connection fails after retries
    """
    # Get configuration from environment if not explicitly provided
    mode = (mode or os.getenv("REDIS_MODE", "standalone")).lower()
    db_index = db if db is not None else int(os.getenv("REDIS_DB", "0"))
    password = password or os.getenv("REDIS_PASSWORD")

    logger.info(f"Initializing Redis client in {mode} mode")

    if mode == "sentinel":
        # Redis Sentinel for HA deployments
        sentinel_hosts_str = sentinel_hosts or os.getenv("REDIS_SENTINEL_HOSTS", "")
        master_name = master_set or os.getenv("REDIS_MASTER_SET", "mymaster")

        if not sentinel_hosts_str:
            raise ValueError(
                "REDIS_SENTINEL_HOSTS environment variable is required for Sentinel mode"
            )

        # Parse sentinel hosts
        sentinels = _parse_sentinel_hosts(sentinel_hosts_str)

        if not sentinels:
            raise ValueError(
                f"No valid sentinel hosts found in: {sentinel_hosts_str}"
            )

        logger.info(
            f"Connecting to Redis Sentinel: master={master_name}, "
            f"sentinels={sentinels}"
        )

        # Create Sentinel client
        sentinel_client = Sentinel(
            sentinels,
            sentinel_kwargs={"password": password} if password else {},
            socket_keepalive=socket_keepalive,
            health_check_interval=health_check_interval
        )

        # Get master connection (automatically handles failover)
        redis_client = sentinel_client.master_for(
            master_name,
            db=db_index,
            password=password,
            decode_responses=decode_responses,
            socket_keepalive=socket_keepalive,
            health_check_interval=health_check_interval
        )

        # Verify connection with retry logic
        await _verify_redis_connection(redis_client)

        logger.info(
            f"Redis Sentinel connection established: master={master_name}, "
            f"db={db_index}"
        )

    else:
        # Standalone Redis (default for development/self-hosted)
        redis_host = host or os.getenv("REDIS_HOST", "localhost")
        redis_port = port if port is not None else int(os.getenv("REDIS_PORT", "6379"))

        logger.info(
            f"Connecting to standalone Redis: {redis_host}:{redis_port}/{db_index}"
        )

        redis_client = Redis(
            host=redis_host,
            port=redis_port,
            db=db_index,
            password=password,
            decode_responses=decode_responses,
            socket_keepalive=socket_keepalive,
            health_check_interval=health_check_interval,
            socket_connect_timeout=5
        )

        # Verify connection with retry logic
        await _verify_redis_connection(redis_client)

        logger.info(
            f"Standalone Redis connection established: {redis_host}:{redis_port}/{db_index}"
        )

    return redis_client
