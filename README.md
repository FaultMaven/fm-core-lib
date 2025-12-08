# fm-core-lib

> **Part of [FaultMaven](https://github.com/FaultMaven/faultmaven)** —
> The AI-Powered Troubleshooting Copilot

FaultMaven Core Library - Shared models, LLM infrastructure, and preprocessing logic for FaultMaven microservices.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## Overview

This library provides the foundational components shared across all FaultMaven microservices:

- **Models**: Case, Evidence, Hypothesis, InvestigationProgress, and API models
- **LLM Infrastructure**: Multi-provider LLM routing with failover (OpenAI, Anthropic, Fireworks AI)
- **Service Discovery**: Deployment-neutral service URL resolution (Docker/Kubernetes/Local)
- **Preprocessing**: Data sanitization and preprocessing logic

## Installation

```bash
pip install fm-core-lib
```

## Usage

### Models

```python
from fm_core_lib.models import Case, CaseStatus, InvestigationProgress

# Create a new case
case = Case(
    case_id="case_123",
    user_id="user_456",
    status=CaseStatus.CONSULTING,
    investigation_progress=InvestigationProgress()
)
```

### LLM Router

```python
from fm_core_lib.infrastructure.llm import LLMRouter

router = LLMRouter()
response = await router.generate(
    messages=[{"role": "user", "content": "Hello"}],
    temperature=0.7
)
```

### Service Discovery

```python
from fm_core_lib.discovery import get_service_registry

# Get service registry singleton
registry = get_service_registry()

# Get service URL (automatically resolves based on DEPLOYMENT_MODE)
auth_url = registry.get_url("auth")
# Returns:
#   Docker:     http://fm-auth-service:8000
#   Kubernetes: http://fm-auth-service.faultmaven.svc.cluster.local:8000
#   Local:      http://localhost:8000
```

**Configuration:**

Set the deployment mode via environment variables:

```bash
# Docker Compose (default)
DEPLOYMENT_MODE=docker

# Kubernetes
DEPLOYMENT_MODE=kubernetes
K8S_NAMESPACE=faultmaven

# Local development
DEPLOYMENT_MODE=local
```

**Supported Services:**

- `auth` - fm-auth-service (port 8000)
- `agent` - fm-agent-service (port 8001)
- `session` - fm-session-service (port 8002)
- `knowledge` - fm-knowledge-service (port 8003)
- `evidence` - fm-evidence-service (port 8004)
- `case` - fm-case-service (port 8005)
- `gateway` - fm-api-gateway (port 8090)

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src tests

# Type checking
mypy src
```

## Architecture

This library follows the "Open Core" strategy - a single codebase that can be configured at runtime via the `PROFILE` environment variable:

- `PROFILE=public`: Community edition features
- `PROFILE=enterprise`: Enterprise edition with additional capabilities

## Contributing

See our [Contributing Guide](https://github.com/FaultMaven/.github/blob/main/CONTRIBUTING.md) for detailed guidelines.

## Support

- **Discussions:** [GitHub Discussions](https://github.com/FaultMaven/faultmaven/discussions)
- **Issues:** [GitHub Issues](https://github.com/FaultMaven/fm-core-lib/issues)

## Related Projects

- **[faultmaven](https://github.com/FaultMaven/faultmaven)** - Main repository and documentation
- **[faultmaven-deploy](https://github.com/FaultMaven/faultmaven-deploy)** - Deployment configurations
- **[fm-agent-service](https://github.com/FaultMaven/fm-agent-service)** - AI agent service
- **[fm-knowledge-service](https://github.com/FaultMaven/fm-knowledge-service)** - Knowledge base service

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.
