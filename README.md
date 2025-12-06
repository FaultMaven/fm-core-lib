# fm-core-lib

> **Part of [FaultMaven](https://github.com/FaultMaven/faultmaven)** —
> The AI-Powered Troubleshooting Copilot

FaultMaven Core Library - Shared models, LLM infrastructure, and preprocessing logic for FaultMaven microservices.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## Overview

This library provides the foundational components shared across all FaultMaven microservices:

- **Models**: Case, Evidence, Hypothesis, InvestigationProgress, and API models
- **LLM Infrastructure**: Multi-provider LLM routing with failover (OpenAI, Anthropic, Fireworks AI)
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
