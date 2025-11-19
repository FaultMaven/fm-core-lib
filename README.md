# fm-core-lib

FaultMaven Core Library - Shared models, LLM infrastructure, and preprocessing logic for FaultMaven microservices.

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

## License

MIT
