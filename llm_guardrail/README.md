# LLM Guardrail

A Python package that wraps LLM API calls with validation and auto-retry capabilities.

## Features

- Validates LLM outputs against specified formats (e.g., JSON)
- Auto-retries with improved prompts on validation failures
- Configurable retry strategies and validation rules
- Detailed response metadata (retry count, status, reason)
- Built-in logging support

## Installation

```bash
pip install llm_guardrail
```

## Quick Start

```python
from llm_guardrail import safe_call

# Set your OpenAI API key in environment variables
# export OPENAI_API_KEY='your-api-key'

result = safe_call(
    prompt="List three colors in JSON format",
    model="gpt-4",
    validate="json",
    max_retries=3
)

print(result)
```

Example output:
```python
{
    "result": {"colors": ["red", "blue", "green"]},
    "status": "success",
    "retry_count": 0,
    "reason": "Valid JSON response received"
}
```

## Configuration

### Validation Strategies
- `json`: Validates JSON structure using `json.loads()`
- More validators coming soon!

### Retry Strategies
- Default strategy improves prompts for better structured output
- Configurable max retries
- Custom retry strategies can be implemented

## License

MIT License
