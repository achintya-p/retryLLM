# retryLLM

A reliable LLM interface with retries, validation, and RL-based routing.

## Features

- Smart model selection using reinforcement learning
- Automatic retries and fallbacks
- Response validation (JSON and LLM judge)
- Performance monitoring and optimization
- Efficient caching
- Support for Google (Gemini) and Groq models

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/retryLLM.git
cd retryLLM
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

4. Create a `.env` file with your API keys:
```
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
```

## Usage

### Command Line

1. Basic usage (auto-selects best model):
```bash
retryLLM "Your prompt here"
```

2. Specify a model:
```bash
retryLLM --model gemini-pro "Your prompt here"
```

3. Use JSON validation:
```bash
retryLLM --validate json "Generate a JSON object with name and age fields"
```

4. Use LLM judge validation:
```bash
retryLLM --validate llm_judge --judge-model gemini-pro "Explain quantum computing"
```

5. Set fallback model and retries:
```bash
retryLLM --model llama3-70b-8192 --fallback gemini-pro --max-retries 2 "Your prompt here"
```

6. Get full JSON output:
```bash
retryLLM --json "Your prompt here"
```

### Python API

```python
from llm_guardrail import safe_call

# Basic call
result = safe_call("Your prompt here")

# With validation and specific model
result = safe_call(
    prompt="Generate a JSON object with name and age fields",
    model="gemini-pro",
    validate="json",
    max_retries=3
)

# With LLM judge validation
result = safe_call(
    prompt="Explain quantum computing",
    validate="llm_judge",
    judge_model="gemini-pro"
)
```

## Available Models

- `gemini-pro`: Google's powerful general-purpose model
- `llama3-70b-8192`: Groq's high-performance Llama model

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 