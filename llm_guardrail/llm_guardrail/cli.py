import argparse
import json
import sys
from typing import Optional

from .core import safe_call

def main(argv: Optional[list] = None):
    """
    CLI entry point for llm_guardrail.
    
    Args:
        argv: List of command line arguments (optional)
    """
    parser = argparse.ArgumentParser(description="Make LLM calls with validation and auto-retry")
    
    parser.add_argument(
        "prompt",
        help="The prompt to send to the LLM"
    )
    
    parser.add_argument(
        "--model",
        default="gpt-4",
        help="The OpenAI model to use (default: gpt-4)"
    )
    
    parser.add_argument(
        "--validate",
        default="json",
        choices=["json"],
        help="Validation strategy to use (default: json)"
    )
    
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retry attempts (default: 3)"
    )
    
    parser.add_argument(
        "--api-key",
        help="OpenAI API key (can also be set via OPENAI_API_KEY env var)"
    )
    
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty print the JSON output"
    )
    
    args = parser.parse_args(argv)
    
    try:
        result = safe_call(
            prompt=args.prompt,
            model=args.model,
            validate=args.validate,
            max_retries=args.max_retries,
            api_key=args.api_key
        )
        
        if args.pretty:
            print(json.dumps(result, indent=2))
        else:
            print(json.dumps(result))
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
