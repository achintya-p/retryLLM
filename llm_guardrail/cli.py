#!/usr/bin/env python3
import argparse
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check API keys
google_api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

if not google_api_key and not groq_api_key:
    print("Warning: No API keys found. Please set GOOGLE_API_KEY and/or GROQ_API_KEY in your .env file")
elif not google_api_key:
    print("Warning: GOOGLE_API_KEY not found")
elif not groq_api_key:
    print("Warning: GROQ_API_KEY not found")

from llm_guardrail import safe_call
from llm_guardrail.models.router import SmartRouter

# Available models
AVAILABLE_MODELS = ["gemini-pro", "llama3-70b-8192"]

def main():
    # Load environment variables
    router = SmartRouter()
    
    parser = argparse.ArgumentParser(description="retryLLM - Reliable LLM calls with validation and retries")
    
    parser.add_argument(
        "prompt",
        help="The prompt to send to the model"
    )
    
    parser.add_argument(
        "--model",
        choices=AVAILABLE_MODELS,
        help="Specific model to use (optional, will auto-select if not specified)"
    )
    
    parser.add_argument(
        "--validate",
        choices=["json", "llm_judge"],
        help="Validation method to use (optional)"
    )
    
    parser.add_argument(
        "--judge-model",
        choices=AVAILABLE_MODELS,
        help="Model to use for LLM judging (if using llm_judge validation)"
    )
    
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries per model (default: 3)"
    )
    
    parser.add_argument(
        "--fallback",
        choices=AVAILABLE_MODELS,
        help="Fallback model to use if primary fails"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for model generation (0.0 to 1.0, default: 0.7)"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output full response as JSON"
    )
    
    args = parser.parse_args()
    
    # Prepare fallback models list if specified
    fallback_models = [args.fallback] if args.fallback else None
    
    try:
        # Make the call
        result = safe_call(
            prompt=args.prompt,
            model=args.model,
            validate=args.validate,
            judge_model=args.judge_model,
            max_retries=args.max_retries,
            fallback_models=fallback_models,
            temperature=args.temperature
        )
        
        if args.json:
            # Output full response as JSON
            print(json.dumps(result, indent=2))
        else:
            # Output formatted response
            print("\n=== Results ===")
            print(f"Status: {result['status']}")
            print(f"Model Used: {result.get('model_used', result.get('model'))}")
            print(f"Retry Count: {result['retry_count']}")
            
            if args.validate == "llm_judge" and "validation" in result:
                print("\n=== Judge's Evaluation ===")
                validation = result["validation"]
                print(f"Score: {validation['score']}")
                print(f"Reason: {validation['reason']}")
                if validation['improvements_needed']:
                    print("Improvements needed:")
                    for imp in validation['improvements_needed']:
                        print(f"- {imp}")
            
            print("\n=== Generated Content ===")
            if args.validate == "json":
                # Pretty print JSON content
                content = result["result"]
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except json.JSONDecodeError:
                        pass
                print(json.dumps(content, indent=2))
            else:
                print(result["result"])
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 