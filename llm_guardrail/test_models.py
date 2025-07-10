import os
from pathlib import Path
from dotenv import load_dotenv
from llm_guardrail.models import get_model
import traceback
import google.generativeai as genai

# Load environment variables with explicit path
env_path = Path(__file__).parent / '.env'
print(f"Looking for .env file at: {env_path}")
load_dotenv(env_path)

# Print loaded API keys (safely)
print("\nAPI Keys loaded:")
print(f"Google API Key: {'[Set]' if os.getenv('GOOGLE_API_KEY') else '[Not Set]'} (starts with: {os.getenv('GOOGLE_API_KEY')[:8]}...)")

def test_model(model_name: str, prompt: str = "Tell me a short joke"):
    """
    Test a specific model with the given prompt.
    
    Args:
        model_name (str): Name of the model to test
        prompt (str): Prompt to send to the model
    """
    print(f"\n=== Testing {model_name} ===")
    try:
        model = get_model(model_name)
        response = model.call(prompt)
        print("\n✅ Success!")
        print(f"Response: {response}")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("Detailed error:")
        import traceback
        print(traceback.format_exc())

def main():
    # Test prompt
    prompt = "Tell me a short joke about programming"
    
    # Test each model
    models_to_test = [
        "models/gemini-2.5-pro",  # Google
        "gpt-3.5-turbo",         # OpenAI
        "mixtral-8x7b-32768"     # Groq
    ]
    
    for model_name in models_to_test:
        test_model(model_name, prompt)

if __name__ == "__main__":
    main() 