import google.generativeai as genai
from dotenv import load_dotenv
from .base import BaseModel
import os

class GoogleModel(BaseModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise Exception("GOOGLE_API_KEY not found in environment variables")
        
        print("Checking Google API configuration...")
        genai.configure(api_key=api_key)
        
        # List available models for debugging
        model_list = genai.list_models()
        print("\nAvailable Google models:")
        for m in model_list:
            print(f"- {m.name} (supports: {m.supported_generation_methods})")
        
        print(f"\n🤖 Testing {model_name}...")
        print("Initializing model...")
        self.model = genai.GenerativeModel("models/gemini-2.5-pro")
        print("Model initialized successfully")

    def call(self, prompt: str, **kwargs) -> str:
        try:
            print("Calling model...")
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            raise Exception(f"Google API error: {str(e)}") 