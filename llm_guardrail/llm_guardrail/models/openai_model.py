import os
from openai import OpenAI
from dotenv import load_dotenv
from .base import BaseModel

class OpenAIModel(BaseModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise Exception("OPENAI_API_KEY not found in environment variables")
        
        print("Checking OpenAI API configuration...")
        self.client = OpenAI(api_key=api_key)
        
        # List available models for debugging
        models = self.client.models.list()
        print("\nAvailable OpenAI models:")
        for model in models.data:
            print(f"- {model.id}")
        
        print(f"\n🤖 Testing {model_name}...")
        print("Initializing model...")
        self.model_name = model_name
        print("Model initialized successfully")

    def call(self, prompt: str, **kwargs) -> str:
        try:
            print("Calling model...")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}") 