from setuptools import setup, find_packages

setup(
    name="llm-guardrail",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "google-generativeai>=0.8.5",
        "groq>=0.29.0",
        "python-dotenv>=0.19.0",
    ],
    entry_points={
        "console_scripts": [
            "retryLLM=llm_guardrail.cli:main"
        ]
    },
    author="Achintya Paningapalli",
    description="LLM Output Validator with auto-retry and Gemini/Groq model support.",
    python_requires=">=3.8",
)
