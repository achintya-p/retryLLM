from setuptools import setup, find_packages

setup(
    name="retryLLM",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "anthropic>=0.57.1",
        "google-generativeai>=0.8.5",
        "groq>=0.29.0",
        "python-dotenv>=0.19.0",
    ],
    author="Achintya Paningapalli",
    author_email="apaninga@berkeley.edu",  # Update if needed
    description="LLM Output Validator + Auto-Retry Layer",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/achintya-p/retryLLM",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
