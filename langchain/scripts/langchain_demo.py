# pip install langchain-openai
# pip install langchain_community
# pip install python-dotenv

import os
from dotenv import load_dotenv
from langchain_openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the LLM
llm = OpenAI(api_key=openai_api_key)

# Basic prompt
prompt = "What is the capital of France?"

# Invoke the LLM using the new invoke method
response = llm.invoke(prompt)
print("Response:", response)
