# 🚦 Intuition
# T = 0.0 → deterministic, same top choice every time (boring but stable).
# T ~ 0.3–0.7 → focused, good for Q&A and coding.
# T ~ 0.8–1.0 → balanced, natural conversations.
# T > 1.0 → wild, unexpected, sometimes nonsense (fun for brainstorming).

import os
from dotenv import load_dotenv
from langchain_openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize two LLMs with different temperature settings
llm_cold = OpenAI(api_key=openai_api_key, temperature=0.0)
llm_hot = OpenAI(api_key=openai_api_key, temperature=1.2)

prompt = "Invent 5 unusual ice cream flavors inspired by space travel."

print("Temperature 0.0 (deterministic):")
print(llm_cold.invoke(prompt))

print("\nTemperature 1.2 (creative):")
print(llm_hot.invoke(prompt))
