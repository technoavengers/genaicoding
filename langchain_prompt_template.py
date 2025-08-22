from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

# Define a prompt template with variables
template = "Translate the following English {text} to {language}"

prompt = PromptTemplate(
    input_variables=["language", "text"],
    template=template,
)

# Take input from user
language = input("Enter the language: ")
text = input("Enter the text: ")

# Fill the prompt
filled_prompt = prompt.format(language=language, text=text)


# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the LLM
llm = OpenAI(api_key=openai_api_key)

# Invoke the LLM with the filled prompt
response = llm.invoke(filled_prompt)

print(response)