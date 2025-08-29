from langchain_community.llms import Ollama

# Connect to Ollama model
llm = Ollama(model="mistral")

# Send a message
response = llm.invoke("Explain async vs multithreading in simple words")"")
print(response.content)