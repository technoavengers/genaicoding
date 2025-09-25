from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage,SystemMessage

# Connect to Ollama model
llm = ChatOllama(model="mistral",streaming=False)

# Send a message
response = llm.invoke("What is capital of france")
print(response.content)
