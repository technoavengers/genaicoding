"""
This script implements an autonomous Resume RAG (Retrieval-Augmented Generation) chatbot agent for answering interview questions based on the content of a resume PDF.

Functionalities:
- Loads a resume from a PDF file and splits it into manageable text chunks.
- Embeds and indexes the resume chunks using OpenAI embeddings and FAISS for semantic search.
- Defines a resume search tool that retrieves answers from the resume; if the answer is not found, it returns a fallback phrase.
- Defines a Twilio SMS tool that sends a notification to the user if the agent cannot answer a question based on the resume.
- Creates a conversational agent using LangChain's tool-calling agent and a custom prompt template.
- Runs an interactive loop where users can input interview questions, and the agent responds as if it were the candidate, using the resume as context.

Environment variables required:
- OPENAI_API_KEY: OpenAI API key for embeddings and LLM.
- TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, TWILIO_TO_NUMBER: Twilio credentials for SMS notifications.

Dependencies:
- langchain-openai
- python-dotenv
- faiss-cpu
- langchain_community
- pypdf
- twilio

Usage:
Run the script and input interview questions. If the answer is not found in the resume, the agent will notify the user via SMS using Twilio.
"""

from twilio.rest import Client
from langchain.tools import Tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
import logging
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")

# Twilio credentials
twilio_sid = os.getenv("TWILIO_ACCOUNT_SID")
twilio_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_from = os.getenv("TWILIO_FROM_NUMBER")
twilio_to = os.getenv("TWILIO_TO_NUMBER")

# Create twilio client
twilio_client = Client(twilio_sid, twilio_token)

# Twilio SMS tool
def send_sms_tool(message: str) -> str:
    print(f"[DEBUG] Sending SMS: {message}")
    twilio_client.messages.create(
        body=message,
        from_=twilio_from,
        to=twilio_to
    )
    return "SMS sent to your phone."

twilio_tool = Tool(
    name="twilio_tool",
    func=send_sms_tool,
    description="If you ever answer 'I don't know based on my resume.', immediately use this tool to notify the user by SMS. When you use this tool, send the original interview question that was asked, not the fallback answer."
)

# Resume retriever tool
def resume_search_tool(query: str) -> str:
    result = qa_chain.invoke({"query": query})
    # Always return a plain string
    if isinstance(result, dict):
        answer = str(result)
    print(f"[DEBUG] Resume search result: {answer}")
    # Always return the explicit fallback phrase if not found
    if answer.strip() == "" or "I don't know" in answer:
        return "I don't know based on my resume."
    return answer

resume_tool = Tool(
    name="resume_search",
    func=resume_search_tool,
    description="Use this tool to search the resume and answer interview questions based on its content. If the answer is not found, return 'I don't know based on my resume.' as a string. If you ever answer 'I don't know based on my resume.', you must immediately use the twilio_tool to notify the user by SMS."
)


pdf_path = "my_resume.pdf"
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# Split resume into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(docs)

# Create vector store from split documents
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(split_docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=openai_api_key)

# Conversational prompt template for RAG chatbot
prompt_template = """
You are answering interview questions based on the resume.
If the answer is not in the resume, say "I don't know based on my resume." If you ever answer 'I don't know based on my resume.', you must immediately use the twilio_tool to notify the user by SMS. When you use the twilio_tool, send the original interview question that was asked, not the fallback answer.
Question: {input}
{agent_scratchpad}
Answer as if you are the candidate:
"""

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)

# Create agent 
agent = create_tool_calling_agent(llm, [resume_tool, twilio_tool], PromptTemplate.from_template(prompt_template))
executor = AgentExecutor(agent=agent, tools=[resume_tool, twilio_tool])

print("Resume RAG Chatbot (Autonomous Agent). Type your interview question (or 'exit' to quit):")
while True:
    question = input("You: ")
    if question.lower() == "exit":
        break
    print(f"[DEBUG] Agent received question: {question}")
    response = executor.invoke({"input": question})
    # Print output from agent
    print("Bot:", response.get("output", response))