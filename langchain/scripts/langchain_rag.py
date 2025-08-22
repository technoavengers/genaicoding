
# pip install langchain-openai
# pip install python-dotenv
# pip install faiss-cpu
# pip install langchain_community
# pip install pypdf


import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
# RAG workflow using LangChain's PyPDFLoader
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

llm = OpenAI(api_key=openai_api_key)
# Create RAG chatbot chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)

# Conversational prompt template for RAG chatbot
prompt_template = """
You are answering interview questions based on the following resume context.
If the answer is not in the context, say "I don't know based on my resume."
Context: {context}
Question: {question}
Answer as if you are the candidate:
"""

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False,
    chain_type_kwargs={"prompt": PromptTemplate.from_template(prompt_template)}
)

# RAG chatbot loop
print("Resume RAG Chatbot. Type your interview question (or 'exit' to quit):")
while True:
    question = input("You: ")
    if question.lower() == "exit":
        break
    answer = qa_chain.invoke({"query": question})
    print("Bot:", answer)