import os
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()
openaiApiKey = os.getenv("OPENAI_API_KEY")

# Load and chunk the company policy PDF
def load_and_chunk_policy(pdfPath):
    loader = PyPDFLoader(pdfPath)
    docs = loader.load()
    textSplitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splitDocs = textSplitter.split_documents(docs)
    return splitDocs

# Create a vector store and retriever
def create_retriever(splitDocs):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splitDocs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return retriever

# Build the RAG chatbot chain using LangGraph
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain.chains import LLMChain

promptTemplate = """
You are answering Company Policy questions based on the following policy document.
If the answer is not in the context, say "I don't know based on policy document."
Context: {context}
Question: {question}
Answer as if you are the HR:
"""

def build_langgraph_chain(retriever, openaiApiKey):
    llm = OpenAI(api_key=openaiApiKey)
    prompt = PromptTemplate.from_template(promptTemplate)
    llmChain = prompt | llm
    
    def retrieve_context(state):
        question = state["question"]
        docs = retriever.invoke(question)
        context = "\n".join([doc.page_content for doc in docs])
        return {"context": context, "question": question}

    def generate_answer(state):
        context = state["context"]
        question = state["question"]
        answer = llmChain.invoke({"context": context, "question": question})
        return {"answer": answer}

    from typing import TypedDict

    class RagState(TypedDict):
        question: str
        context: str
        answer: str

    workflow = StateGraph(RagState)
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_edge("retrieve_context", "generate_answer")
    workflow.add_edge("generate_answer", END)
    workflow.set_entry_point("retrieve_context")
    return workflow.compile()

# Main function to run the chatbot
def main():
    pdfPath = "C:\\Users\\Lenovo\\genaicoding\\langchain\\Labs\\Lab3\\company_policy.pdf"
    splitDocs = load_and_chunk_policy(pdfPath)
    retriever = create_retriever(splitDocs)
    chain = build_langgraph_chain(retriever, openaiApiKey)
    question = "What is WFH Policy?"
    result = chain.invoke({"question": question})
    print("Bot:", result["answer"])

if __name__ == "__main__":
    main()

def test_langgraph_chain():
    """Test the LangGraph RAG chain with a sample question."""
    pdfPath = "company_policy.pdf"
    splitDocs = load_and_chunk_policy(pdfPath)
    retriever = create_retriever(splitDocs)
    chain = build_langgraph_chain(retriever, openaiApiKey)
    question = "What is the leave policy?"
    result = chain.invoke({"question": question})
    assert isinstance(result["answer"], str)
    print("Test passed! Bot answered:", result["answer"])
