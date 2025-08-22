import asyncio
from fastmcp import FastMCP, Client
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv


load_dotenv()
 
llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))
 
mcp = FastMCP("My MCP Server")
 
@mcp.tool
def translator(text: str, target_lang: str) -> str:
    """Translate text into the target language using LangChain's ChatOpenAI."""
    prompt = f"Translate the following text to {target_lang}: {text}"
    return str(llm.invoke(prompt).content)

if __name__ == "__main__":
    # This will start the MCP server and listen for HTTP requests on port 8000
    mcp.run(transport="http", port=8000)

