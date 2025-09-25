from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
import asyncio
import os
from dotenv import load_dotenv
from fastmcp import Client
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from fastmcp.client.transports import StdioTransport  # or NpxStdioTransport

# Load environment variables from .env file
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GITHUB_PERSONAL_ACCESS_TOKEN"] = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")

# ---------- Step 1: async call to MCP server ----------
async def list_commits(
    owner: str,
    repo: str,
    page: str = None,
    per_page: str = None,
    sha: str = None
) -> list:
    # Launch the MCP server as a subprocess via stdio
     transport = StdioTransport(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")}
    )
    
     client = Client(transport)

     # ...existing code...
     async with client:
        params = {"owner": owner, "repo": repo}
        if page:
            params["page"] = int(page)
        if per_page:
            params["per_page"] = int(per_page)
        if sha:
            params["sha"] = sha

            result = await client.call_tool("list_commits", params)
            if result.content:
                return result.content
            else:
                raise ValueError(f"No commit data returned for params: {params}")
        # ...existing code...


@tool
def get_commit_list(owner: str, repo: str, page: str = None, per_page: str = None, sha: str = None) -> list:
    """Fetch commit list using MCP server."""
    return asyncio.run(list_commits(owner, repo, page, per_page, sha))



llm = ChatOpenAI(model="gpt-4o-mini")
tools = [get_commit_list]
agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_MULTI_FUNCTIONS, verbose=False)

# In Jupyter, you can do:
result = agent.run("List latest commits from main branch of technoavengers/genaicoding")
print(result)
