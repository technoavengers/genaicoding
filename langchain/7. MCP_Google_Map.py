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
os.environ["GOOGLE_MAPS_API_KEY"] = os.getenv("GOOGLE_MAPS_API_KEY")


# ---------- Step 1: async call to MCP server ----------
async def call_directions(origin: str, destination: str, mode: str = "driving"):
    # Launch the MCP server as a subprocess via stdio
     transport = StdioTransport(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-google-maps"],
        env={"GOOGLE_MAPS_API_KEY": os.getenv("GOOGLE_MAPS_API_KEY")},
    )
    
     client = Client(transport)

     async with client:
        result = await client.call_tool(
            "maps_directions",
            {"origin": origin, "destination": destination, "mode": mode},
        )
        return result.data


# ---------- Step 2: Wrap in LangChain tool ----------
@tool
def get_directions(origin: str, destination: str, mode: str = "driving") -> str:
    """Get Google Maps directions using MCP server."""
    return asyncio.run(call_directions(origin, destination, mode))


# ---------- Step 3: Build LangChain agent ----------
def main():
    llm = ChatOpenAI(model="gpt-4o-mini")   # Or gpt-4o / gpt-3.5-turbo
    tools = [get_directions]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_MULTI_FUNCTIONS,
        verbose=True,
    )

    # Example query
    result = agent.run("Find me driving directions from Delhi to Jaipur")
    print("\nFinal Answer:\n", result)


if __name__ == "__main__":
    main()
