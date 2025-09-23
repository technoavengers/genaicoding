import os
import re
import googlemaps
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
google_map_key= os.getenv("GOOGLE_MAPS_API_KEY")
print(f"Google Maps API Key: {google_map_key}")

# ✅ Initialize Google Maps client (replace with your API key or env var)
gmaps = googlemaps.Client(key=google_map_key)



# 🛠️ Define the custom tool
@tool
def directions_tool(origin: str, destination: str, mode: str = "driving") -> str:
    """Get directions between two places using Google Maps (modes: driving, walking, bicycling, transit)."""
    try:
       directions = gmaps.directions(origin, destination, mode)
       instructions = []
       for step in directions[0]["legs"][0]["steps"]:
            instructions.append(step["html_instructions"])
       return "\n".join(instructions)
    except Exception as e:
        return f"❌ Error fetching directions: {str(e)}"


# 🤖 Initialize LLM
llm = ChatOpenAI(api_key=openai_api_key,model="gpt-4o-mini", temperature=0)


# 3️⃣ PromptTemplate (to guide the agent)
template = """You are a helpful navigation assistant.
A user will give you an origin, destination, and travel mode.
Use the directions_tool to fetch step-by-step directions.
input: {input}
{agent_scratchpad}
Now provide the directions clearly:
"""
prompt = PromptTemplate(
    input_variables=["origin", "destination", "mode"],
    template=template
)

# 🔗 Register tool with the agent
agent = create_tool_calling_agent(
    llm=llm,
    tools=[directions_tool],
    prompt=prompt
)

# 6️⃣ Take input from user
origin = input("Enter starting location: ")
destination = input("Enter destination: ")
mode = input("Enter travel mode (driving/walking/bicycling/transit): ")

# 7️⃣ Format the prompt with user input
query = f"Give me direction from {origin}, to {destination}, by {mode}."

executor = AgentExecutor(agent=agent, tools=[directions_tool], verbose=True)

response = executor.invoke({"input": query})
