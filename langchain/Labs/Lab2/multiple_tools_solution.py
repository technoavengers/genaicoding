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
import requests

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
google_map_key= os.getenv("GOOGLE_MAPS_API_KEY")
print(f"Google Maps API Key: {google_map_key}")

# ✅ Initialize Google Maps client (replace with your API key or env var)
gmaps = googlemaps.Client(key=google_map_key)


@tool
def weather_tool(origin_city: str) -> str:
    """Get current weather for a city using OpenWeatherMap API."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={origin_city}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    if "weather" in data and "main" in data:
        return f"{origin_city.title()} weather: {data['weather'][0]['description']}, {data['main']['temp']}°C"
    else:
        return f"Could not fetch weather for {origin_city}."

@tool
def get_travel_time(origin: str, destination: str, mode: str = "driving") -> str:
    """Get travel time between two places using Google Maps (modes: driving, walking, bicycling, transit)."""
    directions = gmaps.directions(
        origin=origin,
        destination=destination,
        mode=mode,
        departure_time="now"  # Use live traffic data
    )
    travel_time = directions[0]["legs"][0].get("duration_in_traffic", directions[0]["legs"][0]["duration"])
    travel_time_text = travel_time["text"]
    return f"Estimated travel time: {travel_time_text}"

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
template = """You are a helpful weather and navigation assistant.
use the tools below to answer the user query.
input: {input}
{agent_scratchpad}
"""
prompt = PromptTemplate(
    input_variables=["origin", "destination", "mode"],
    template=template
)

# 🔗 Register tool with the agent
agent = create_tool_calling_agent(
    llm=llm,
    tools=[directions_tool, weather_tool,get_travel_time],
    prompt=prompt
)



executor = AgentExecutor(agent=agent, tools=[directions_tool,weather_tool,get_travel_time], verbose=True)

response = executor.invoke({"input": "I am travelling from newyork to boston by car, Based on travel time, should i do WFH?"})
