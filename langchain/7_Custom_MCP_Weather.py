import asyncio
from fastmcp import FastMCP, Client
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
import requests


load_dotenv()
mcp = FastMCP("custom-weather-mcp")

@mcp.tool
def get_weather(city: str) -> str:
    """Fetch weather information for a given city."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    if "weather" in data and "main" in data:
        return f"{city.title()} weather: {data['weather'][0]['description']}, {data['main']['temp']}°C"
    else:
        return f"Could not fetch weather for {city}."

if __name__ == "__main__":
    # This will start the MCP server and listen for HTTP requests on port 8002
    mcp.run(transport="http", port=8002)

