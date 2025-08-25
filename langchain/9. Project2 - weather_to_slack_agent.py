"""
Problem Statement
Design a LangChain-based GenAI agent that accepts a city name as input, retrieves live weather data from the OpenWeatherMap API, and posts a formatted weather update to a Slack channel using a webhook. The system should demonstrate end-to-end integration of external tools within a LangChain agent pipeline, enabling real-time data retrieval and enterprise messaging.

Objectives
Build a Tool-Calling Agent using LangChain that integrates with the OpenWeatherMap API.

Dynamically fetch weather data based on user input (city name).

Format the response into a human-readable weather summary.

Send the result to a specified Slack channel using Slack’s incoming webhook.

Demonstrate practical agent workflows, including prompt handling, tool invocation, response parsing, and messaging integration.
"""

# Required Libraries
import os
import requests
from dotenv import load_dotenv
from langchain.agents import create_tool_calling_agent
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor

# Load environment variables from .env
load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

# Define function to call OpenWeatherMap API
def get_weather(city: str) -> str:
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return f"Failed to retrieve weather data: {response.text}"

    data = response.json()
    weather_main = data["weather"][0]["main"]
    weather_desc = data["weather"][0]["description"]
    temp = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    humidity = data["main"]["humidity"]
    wind_speed = data["wind"]["speed"]

    formatted = (
        f"Weather in {city.title()}\n"
        f"- Condition: {weather_main} ({weather_desc})\n"
        f"- Temperature: {temp}°C (Feels like {feels_like}°C)\n"
        f"- Humidity: {humidity}%\n"
        f"- Wind Speed: {wind_speed} m/s"
    )
    return formatted

# Define tool for LangChain
weather_tool = Tool(
    name="WeatherAPI",
    description="Fetches current weather for a given city using OpenWeatherMap",
    func=get_weather
)

# Setup LLM and Agent
llm = ChatOpenAI(model="gpt-4", temperature=0)
agent = create_tool_calling_agent(llm, [weather_tool])
agent_executor = AgentExecutor(agent=agent, tools=[weather_tool], verbose=True)

# Run agent
city_input = input("Enter a city name to get weather update: ")
response = agent_executor.invoke({"input": f"What is the weather in {city_input}?"})

# Post to Slack
slack_payload = {
    "text": f"GenAI Agent Response:\n{response['output']}"
}
slack_response = requests.post(SLACK_WEBHOOK_URL, json=slack_payload)

# Slack result feedback
if slack_response.status_code == 200:
    print("Weather update sent to Slack successfully.")
else:
    print(f"Failed to send to Slack: {slack_response.text}")
