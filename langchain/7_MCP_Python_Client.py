import asyncio
from fastmcp import Client

# Connect to MCP server running at localhost:8002/mcp
client = Client("http://localhost:8002/mcp")

async def call_weather_service(city: str):
    async with client:
        result = await client.call_tool("get_weather", {"city": city})
        print(result.data)  # Print only the weather information


asyncio.run(call_weather_service("Paris"))