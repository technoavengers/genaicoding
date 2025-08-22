import asyncio
from fastmcp import Client

# Connect to MCP server running at localhost:8000/mcp
client = Client("http://localhost:8000/mcp")

async def call_translator(text: str, target_lang: str):
    async with client:
        result = await client.call_tool("translator", {"text": text, "target_lang": target_lang})
        print(result.data)  # Print only the translated text


asyncio.run(call_translator("Hello Robo 1!", "French"))