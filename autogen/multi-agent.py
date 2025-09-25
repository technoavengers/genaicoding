import os
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()

model_client = OpenAIChatCompletionClient(model="gpt-4o-mini")

planner_agent = AssistantAgent(
    name="planner_agent",
    model_client=model_client,
    description="Creates personalized travel plans.",
    system_message="You are a travel planner. Suggest a detailed itinerary based on destination, budget, and preferences."
)

budget_agent = AssistantAgent(
    name="budget_agent",
    model_client=model_client,
    description="Provides budgeting advice for travel.",
    system_message="You are a budgeting expert. Ensure the travel plan fits within the user's budget and suggest cost-saving tips."
)

activity_agent = AssistantAgent(
    name="activity_agent",
    model_client=model_client,
    description="Recommends activities and places to visit.",
    system_message="You are an activity expert. Recommend attractions, activities, and experiences for the destination."
)

group_chat = RoundRobinGroupChat(
    [planner_agent, budget_agent, activity_agent]
)

async def main():
    await Console(group_chat.run_stream(task="I want a 5-day vacation from delhi to Tokyo under $2500."))
    await model_client.close()

asyncio.run(main())