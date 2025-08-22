"""
Problem Statement
Develop a LangChain-powered GenAI agent that assists quality assurance (QA) teams by automating test case generation, code debugging, and error analysis. The agent should convert user stories into executable pytest test cases, explain and fix broken Python code, and analyze error tracebacks to recommend actionable fixes—all using GPT-4 and modular LangChain tools.

Objectives
Convert user stories into testable code by generating pytest test cases using GPT-4.

Debug broken Python code by analyzing and auto-correcting it with step-by-step explanations.

Analyze runtime error logs or tracebacks and provide root cause analysis with suggested fixes.

Leverage LangChain’s tool-calling agent pattern to modularize and execute each of the above capabilities.

Enable QA engineers and developers to accelerate validation, debugging, and test automation tasks using GenAI.
"""

import os
import traceback
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# GPT-4 model for reasoning
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Tool 1: Convert user story to pytest test case
def story_to_test_case(user_story: str) -> str:
    prompt = f"""
You are a senior QA engineer. Convert the following user story into a pytest-style test function in Python.
Make sure to include necessary assertions and clear naming.

User Story:
\"\"\"{user_story}\"\"\"
"""
    response = llm.invoke(prompt)
    return response.content

# Tool 2: Explain and fix broken Python code
def fix_broken_code(code: str) -> str:
    prompt = f"""
You are a Python expert. The following code is broken. Please explain the error and return the corrected version.

Broken Code:
\"\"\"{code}\"\"\"
"""
    response = llm.invoke(prompt)
    return response.content

# Tool 3: Analyze error message and suggest a fix
def analyze_error(error_log: str) -> str:
    prompt = f"""
You are an AI debug assistant. Analyze the following Python error message or traceback and suggest the root cause and a fix.

Error Log:
\"\"\"{error_log}\"\"\"
"""
    response = llm.invoke(prompt)
    return response.content

# Wrap each function as a LangChain tool
tools = [
    Tool(
        name="TestCaseGenerator",
        description="Converts user story into pytest test case",
        func=story_to_test_case
    ),
    Tool(
        name="BrokenCodeFixer",
        description="Fixes broken Python code and explains the error",
        func=fix_broken_code
    ),
    Tool(
        name="ErrorAnalyzer",
        description="Analyzes Python error logs and suggests fixes",
        func=analyze_error
    )
]

# Create LangChain agent
agent = create_tool_calling_agent(llm=llm, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# User interface for testing the tools
print("Choose an option:")
print("1. Convert user story to test case")
print("2. Fix broken code")
print("3. Analyze error log")
choice = input("Enter your choice (1/2/3): ").strip()

if choice == "1":
    user_story = input("Enter the user story:\n")
    result = agent_executor.invoke({"input": f"Convert this user story to a pytest test: {user_story}"})
    print("\nGenerated Test Case:\n", result["output"])

elif choice == "2":
    code = input("Paste the broken Python code:\n")
    result = agent_executor.invoke({"input": f"Fix this broken code and explain the issue: {code}"})
    print("\nFix and Explanation:\n", result["output"])

elif choice == "3":
    error_log = input("Paste the error message or traceback:\n")
    result = agent_executor.invoke({"input": f"Analyze this error and suggest a fix: {error_log}"})
    print("\nAnalysis and Fix:\n", result["output"])

else:
    print("Invalid option selected.")
