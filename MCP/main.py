
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
SERVERS_DIR = BASE_DIR / "servers"

client = MultiServerMCPClient(
    {
        "math": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(SERVERS_DIR / "math_server.py")],
        },
        "weather": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [str(SERVERS_DIR / "weatherServer.py")],
        },
    }
)

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)


async def main():
    # Get tools from all configured MCP servers
    tools = await client.get_tools()

    print("Available tools:")
    for tool in tools:
        print(f"- {tool.name}")

    # Create an agent with access to all tools
    agent = create_agent(
        model=llm,
        tools=tools,
    )

    # Test both MCP servers
    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Calculate 12 + 8 and 7 * 6 using the math tools. "
                        "Then tell me the mock weather in Cairo."
                    ),
                }
            ]
        }
    )

    print("\nAgent response:")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
