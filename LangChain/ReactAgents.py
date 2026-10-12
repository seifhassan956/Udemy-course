from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from tavily import TavilyClient

from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()
tavily_client = TavilyClient()

class Source(BaseModel):
    """Schema for a source used by the agent."""
    url: str = Field(description="The URL of the source.")

class AgentResponse(BaseModel):
    """Schema for the agent's response with answer and sources."""
    answer: str = Field(description="The answer provided by the agent.")
    sources: list[Source] = Field(default=[], description="List of sources used to generate the answer.")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash"
)

@tool
def search(query: str) -> str:
    """Search the web for current information."""
    print(f"Searching for: {query}")

    result = tavily_client.search(query)
    return str(result)

tools = [search]
agent = create_agent(llm, tools,response_format=AgentResponse)
response = agent.invoke({"messages": [HumanMessage(content="What is the weather in Tokyo?")]})
print(response)