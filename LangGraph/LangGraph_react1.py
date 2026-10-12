from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

load_dotenv()

@tool
def triple(num:float) -> float:
    """
        param num: The number to triple
        return: The tripled value of the input number
    """
    return num * 3

tools = [TavilySearch(max_results=3), triple]
llm_with_tools = ChatGroq(model="llama-3.3-70b-versatile").bind_tools(tools)