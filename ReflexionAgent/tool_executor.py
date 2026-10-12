from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode
from schemas import AnswerQuestion, RevisedAnswer
load_dotenv()

tavily_search = TavilySearch()

def run_queries(search_queries:list[str],**kwargs):
    return tavily_search.batch([{"query": query} for query in search_queries])

execute_tools = ToolNode(
    [
        StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
        StructuredTool.from_function(run_queries, name=RevisedAnswer.__name__),
    ]
)