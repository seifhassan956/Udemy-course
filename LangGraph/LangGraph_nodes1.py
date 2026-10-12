from dotenv import load_dotenv
from LangGraph_react1 import llm, tools

from langgraph.graph import MessageState
from langgraph.prebuilt import ToolNode

SystemMessage = """
    You are a helpful assistant that can use tools to assist with user queries.
"""

def run_agent_responding(state: MessageState) -> MessageState:
    response = llm.invoke([{"role": "system", "content": SystemMessage}, *state["messages"]])
    return {"messages": [response]}

def should_continue(state: MessageState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls: return "continue"
    return "end"

tool_node = ToolNode(tools)