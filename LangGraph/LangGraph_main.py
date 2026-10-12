from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from LangGraph_nodes1 import run_agent_responding, should_continue, tool_node

from langgraph.graph import END, START, MessageState, StateGraph

load_dotenv()

AgentReason = "agent_reason"
Act = "act"

flow = StateGraph(MessageState)

flow.add_node(AgentReason, run_agent_responding)
flow.add_node(Act, tool_node)

flow.add_edge(START, AgentReason)
flow.add_conditional_edges(AgentReason, should_continue, {"continue": Act, "end": END})
flow.add_edge(Act, AgentReason)

app = flow.compile()
res = app.invoke({"messages": [HumanMessage(content="What is the capital of France?")]})
print(res["messages"][-1].content)