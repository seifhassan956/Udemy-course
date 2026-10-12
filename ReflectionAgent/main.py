from typing import Annotated, TypedDict

from chains import generation_chain, reflection_chain
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage

from langgraph import END, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()

class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

REFLECT = 'reflect'
GENERATE = 'generate'

def generation_node(state: dict):
    return {"messages": [generation_chain.invoke({"messages": state["messages"]})]}

def reflection_node(state: MessageGraph):
    res = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}

build_graph = StateGraph(state_schema=MessageGraph)
build_graph.add_node(GENERATE, generation_node)
build_graph.add_node(REFLECT, reflection_node)
build_graph.set_entry_point(GENERATE)

def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6: return END
    return REFLECT

build_graph.add_conditional_edges(
    GENERATE,
    should_continue,
    path_map ={
        REFLECT: REFLECT,
        END: END
    }
)

build_graph.add_conditional_edges(REFLECT, GENERATE)
build_graph.add_edge(GENERATE, REFLECT)

app = build_graph.compile()
print(app.draw_graph().draw_mermaid())