from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import START, END, StateGraph, MessagesState
from chains import revisor, first_responder
from tool_executor import execute_tools

MAX_ITERATIONS = 2
DRAFT_NODE = "draft"
EXECUTE_NODE = "execute_tools"
REVISE_NODE = "revise"

def draft_node(state: MessagesState):
    """起草初步answer。"""
    response = first_responder.invoke({"messages": state["messages"]})
    return {"messages": [response]}

def revise_node(state: MessagesState):
    """根据工具查询结果修改answer。"""
    response = revisor.invoke({"messages": state["messages"]})
    return {"messages": [response]}

def event_loop(state: MessagesState):
    """根据迭代次数决定是否继续或结束。"""
    tool_msg_count = len([msg for msg in state["messages"] if isinstance(msg, ToolMessage) ])
    if tool_msg_count > MAX_ITERATIONS:
        return END
    else:
        return EXECUTE_NODE

builder = StateGraph(MessagesState)
builder.add_node(DRAFT_NODE, draft_node)
builder.add_node(EXECUTE_NODE, execute_tools)
builder.add_node(REVISE_NODE, revise_node)
builder.add_edge(START, DRAFT_NODE)
builder.add_edge(DRAFT_NODE, EXECUTE_NODE)
builder.add_edge(EXECUTE_NODE, REVISE_NODE)
builder.add_conditional_edges(REVISE_NODE, event_loop, [END, EXECUTE_NODE])

graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="reflexion_agent.png")

res = graph.invoke({"messages": [
    {
        "role": "user",
        "content": "撰写关于人工智能驱动的 SOC / autonomous SOC 问题领域的文章, 列出从事该领域研究并已获得融资的初创公司。"
    }
]})

# 使用工具调用从最后一条消息中提取最终答案
last_message = res["messages"][-1]
if isinstance(last_message, AIMessage) and last_message.tool_calls:
    print(last_message.tool_calls[0]["args"]["answer"])

print(res)
