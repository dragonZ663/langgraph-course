from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import MessagesState, StateGraph, END
from langchain_core.messages import HumanMessage
from nodes import run_agent_reasoning, tool_node

AGENT_REASON = "agent_reason"
ACT = "act"
LAST = -1

def should_continue(state: MessagesState) -> str:
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT

flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASON, run_agent_reasoning)
# 设置入口点（虚拟start -> 实际节点）
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)

# 添加条件edge(虚线部分)
flow.add_conditional_edges(AGENT_REASON, should_continue, {
    ACT: ACT,
    END: END
})
flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")

def main():
    print("Hello ReAct LangGraph with Function Calling")
    res = app.invoke({"messages": [HumanMessage(content="What's the temperature in NanJing JiangSu China? List it and then triple it")]})
    print(res["messages"][LAST].content)


if __name__ == "__main__":
    main()
