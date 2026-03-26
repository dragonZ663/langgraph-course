from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, BaseMessage
load_dotenv()
from chains import generate_chain, reflect_chain

# 定义graph的state
class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 定义node名称
REFLECT = "reflect"
GENERATE = "generate"

# 定义generate node, 返回包含 messages 字段的字典
def generate_node(state: MessageGraph):
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})] }

# 定义generate node, 返回包含 messages 字段的字典
def reflect_node(state: MessageGraph):
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}

builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generate_node)
builder.add_node(REFLECT, reflect_node)
# 设置 start -> node
builder.set_entry_point(GENERATE)

# 路由函数，用于生成 conditional edge
def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT

# 添加条件边
builder.add_conditional_edges(GENERATE, should_continue, path_map={
    END: END,
    REFLECT: REFLECT
})
# 添加固定边
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
# graph.get_graph().draw_mermaid_png(output_file_path="relect_graph.png")
print(graph.get_graph().draw_ascii())


if __name__ == "__main__":
    inputs = {
        "messages": [
            HumanMessage(
                content="""让这条推文更完善：
                @LangChainAI
                新推出的工具调用功能被严重低估了。
                经过漫长的等待，它终于来了让跨不同模型实现代理的函数调用变得超级简单。
                我制作了一个视频，介绍他们最新的博客文章。
                """
            )
        ]
    }

    response = graph.invoke(inputs)
    print(response)
