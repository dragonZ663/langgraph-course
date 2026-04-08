from typing import Literal

from langgraph.graph import END, StateGraph

from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

build = StateGraph(GraphState)

build.add_node(RETRIEVE, retrieve)
build.add_node(GRADE_DOCUMENTS, grade_documents)
build.add_node(WEBSEARCH, web_search)
build.add_node(GENERATE, generate)


def should_web_search(state: GraphState) -> Literal[WEBSEARCH, GENERATE]:
    print("---ASSESS GRADED DOCUMENTS---")
    if state["web_search"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE


build.set_entry_point(RETRIEVE)
build.add_edge(RETRIEVE, GRADE_DOCUMENTS)
build.add_conditional_edges(
    GRADE_DOCUMENTS, should_web_search, {WEBSEARCH: WEBSEARCH, GENERATE: GENERATE}
)
build.add_edge(WEBSEARCH, GENERATE)
build.add_edge(GENERATE, END)

app = build.compile()
# app.get_graph().draw_mermaid_png(output_file_path="agentic-rag.png")
