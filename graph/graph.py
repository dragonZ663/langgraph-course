from typing import Literal

from langgraph.graph import END, StateGraph

from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState
from graph.chains.hallucination_grader import hallucination_grader, GradeHallucinations
from graph.chains.answer_grader import answer_grader, GradeAnswer

NOT_SUPPORTED = "not support"
NOT_USEFUL = "not useful"
USERFUL = "useful"


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    """
    判断LLM generation是否产生了幻觉，是否解决了问题
    """
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    hallucination_res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_res.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        answer_res: GradeAnswer = answer_grader.invoke(
            {"question": question, "generation": generation}
        )
        if answer_res.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return USERFUL
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return NOT_USEFUL
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return NOT_SUPPORTED


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
build.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {USERFUL: END, NOT_USEFUL: WEBSEARCH, NOT_SUPPORTED: GENERATE},
)
build.add_edge(WEBSEARCH, GENERATE)
build.add_edge(GENERATE, END)


app = build.compile()
app.get_graph().draw_mermaid_png(output_file_path="agentic-rag.png")
