from typing import Any, Dict

from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    web_search = False
    filtered_docs = []

    for doc in documents:
        res: GradeDocuments = retrieval_grader.invoke(
            {"question": question, "documents": doc.page_content}
        )

        if res.binary_score.lower() == "no":
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = True
            continue
        else:
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(doc)
    return {"question": question, "web_search": web_search, "documents": filtered_docs}
