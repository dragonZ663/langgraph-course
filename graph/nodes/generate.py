from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    """
    根据检索到的上下文信息（文档块），回答用户的问题
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    answer = generation_chain.invoke({"context": documents, "question": question})

    return {"question": question, "documents": documents, "generation": answer}
