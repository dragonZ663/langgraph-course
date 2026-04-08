from dotenv import load_dotenv

load_dotenv()
import os
from typing import Any, Dict

from langchain_chroma import Chroma

from graph.llm import embedding_llm
from graph.state import GraphState

# 创建 retriever 供向量检索
retriever = Chroma(
    collection_name=os.environ.get("INDEX_NAME"),
    embedding_function=embedding_llm,
    persist_directory=os.environ.get("PERSIST_DIRECTORY"),
).as_retriever()


def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Vector Store search based on user's question.
    """
    question = state["question"]
    docs = retriever.invoke(question)

    return {"documents": docs, "question": question}
