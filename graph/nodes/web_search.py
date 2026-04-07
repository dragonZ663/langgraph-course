from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState

load_dotenv()

tavily_search = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    """
    根据用户的问题进行检索
    """
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    is_web_search = state.get("web_search", False)

    res = tavily_search.invoke({"query": question})

    result = "\n".join(item["content"] for item in res["results"])

    search_doc = Document(page_content=result)

    if documents is not None:
        documents.append(search_doc)
    else:
        documents = [search_doc]

    return {"question": question, "documents": documents, "web_search": is_web_search}


if __name__ == "__main__":
    web_search(state={"question": "agent memory", "documents": None})