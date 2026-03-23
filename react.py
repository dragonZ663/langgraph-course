from dotenv import load_dotenv

load_dotenv()

import os

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


@tool
def triple(num: float) -> float:
    """
    param num: a number to triple
    returns: the triple of the input number
    """
    return float(num) * 3


tools = [TavilySearch(max_results=1), triple]
llm = ChatOpenAI(
    model="qwen/qwen3.5-9b",
    base_url=os.environ.get("LM_STUDIO_BASE_URL"),
    api_key=os.environ.get("LM_STUDIO_API_KEY"),
    temperature=0,
).bind_tools(tools)
