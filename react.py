from dotenv import load_dotenv

load_dotenv()

import os

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from pydantic import SecretStr


@tool
def triple(num: float) -> float:
    """
    param num: a number to triple
    returns: the triple of the input number
    """
    return float(num) * 3


@tool
def get_cur_weather(city: str) -> str:
    """查询指定城市的当日天气

    Args:
        - city： 城市名称

    """
    return f"{city} 当前的天气为：多云转晴, 气温 26度, 湿度 50%"


search_tool = TavilySearch(max_results=3)

tools = [get_cur_weather, triple]
llm = ChatOpenAI(
    model="qwen3.5:9b",
    base_url=os.environ.get("OLLAMA_BASE_URL"),
    api_key=SecretStr(os.environ.get("OLLAMA_API_KEY", "")),
    temperature=0,
).bind_tools(tools)
