from typing import Literal

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from graph.consts import RETRIEVE, WEBSEARCH
from graph.llm import chat_llm


class QuestionRouter(BaseModel):
    """对用户的问题，进行路由判断，以选择到最相关的下一步节点"""

    next_node: Literal[WEBSEARCH, RETRIEVE] = Field(
        description="下一步节点的名称, 只能是: websearch 或者 retrieve"
    )


pydantic_parser = PydanticOutputParser(pydantic_object=QuestionRouter)

system = """
你是一个路由选择器，擅长基于用户的问题，判断下一步应该选择哪个节点进行执行后续逻辑。\n\n
当用户的问题，是 agents, prompt engineering, 和 adversarial attacks(对抗性攻击)这些领域时，下一步走 retrieve 去查询向量库。
否则，其他情况时，下一步走 websearch 去联网搜索
你的回答应满足如下格式: {instruction}
"""

prompt = ChatPromptTemplate.from_messages(
    [("system", system), ("human", "用户的问题：{question}")]
).partial(instruction=pydantic_parser.get_format_instructions())

router = prompt | chat_llm | pydantic_parser
